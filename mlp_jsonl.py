import os
import json
import numpy as np
from typing import Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, f1_score)
from sklearn.utils.class_weight import compute_sample_weight
import joblib

import Utils

# 加载配置 & 路径
config = Utils.load_config()
cfg = config.get('mlp_classifier_jsonl', {})

pe_meta_dir   = cfg.get('input_dir')
output_dir    = cfg.get('output_dir', './output/model/mlp_model_jsonl')
Utils.check_directory_exists(output_dir)

model_path    = os.path.join(output_dir, 'mlp_model_jsonl.pkl')
metrics_path  = os.path.join(output_dir, 'mlp_metrics.json')
losscurve_csv = os.path.join(output_dir, 'mlp_loss_curve.csv')

# 划分 & 随机数
test_size     = float(cfg.get('test_size', 0.2))
val_size      = float(cfg.get('val_size', 0.2))  # 在 train 内再切 val
random_state  = int(cfg.get('random_state', 42))

# MLP 超参（与 config.yaml 一一对应）
hidden_layer_sizes = tuple(cfg.get('hidden_layer_sizes', [256, 128]))
activation   = cfg.get('activation', 'relu')   # 'relu' | 'tanh' | 'logistic' | 'identity'
solver       = cfg.get('solver', 'adam')       # 'adam' | 'sgd' | 'lbfgs'
alpha        = float(cfg.get('alpha', 1e-4))
max_iter     = int(cfg.get('max_iter', 200))
batch_size   = cfg.get('batch_size', 'auto')   # 'auto' 或 正整数
learning_rate = cfg.get('learning_rate', 'adaptive')  # 'constant' | 'invscaling' | 'adaptive'
learning_rate_init = float(cfg.get('learning_rate_init', 0.001))
early_stopping = bool(cfg.get('early_stopping', True))
validation_fraction = float(cfg.get('validation_fraction', 0.1))
n_iter_no_change = int(cfg.get('n_iter_no_change', 10))
tol          = float(cfg.get('tol', 1e-4))
shuffle      = bool(cfg.get('shuffle', True))
beta_1       = float(cfg.get('beta_1', 0.9))
beta_2       = float(cfg.get('beta_2', 0.999))
epsilon      = float(cfg.get('epsilon', 1e-8))

# 类不平衡策略：'balanced' | None
class_weight_mode = cfg.get('class_weight', 'balanced')

# 阈值调优策略：'f1' | 'youden' | 'pr_auc_hold'
threshold_strategy = cfg.get('threshold_strategy', 'f1')

# 阈值调优
def best_threshold_by_strategy(y_true, prob, strategy: str = 'f1') -> float:
    if strategy == 'pr_auc_hold':
        return 0.5
    candidates = np.linspace(0.01, 0.99, 99)
    if strategy == 'youden':
        # 简化用 F1 近似（或可实现 TPR-FPR 扫描）
        scores = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in candidates]
        return float(candidates[int(np.argmax(scores))])
    # 默认：最大化 F1
    scores = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])

# 训练与保存
def main():

    print(f"[INFO] Loading JSONL from: {pe_meta_dir}")
    # ★ 复用 Utils：避免重复代码
    X, y = Utils.load_dataset_from_jsonl_dir(pe_meta_dir)
    y_bin = np.where(y == 1, 1, 0).astype(int) # 忽略掉label作为-1的情况

    feat_names = Utils.feature_names()

    # 三段划分
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y_bin
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=val_size, random_state=random_state, stratify=y_train_all
    )

    # 类不平衡 → sample_weight（仅用于训练集）
    if class_weight_mode == 'balanced':
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    else:
        sample_weight = None

    # 流水线：缺失值兜底 + 标准化 + MLP
    mlp_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,            # 内部会再切出一部分 val（用于早停）
            validation_fraction=validation_fraction,  # 仅在 early_stopping=True 时使用
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            shuffle=shuffle,
            random_state=random_state,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            verbose=False
        ))
    ])

    # 训练
    mlp_pipe.fit(X_train, y_train,
                 **({'mlp__sample_weight': sample_weight} if sample_weight is not None else {}))

    # 验证集阈值调优（使用 predict_proba）
    y_val_prob = mlp_pipe.predict_proba(X_val)[:, 1]
    best_thr = best_threshold_by_strategy(y_val, y_val_prob, threshold_strategy)

    # 测试集评估
    y_test_prob = mlp_pipe.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_thr).astype(int)

    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    report = classification_report(y_test, y_test_pred, labels=[0, 1],
                                   target_names=["Benign", "Malware"], zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_test_prob)
    except Exception:
        roc = None
    try:
        pr_auc = average_precision_score(y_test, y_test_prob)
    except Exception:
        pr_auc = None

    print("\n[Confusion Matrix] labels=[0,1]\n", cm)
    print("\n[Classification Report]\n", report)
    print(f"[ROC-AUC] {roc:.4f}" if roc is not None else "[ROC-AUC] N/A")
    print(f"[PR-AUC ] {pr_auc:.4f}" if pr_auc is not None else "[PR-AUC ] N/A")

    # 保存模型（含流水线 + 阈值）
    payload = {
        "pipeline": mlp_pipe,
        "feature_names": feat_names,
        "meta": {
            "version": "mlp_from_jsonl_v2",
            "feature_dim": int(X.shape[1]),
            "test_size": float(test_size),
            "val_size": float(val_size),
            "random_state": int(random_state),
            "hidden_layer_sizes": list(hidden_layer_sizes),
            "activation": activation,
            "solver": solver,
            "alpha": float(alpha),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "learning_rate_init": float(learning_rate_init),
            "max_iter": int(max_iter),
            "early_stopping": bool(early_stopping),
            "validation_fraction": float(validation_fraction),
            "n_iter_no_change": int(n_iter_no_change),
            "tol": float(tol),
            "shuffle": bool(shuffle),
            "class_weight": class_weight_mode,
            "best_threshold": float(best_thr),
            "threshold_strategy": threshold_strategy
        }
    }
    joblib.dump(payload, model_path, compress=3)
    print(f"[SAVE] model -> {model_path}")

    # 保存评估指标到 JSON
    metrics = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "best_threshold": float(best_thr),
        "threshold_strategy": threshold_strategy,
        "pos_rate_test": float(np.mean(y_test)),
        "pos_rate_train": float(np.mean(y_train_all))
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] metrics -> {metrics_path}")

    # 训练曲线落盘（若可用）
    try:
        mlp = mlp_pipe.named_steps['mlp']
        if getattr(mlp, "loss_curve_", None) is not None:
            with open(losscurve_csv, "w", encoding="utf-8") as f:
                f.write("epoch,loss\n")
                for i, loss in enumerate(mlp.loss_curve_, 1):
                    f.write(f"{i},{loss}\n")
            print(f"[SAVE] loss curve -> {losscurve_csv}")
    except Exception as e:
        print(f"[WARN] save loss curve failed: {e}")

if __name__ == "__main__":
    main()
    Utils.notice_ntfy('MLP 模型（JSONL）训练完毕！（含缺失值兜底 & 类权重 & 阈值调优 & 指标/曲线落盘）')