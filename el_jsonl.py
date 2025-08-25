import os
import json
import numpy as np
from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, f1_score)
from sklearn.utils.class_weight import compute_sample_weight
import joblib

import Utils

# 加载配置 & 路径
config = Utils.load_config()
cfg = config.get('ensemble_classifier_jsonl', {})

pe_meta_dir = cfg.get('jsonl_input_dir')
output_dir  = cfg.get('jsonl_output_dir', './output/model/ensemble_model_jsonl')
Utils.check_directory_exists(output_dir)

voting_path    = os.path.join(output_dir, 'voting_ensemble_jsonl.pkl')
stacking_path  = os.path.join(output_dir, 'stacking_ensemble_jsonl.pkl')
voting_metrics = os.path.join(output_dir, 'voting_metrics.json')
stack_metrics  = os.path.join(output_dir, 'stacking_metrics.json')

# 通用
test_size    = float(cfg.get('test_size', 0.2))
val_size     = float(cfg.get('val_size', 0.2))     # 在 train 内再切 val
random_state = int(cfg.get('random_state', 42))
threshold_strategy = cfg.get('threshold_strategy', 'f1')  # f1 | youden | pr_auc_hold

# RF 超参
rf_cfg = cfg.get('rf', {})
rf_n_estimators = int(rf_cfg.get('n_estimators', 300))
rf_max_depth    = rf_cfg.get('max_depth', None)
rf_max_depth    = None if rf_max_depth in [None, "", "None"] else int(rf_max_depth)
rf_n_jobs       = int(rf_cfg.get('n_jobs', -1))
rf_class_weight = rf_cfg.get('class_weight', 'balanced_subsample')
rf_max_features = rf_cfg.get('max_features', 'sqrt')
rf_min_samples_leaf = int(rf_cfg.get('min_samples_leaf', 1))
rf_bootstrap    = bool(rf_cfg.get('bootstrap', True))
rf_criterion    = rf_cfg.get('criterion', 'gini')

# SVM 超参
svm_cfg = cfg.get('svm', {})
svm_C           = float(svm_cfg.get('C', 1.0))
svm_max_iter    = int(svm_cfg.get('max_iter', 2000))
svm_class_weight= svm_cfg.get('class_weight', 'balanced')
svm_kernel      = svm_cfg.get('kernel', 'linear')   # 建议 linear
svm_gamma       = svm_cfg.get('gamma', 'scale')     # rbf 时有效
svm_cache_mb    = float(svm_cfg.get('cache_size_mb', 1024))

# MLP 超参
mlp_cfg = cfg.get('mlp', {})
mlp_hidden_sizes   = tuple(mlp_cfg.get('hidden_layer_sizes', [256, 128]))
mlp_max_iter       = int(mlp_cfg.get('max_iter', 200))
mlp_alpha          = float(mlp_cfg.get('alpha', 1e-4))
mlp_early_stopping = bool(mlp_cfg.get('early_stopping', True))
mlp_val_frac       = float(mlp_cfg.get('validation_fraction', 0.1))
mlp_learning_rate  = mlp_cfg.get('learning_rate', 'adaptive')
mlp_batch_size     = mlp_cfg.get('batch_size', 'auto')
mlp_lr_init        = float(mlp_cfg.get('learning_rate_init', 1e-3))

# Voting/Stacking 其他
voting_weights = cfg.get('voting_weights', [2, 1, 1])  # 对应 [rf, svm, mlp]

# 阈值调优
def best_threshold_by_strategy(y_true, prob, strategy: str = 'f1') -> float:
    if strategy == 'pr_auc_hold':
        return 0.5
    cand = np.linspace(0.01, 0.99, 99)
    if strategy == 'youden':
        # 简化为用 F1 近似，也可自行实现 TPR-FPR 扫描
        f1s = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in cand]
        return float(cand[int(np.argmax(f1s))])
    # 默认 F1
    f1s = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in cand]
    return float(cand[int(np.argmax(f1s))])

# 基学习器流水线
def make_rf(rs: int):
    # RF 不能接受 NaN → 前置 Imputer；无需标准化
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('rf', RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            max_features=rf_max_features,
            min_samples_leaf=rf_min_samples_leaf,
            class_weight=rf_class_weight,
            bootstrap=rf_bootstrap,
            criterion=rf_criterion,
            random_state=rs,
            n_jobs=rf_n_jobs
        ))
    ])

def make_svm(rs: int):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('svc', SVC(
            kernel=svm_kernel,
            C=svm_C,
            gamma=(svm_gamma if svm_kernel != 'linear' else 'scale'),
            probability=True,
            class_weight=svm_class_weight,
            cache_size=svm_cache_mb,
            random_state=rs,
            max_iter=svm_max_iter
        ))
    ])

def make_mlp(rs: int):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=mlp_hidden_sizes,
            activation='relu',
            solver='adam',
            alpha=mlp_alpha,
            batch_size=mlp_batch_size,
            learning_rate=mlp_learning_rate,
            learning_rate_init=mlp_lr_init,
            max_iter=mlp_max_iter,
            early_stopping=mlp_early_stopping,
            validation_fraction=mlp_val_frac,
            random_state=rs,
            verbose=False
        ))
    ])

# 构建集成器
def build_voting(rs=42):
    return VotingClassifier(
        estimators=[('rf', make_rf(rs)), ('svm', make_svm(rs)), ('mlp', make_mlp(rs))],
        voting='soft',
        weights=voting_weights,
        n_jobs=None,  # Sklearn 会并行到子模型内部
        flatten_transform=True
    )

def build_stacking(rs=42):
    meta = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=rs)
    return StackingClassifier(
        estimators=[('rf', make_rf(rs)), ('svm', make_svm(rs)), ('mlp', make_mlp(rs))],
        final_estimator=meta,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=None
    )

# 训练&评估&落盘（通用器）
def fit_eval_save(est, name: str,
                  X_train, y_train, X_val, y_val, X_test, y_test,
                  model_path: str, metrics_path: str,
                  feat_names, rs: int):
    # 类不平衡：计算并传入 sample_weight
    sw = compute_sample_weight(class_weight='balanced', y=y_train)
    est.fit(X_train, y_train, **({'sample_weight': sw} if 'Voting' in est.__class__.__name__ or 'Stacking' in est.__class__.__name__ else {}))

    # 验证集调阈
    y_val_prob = est.predict_proba(X_val)[:, 1]
    best_thr = best_threshold_by_strategy(y_val, y_val_prob, threshold_strategy)

    # 测试集评估
    y_test_prob = est.predict_proba(X_test)[:, 1]
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

    print(f"\n[{name}] Confusion Matrix (labels=[0,1])\n{cm}")
    print(f"\n[{name}] Classification Report\n{report}")
    print(f"[{name}] ROC-AUC: {roc:.4f}" if roc is not None else f"[{name}] ROC-AUC: N/A")
    print(f"[{name}] PR-AUC : {pr_auc:.4f}" if pr_auc is not None else f"[{name}] PR-AUC : N/A")

    # 落盘：模型 + 元信息
    payload: Dict[str, Any] = {
        "model": est,
        "feature_names": feat_names,
        "meta": {
            "version": f"{name.lower()}_jsonl_v2",
            "random_state": rs,
            "test_size": float(test_size),
            "val_size": float(val_size),
            "threshold_strategy": threshold_strategy,
            "best_threshold": float(best_thr),
            "rf": {
                "n_estimators": rf_n_estimators,
                "max_depth": rf_max_depth,
                "max_features": rf_max_features,
                "min_samples_leaf": rf_min_samples_leaf,
                "class_weight": rf_class_weight,
                "bootstrap": rf_bootstrap,
                "criterion": rf_criterion
            },
            "svm": {
                "kernel": svm_kernel, "C": svm_C, "gamma": (svm_gamma if svm_kernel != 'linear' else None),
                "max_iter": svm_max_iter, "class_weight": svm_class_weight, "cache_size_mb": svm_cache_mb
            },
            "mlp": {
                "hidden_layer_sizes": list(mlp_hidden_sizes),
                "alpha": mlp_alpha,
                "max_iter": mlp_max_iter,
                "early_stopping": mlp_early_stopping,
                "validation_fraction": mlp_val_frac,
                "learning_rate": mlp_learning_rate,
                "batch_size": mlp_batch_size,
                "learning_rate_init": mlp_lr_init
            },
            "voting_weights": voting_weights if isinstance(est, VotingClassifier) else None
        }
    }
    joblib.dump(payload, model_path, compress=3)
    print(f"[SAVE] {name} -> {model_path}")

    # 落盘：指标
    metrics = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "best_threshold": float(best_thr),
        "threshold_strategy": threshold_strategy,
        "pos_rate_test": float(np.mean(y_test)),
        "pos_rate_train": float(np.mean(y_train))
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {name} metrics -> {metrics_path}")

# 主流程
def main():
    print(f"[INFO] Loading JSONL from: {pe_meta_dir}")
    X, y = Utils.load_dataset_from_jsonl_dir(pe_meta_dir)
    feat_names = Utils.feature_names()

    # 三段划分
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=val_size, random_state=random_state, stratify=y_train_all
    )

    # Voting
    voting = build_voting(rs=random_state)
    fit_eval_save(voting, "Voting", X_train, y_train, X_val, y_val, X_test, y_test,
                  voting_path, voting_metrics, feat_names, random_state)

    # Stacking
    stacking = build_stacking(rs=random_state)
    fit_eval_save(stacking, "Stacking", X_train, y_train, X_val, y_val, X_test, y_test,
                  stacking_path, stack_metrics, feat_names, random_state)


if __name__ == "__main__":
    main()
    Utils.notice_bark('集成模型（JSONL）训练完毕！（Voting + Stacking，含缺失值兜底 & 类权重 & 阈值调优 & 指标落盘）')