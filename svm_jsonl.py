import os
import json
import glob
import numpy as np
from typing import List, Dict, Any, Tuple

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, f1_score)
import joblib

import Utils

# 加载配置 & 路径
config = Utils.load_config()
cfg = config.get('svm_classifier_jsonl', {})

pe_meta_dir = cfg.get('jsonl_input_dir')
output_dir  = cfg.get('jsonl_output_dir', './output/model/svm_model_jsonl')
Utils.check_directory_exists(output_dir)

model_path    = os.path.join(output_dir, 'svm_model_jsonl.pkl')
metrics_path  = os.path.join(output_dir, 'svm_metrics.json')
featcoef_path = os.path.join(output_dir, 'svm_linear_coef.csv')  # 仅 linear 核写出

# 评估/随机种子/模型超参
test_size     = float(cfg.get('test_size', 0.2))
val_size      = float(cfg.get('val_size', 0.2))      # 在 train 内部再切 val
random_state  = int(cfg.get('random_state', 42))
C             = float(cfg.get('C', 1.0))
max_iter      = int(cfg.get('max_iter', 2000))
class_weight  = cfg.get('class_weight', 'balanced')
kernel        = cfg.get('kernel', 'linear')          # linear | rbf
gamma         = cfg.get('gamma', 'scale')            # rbf 时常用：'scale' | 'auto' | 数值
cache_size    = float(cfg.get('cache_size_mb', 1024))  # SVC 内部核缓存
probability   = bool(cfg.get('probability', True))   # 为阈值调优和 PR/ROC 需要
threshold_strategy = cfg.get('threshold_strategy', 'f1')  # f1 | youden | pr_auc_hold

def best_threshold_by_strategy(y_true, prob, strategy: str = 'f1') -> float:
    if strategy == 'pr_auc_hold':
        return 0.5
    # 用验证集扫阈值
    cand = np.linspace(0.01, 0.99, 99)
    if strategy == 'youden':
        # 近似用 F1 作为代理（或自行实现 TPR-FPR）
        scores = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in cand]
        return float(cand[int(np.argmax(scores))])
    # 默认 F1
    scores = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in cand]
    return float(cand[int(np.argmax(scores))])

# 训练与保存
def main():

    # 1) 读取数据
    print(f"[INFO] Loading JSONL from: {pe_meta_dir}")
    X, y = Utils.load_dataset_from_jsonl_dir(pe_meta_dir)
    feat_names = Utils.feature_names()

    # 2) 三段划分
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=val_size, random_state=random_state, stratify=y_train_all
    )

    # 标准化 + 线性核 SVM（带概率输出与类不平衡处理）
    svm = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('svc', SVC(kernel=kernel,
                    C=C,
                    gamma=(gamma if kernel != 'linear' else 'scale'),
                    probability=probability,
                    class_weight=class_weight,
                    cache_size=cache_size,
                    random_state=random_state,
                    max_iter=max_iter))
    ])


# 训练
    svm.fit(X_train, y_train)

    # 验证集阈值调优（需要 probability=True）
    if probability:
        y_val_prob = svm.predict_proba(X_val)[:, 1]
        best_thr = best_threshold_by_strategy(y_val, y_val_prob, threshold_strategy)
    else:
        # 没有概率输出就退化为 0.5
        best_thr = 0.5

    # 测试集评估
    if probability:
        y_test_prob = svm.predict_proba(X_test)[:, 1]
    else:
        # 无概率时用 decision_function 过 sigmoid 会更稳，这里简化直接映射
        # 但为了统一，仍给出一个伪概率：>0 -> 1，否则 0
        scores = svm.decision_function(X_test)
        y_test_prob = (scores - scores.min()) / (scores.ptp() + 1e-9)

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
        "pipeline": svm,
        "feature_names": feat_names,
        "meta": {
            "version": "svm_from_jsonl_v2",
            "feature_dim": int(X.shape[1]),
            "test_size": float(test_size),
            "val_size": float(val_size),
            "random_state": int(random_state),
            "C": float(C),
            "max_iter": int(max_iter),
            "class_weight": class_weight,
            "kernel": kernel,
            "gamma": gamma if kernel != 'linear' else None,
            "probability": bool(probability),
            "cache_size_mb": float(cache_size),
            "best_threshold": float(best_thr),
            "threshold_strategy": threshold_strategy
        }
    }
    joblib.dump(payload, model_path, compress=3)
    print(f"[SAVE] model -> {model_path}")

    # 保存评估指标
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

    # 线性核时，输出特征权重（便于解释）
    try:
        if kernel == 'linear':
            # 从 pipeline 里把线性 SVC 的 coef_ 取出来
            svc = svm.named_steps['svc']
            # 二分类时 shape 为 (1, n_features)
            coef = svc.coef_.ravel()
            with open(featcoef_path, "w", encoding="utf-8") as f:
                f.write("feature,weight\n")
                for name, w in zip(feat_names, coef):
                    f.write(f"{name},{w}\n")
            print(f"[SAVE] linear SVM feature weights -> {featcoef_path}")
    except Exception as e:
        print(f"[WARN] save linear coef failed: {e}")

if __name__ == "__main__":
    main()
    Utils.notice_bark('SVM（JSONL）训练完毕！（含缺失值兜底 & 阈值调优 & 指标落盘）')

