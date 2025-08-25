import os
import json
import numpy as np
from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, f1_score)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

import Utils

# 加载配置 & 路径
config = Utils.load_config()
cfg = config.get('random_forest_classifier', {})  # 兼容缺键
pe_meta_dir = cfg.get('jsonl_input_dir')
output_dir  = cfg.get('jsonl_output_dir', './output/model/rf_model_jsonl')
Utils.check_directory_exists(output_dir)
model_path  = os.path.join(output_dir, 'rf_model_jsonl.pkl')
metrics_path = os.path.join(output_dir, 'rf_metrics.json')
featimp_path = os.path.join(output_dir, 'rf_feature_importances.csv')

# 数据与训练参数
test_size    = float(cfg.get('test_size', 0.2))
val_size     = float(cfg.get('val_size', 0.2))         # 在 train 内部分出验证集
random_state = int(cfg.get('random_state', 42))

# 模型超参（保持与你现有配置风格一致）
n_estimators  = int(cfg.get('n_estimators', 300))
max_depth     = cfg.get('max_depth', None)
max_depth     = None if (max_depth in [None, "", "None"]) else int(max_depth)
n_jobs        = int(cfg.get('n_jobs', -1))
class_weight  = cfg.get('class_weight', 'balanced_subsample')
max_features  = cfg.get('max_features', 'sqrt')  # 对高维通常更稳
min_samples_leaf = int(cfg.get('min_samples_leaf', 1))
bootstrap     = bool(cfg.get('bootstrap', True))
oob_score     = bool(cfg.get('oob_score', False))  # 若开，bootstrap 必须 True
criterion     = cfg.get('criterion', 'gini')       # 或 'entropy','log_loss'

# 阈值调优策略（可在配置中切换）
# 可选：'f1', 'youden', 'pr_auc_hold'（保持默认0.5，仅做PR-AUC评估）
threshold_strategy = cfg.get('threshold_strategy', 'f1')

def best_threshold_by_strategy(y_true, prob, strategy: str = 'f1') -> float:
    """基于验证集选择最优阈值。"""
    if strategy == 'pr_auc_hold':
        return 0.5
    p, r, t = precision_recall_curve(y_true, prob)
    if strategy == 'youden':
        # 用 ROC 思路的替代：近似用 F1 的阈值，但也可改为手动扫阈值最大化 (TPR - FPR)
        best_thr, best_score = 0.5, -1.0
        for thr in np.linspace(0.01, 0.99, 99):
            pred = (prob >= thr).astype(int)
            # 近似 Youden: 用 F1 作为代理（或自行实现 TPR-FPR）
            score = f1_score(y_true, pred, zero_division=0)
            if score > best_score:
                best_score, best_thr = score, thr
        return best_thr
    # 默认 F1
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        pred = (prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


# 训练与保存
def main():

    # 1) 读取数据
    print(f"[INFO] Loading JSONL from: {pe_meta_dir}")
    X, y = Utils.load_dataset_from_jsonl_dir(pe_meta_dir)
    y_bin = np.where(y == 1, 1, 0).astype(int) # 忽略掉label作为-1的情况

    feat_names = Utils.feature_names()

    # 2) 三段划分
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y_bin
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=val_size, random_state=random_state, stratify=y_train_all
    )

    # 3) 流水线：缺失值兜底 + RF
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            bootstrap=bootstrap,
            oob_score=oob_score and bootstrap,
            criterion=criterion,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        ))
    ])

    # 4) 训练
    pipe.fit(X_train, y_train)
    rf = pipe.named_steps["rf"]

    # 5) 验证集阈值调优
    y_val_prob = pipe.predict_proba(X_val)[:, 1]
    best_thr = best_threshold_by_strategy(y_val, y_val_prob, threshold_strategy)

    # 6) 测试集评估（使用最佳阈值）
    y_test_prob = pipe.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_thr).astype(int)

    # 常规指标
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    report = classification_report(y_test, y_test_pred, labels=[0, 1],
                                   target_names=["Benign", "Malware"], zero_division=0)
    # 曲线型指标
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
    if oob_score and bootstrap:
        print(f"[OOB Score] {rf.oob_score_:.4f}")

    # 7) 保存模型（含流水线 + 阈值）
    payload: Dict[str, Any] = {
        "pipeline": pipe,
        "feature_names": feat_names,
        "meta": {
            "version": "rf_from_jsonl_v2",
            "feature_dim": int(X.shape[1]),
            "test_size": float(test_size),
            "val_size": float(val_size),
            "random_state": int(random_state),
            "n_estimators": int(n_estimators),
            "max_depth": (None if max_depth is None else int(max_depth)),
            "max_features": max_features,
            "min_samples_leaf": int(min_samples_leaf),
            "class_weight": class_weight,
            "bootstrap": bool(bootstrap),
            "oob_score": bool(oob_score and bootstrap),
            "criterion": criterion,
            "best_threshold": float(best_thr),
            "threshold_strategy": threshold_strategy
        }
    }
    joblib.dump(payload, model_path, compress=3)
    print(f"[SAVE] model -> {model_path}")

    # 8) 保存评估指标到 JSON
    metrics = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "oob_score": (float(rf.oob_score_) if (oob_score and bootstrap) else None),
        "best_threshold": float(best_thr),
        "threshold_strategy": threshold_strategy,
        "pos_rate_test": float(np.mean(y_test)),
        "pos_rate_train": float(np.mean(y_train_all))
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] metrics -> {metrics_path}")

    # 9) 特征重要性落盘
    #   注意：重要性对应的是“Imputer 输出后的列顺序”，与 feat_names 对齐
    try:
        importances = rf.feature_importances_
        with open(featimp_path, "w", encoding="utf-8") as f:
            f.write("feature,importance\n")
            for name, imp in zip(feat_names, importances):
                f.write(f"{name},{imp}\n")
        print(f"[SAVE] feature importances -> {featimp_path}")
    except Exception as e:
        print(f"[WARN] save feature importances failed: {e}")

if __name__ == "__main__":
    main()
    Utils.notice_bark('随机森林模型（JSONL）训练完毕！（含缺失值兜底 & 阈值调优 & 指标落盘）')