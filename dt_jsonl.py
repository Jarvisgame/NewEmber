import os
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

import Utils

# 加载配置
config = Utils.load_config()
pe_meta_dir = config['decision_tree_classifier']['jsonl_input_dir']
Utils.check_directory_exists(pe_meta_dir)
output_dir = config['decision_tree_classifier']['jsonl_output_dir']
Utils.check_directory_exists(output_dir)
model_path = os.path.join(output_dir, 'dt_model_jsonl.pkl')

# 评估/随机种子
test_size = config['decision_tree_classifier']['test_size']
random_state = config['decision_tree_classifier']['random_state']
val_size = config['decision_tree_classifier']['val_size']

# 训练与保存
def main():

    # 1. 加载JSONL数据集
    print(f"[INFO] Loading JSONL from: {pe_meta_dir}")
    # 要求 Utils.load_dataset_from_jsonl_dir 返回 (X: np.ndarray, y: np.ndarray)
    X, y = Utils.load_dataset_from_jsonl_dir(pe_meta_dir)
    feat_names = Utils.feature_names()  # 保证与 X 列顺序一致

    # 2. 划分训练集与测试集
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=val_size, random_state=random_state, stratify=y_train_all
    )

    y_train_all = np.asarray(y_train_all).ravel()
    y_train     = np.asarray(y_train).ravel()
    y_val       = np.asarray(y_val).ravel()
    y_test      = np.asarray(y_test).ravel()

    # 同时也把概率向量确保为一维（通常本来就是）
    y_val_prob  = np.asarray(y_val_prob).ravel()
    y_test_prob = np.asarray(y_test_prob).ravel()

    # 3. 基线流水线：Imputer + DecisionTree
    def make_pipeline(ccp_alpha=0.0, max_depth=None):
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("dt", DecisionTreeClassifier(
                random_state=random_state,
                class_weight="balanced",      # 类不平衡更稳
                max_depth=max_depth,          # 适度限制深度（如 None/20/30）
                min_samples_leaf=5,           # 叶子样本最小数量
                max_features=None,            # 也可试 "sqrt"/"log2"
                ccp_alpha=ccp_alpha           # 成本复杂度剪枝
            ))
        ])

    # 1) 先用未剪枝模型训练，计算剪枝路径
    base_pipe = make_pipeline(ccp_alpha=0.0)
    base_pipe.fit(X_train, y_train)
    dt = base_pipe.named_steps["dt"]
    path = dt.cost_complexity_pruning_path(base_pipe.named_steps["imp"].transform(X_train), y_train)
    alphas = path.ccp_alphas

    # 2) 在验证集上网格尝试若干 alpha（可适当取子集，以防过多）
    #    同时可试几个 max_depth，取表现最优者
    candidate_depths = [None, 20, 30]
    candidate_alphas = np.unique(np.linspace(0, alphas.max(), num=min(20, len(alphas))))  # 取20个点
    best_cfg = None
    best_pr_auc = -1.0

    for d in candidate_depths:
        for a in candidate_alphas:
            pipe = make_pipeline(ccp_alpha=a, max_depth=d)
            pipe.fit(X_train, y_train)
            # 用验证集评估（对不均衡更敏感）
            y_val_prob = pipe.predict_proba(X_val)[:, 1]
            pr_auc = average_precision_score(y_val, y_val_prob)
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_cfg = {"ccp_alpha": float(a), "max_depth": (None if d is None else int(d)), "pr_auc": float(pr_auc)}

    # 3) 用最优 alpha / depth 在  (train+val) 上重训
    best_pipe = make_pipeline(ccp_alpha=best_cfg["ccp_alpha"], max_depth=best_cfg["max_depth"])
    best_pipe.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

    # 4) 测试集评估
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    print("Confusion Matrix (labels=[0,1]):\n",
          confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, labels=[0, 1], target_names=["Benign", "Malware"]))
    try:
        roc = roc_auc_score(y_test, y_prob)
        pr  = average_precision_score(y_test, y_prob)
        print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")
    except Exception as e:
        print(f"[WARN] AUC calc failed: {e}")

    # 5) 保存模型与元数据
    payload = {
        "pipeline": best_pipe,                    # 包含 Imputer + DT
        "feature_names": feat_names,
        "meta": {
            "version": "dt_from_jsonl_v2_pruned",
            "feature_dim": int(X.shape[1]),
            "test_size": float(test_size),
            "val_size": float(val_size),
            "random_state": int(random_state),
            "best_cfg": best_cfg
        }
    }
    joblib.dump(payload, model_path, compress=3)
    print(f"[SAVE] model -> {model_path}")

if __name__ == "__main__":
    main()
    Utils.notice_bark('决策树模型（JSONL）训练完毕！')
