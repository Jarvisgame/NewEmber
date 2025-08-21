import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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

# 训练与保存
def main():

    # 1. 加载JSONL数据集
    print(f"[INFO] Loading JSONL from: {pe_meta_dir}")
    X, y = Utils.load_dataset_from_jsonl_dir(pe_meta_dir)

    # 2. 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. 初始化决策树并训练
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    # 4. 保存模型
    payload = {
        "model": clf,
        "feature_names": Utils.feature_names(),
        "meta": {
            "version": "dt_from_jsonl_v1",
            "feature_dim": X.shape[1],
            "test_size": test_size,
            "random_state": random_state,
        }
    }
    joblib.dump(payload, model_path)
    print(f"[SAVE] model -> {model_path}")

    # 5. 预测并评估
    y_pred = clf.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, labels=[0, 1], target_names=["Benign", "Malware"]))

if __name__ == "__main__":
    main()
    Utils.notice_ntfy('决策树模型（JSONL）训练完毕！')
