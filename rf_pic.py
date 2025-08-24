import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import Utils  # 你已有的配置模块

# 加载配置
config = Utils.load_config()
image_dir = config['random_forest_classifier']['pic_input_dir']
Utils.check_directory_exists(image_dir)
image_size = config['three_gram_byte_plot']['image_size']  # 图像大小为 image_size × image_size × 3
output_dir = config['random_forest_classifier']['pic_output_dir']
Utils.check_directory_exists(output_dir)
model_path = os.path.join(output_dir, 'rf_model.pkl')  # 训练模型保存路径
test_size = config['random_forest_classifier']['test_size']
random_state = config['random_forest_classifier']['random_state']
n_estimators = config['random_forest_classifier'].get('n_estimators', 100)
n_jobs = config['random_forest_classifier'].get('n_jobs', -1)

# 1. 建索引 + 划分
tr_paths, te_paths, tr_labels, te_labels = Utils.load_or_build_index_split(
    image_dir=image_dir,
    test_size=test_size,
    random_state=random_state,
    output_dir=output_dir
)

# 2. 为训练集与测试集分别构建memmap（避免一次性像素进内存）
Xtr_mm, ytr_mm = Utils.load_or_build_memmap(
    paths=tr_paths, labels=tr_labels, image_size=image_size,
    out_X_path=os.path.join(output_dir, 'X_train.mmap'),
    out_y_path=os.path.join(output_dir, 'y_train.mmap'),
    dtype=np.uint8, batch_size=1024
)

Xte_mm, yte_mm = Utils.load_or_build_memmap(
    paths=te_paths, labels=te_labels, image_size=image_size,
    out_X_path=os.path.join(output_dir, 'X_test.mmap'),
    out_y_path=os.path.join(output_dir, 'y_test.mmap'),
    dtype=np.uint8, batch_size=1024
)

# 3. 初始化随机森林并训练
rf = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=random_state,
    n_jobs=n_jobs
)
rf.fit(Xtr_mm, ytr_mm)

# 4. 保存模型
joblib.dump(rf, model_path)
print(f"[✔] 模型已保存至: {model_path}")

# 5. 预测并评估
y_pred = rf.predict(Xte_mm)

print("\n[Confusion Matrix]")
print(confusion_matrix(yte_mm, y_pred, labels=[0, 1]))

print("\n[Classification Report]")
print(classification_report(yte_mm, y_pred, labels=[0, 1], target_names=["Benign", "Malware"]))

Utils.notice_bark('随机森林模型（彩色图像）训练完毕！')