import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

# 1. 加载图像数据集
X, y = Utils.load_images_and_labels(image_dir)

# 2. 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# === 模型训练 ===
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# === 保存模型 ===
joblib.dump(rf, model_path)
print(f"[✔] 模型已保存至: {model_path}")

# === 模型评估 ===
y_pred = rf.predict(X_test)

print("\n[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\n[Classification Report]")
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Benign", "Malware"]))

Utils.notice_bark('随机森林模型训练完毕！')