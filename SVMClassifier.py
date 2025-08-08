import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import Utils

# === 加载配置 ===
config = Utils.load_config()
image_dir = config['three_gram_byte_plot']['output_dir']
image_size = config['three_gram_byte_plot']['image_size']
output_dir = config['svm_classifier']['output_dir']
Utils.check_directory_exists(output_dir)
model_path = os.path.join(image_dir, 'svm_model.pkl')  # 模型保存路径

def load_images_and_labels(image_dir):
    X, y = [], []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                img = Image.open(full_path).convert('RGB')
                img = img.resize((image_size, image_size))
                img_array = np.array(img, dtype=np.uint8).flatten()
                X.append(img_array)
                label = 1 if 'VirusShare' in root else 0
                y.append(label)
    return np.array(X), np.array(y)

# === 读取数据 ===
X, y = load_images_and_labels(image_dir)

# === 数据划分 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 初始化并训练 SVM 模型（使用线性核，适合高维数据）===
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# === 保存模型 ===
joblib.dump(svm, model_path)
print(f"[✔] SVM 模型已保存至: {model_path}")

# === 模型评估 ===
y_pred = svm.predict(X_test)
print("\n[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Benign", "Malware"]))

Utils.notice_bark('支持向量机训练完毕！')