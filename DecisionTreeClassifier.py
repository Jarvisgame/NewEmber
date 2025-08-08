import os
import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import Utils  # 你原来的配置加载模块

# 加载配置
config = Utils.load_config()
image_dir = config['three_gram_byte_plot']['output_dir']
image_size = config['three_gram_byte_plot']['image_size']  # 图像为 image_size × image_size × 3
output_dir = config['decision_tree_classifier']['output_dir']
Utils.check_directory_exists(output_dir)
model_path = os.path.join(output_dir, 'dt_model.pkl')  # 训练模型保存路径

def load_images_and_labels(image_dir):
    X = []
    y = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)

                # 加载图像为RGB并展平
                img = Image.open(full_path).convert('RGB')
                img = img.resize((image_size, image_size))  # 保证尺寸一致
                img_array = np.array(img, dtype=np.uint8).flatten()

                X.append(img_array)

                # 标签判断：路径中是否包含 'VirusShare'
                label = 1 if 'VirusShare' in root else 0
                y.append(label)
    
    return np.array(X), np.array(y)

# 1. 加载数据
X, y = load_images_and_labels(image_dir)

# 2. 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 初始化决策树并训练
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# 4. === 保存模型 ===
joblib.dump(dt, model_path)
print(f"[✔] 模型已保存至: {model_path}")

# 4. 预测并评估
y_pred = dt.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malware']))

Utils.notice_bark('决策树训练完毕！')
