import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import Utils

# === 加载配置 ===
config = Utils.load_config()
image_dir = config['three_gram_byte_plot']['output_dir']
image_size = config['three_gram_byte_plot']['image_size']
output_dir = config['mlp_classifier']['output_dir']
Utils.check_directory_exists(output_dir)
model_path = os.path.join(image_dir, 'mlp_model.pkl')  # 模型保存路径

# === 加载数据 ===
X, y = Utils.load_images_and_labels(image_dir)

# === 数据划分 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 初始化并训练 MLP 模型 ===
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    max_iter=200,
    early_stopping=True,
    random_state=42,
    verbose=True  # 训练过程输出
)
mlp.fit(X_train, y_train)

# === 保存模型 ===
joblib.dump(mlp, model_path)
print(f"[✔] MLP 模型已保存至: {model_path}")

# === 评估模型 ===
y_pred = mlp.predict(X_test)
print("\n[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Benign", "Malware"]))
