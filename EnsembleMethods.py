# ensemble_classifier.py
import os
import numpy as np
from PIL import Image
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import Utils  # 你的配置与工具


# ======================
# 读取配置 & 路径处理（按你现有风格）
# ======================
config = Utils.load_config()

# 图像输入（与决策树/随机森林/SVM/MLP一致）
image_dir = config['ensemble_classifier']['input_dir']
image_size = config['three_gram_byte_plot']['image_size']  # 图像为 image_size × image_size × 3

# 模型输出目录（优先 ensemble_classifier，没有则兜底）
if 'ensemble_classifier' in config and 'output_dir' in config['ensemble_classifier']:
    output_dir = config['ensemble_classifier']['output_dir']
else:
    output_dir = './output/model/ensemble_model'

Utils.check_directory_exists(output_dir)

# 模型保存路径
voting_model_path = os.path.join(output_dir, 'voting_ensemble.pkl')
stacking_model_path = os.path.join(output_dir, 'stacking_ensemble.pkl')

# 其他可选参数（有则用，无则有默认）
test_size = config.get('ensemble_classifier', {}).get('test_size', 0.2)
random_state = config.get('ensemble_classifier', {}).get('random_state', 42)


# ======================
# 数据加载
# ======================
def load_images_and_labels(img_dir, img_size):
    X, y, paths = [], [], []
    for root, _, files in os.walk(img_dir):
        for fname in files:
            if not fname.lower().endswith('.png'):
                continue
            fpath = os.path.join(root, fname)
            try:
                img = Image.open(fpath).convert('RGB')
                img = img.resize((img_size, img_size))
                arr = np.array(img, dtype=np.uint8).flatten()
                X.append(arr)
                # 标签规则：VirusShare开头为恶意（1），其他为良性（0）
                label = 1 if 'VirusShare' in os.path.basename(root) or 'VirusShare' in root else 0
                y.append(label)
                paths.append(fpath)
            except Exception as e:
                print(f"[WARN] Failed to read {fpath}: {e}")
    return np.array(X), np.array(y), paths


# ======================
# 构建集成模型
# ======================
def build_voting_model(rs=42):
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=rs
    )
    svm = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('svc', SVC(kernel='linear', probability=True, random_state=rs))
    ])
    mlp = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128),
                              activation='relu', solver='adam',
                              max_iter=200, early_stopping=True,
                              random_state=rs))
    ])
    voting = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
        voting='soft', weights=[2, 1, 1]
    )
    return voting


def build_stacking_model(rs=42):
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=rs
    )
    svm = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('svc', SVC(kernel='linear', probability=True, random_state=rs))
    ])
    mlp = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128),
                              activation='relu', solver='adam',
                              max_iter=200, early_stopping=True,
                              random_state=rs))
    ])
    meta = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=rs)

    stack = StackingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
        final_estimator=meta,
        stack_method='predict_proba',
        passthrough=False
    )
    return stack


# ======================
# 评估
# ======================
def evaluate_and_print(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    print(f"\n[{title}] Confusion Matrix")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(f"\n[{title}] Classification Report")
    print(classification_report(y_test, y_pred, labels=[0, 1],
                                target_names=['Benign', 'Malware']))


# ======================
# 主流程
# ======================
def main():
    print(f"[INFO] Load images from: {image_dir}")
    X, y, _ = load_images_and_labels(image_dir, image_size)
    if X.size == 0:
        raise RuntimeError(f"No PNG images found under: {image_dir}")

    print(f"[DATA] Total: {len(X)} | Malware: {int(y.sum())} | Benign: {len(y) - int(y.sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Voting
    print("\n[TRAIN] VotingClassifier")
    voting = build_voting_model(rs=random_state)
    voting.fit(X_train, y_train)
    joblib.dump(voting, voting_model_path)
    print(f"[SAVE] Voting model -> {voting_model_path}")
    evaluate_and_print(voting, X_test, y_test, title="Voting")

    # Stacking
    print("\n[TRAIN] StackingClassifier")
    stacking = build_stacking_model(rs=random_state)
    stacking.fit(X_train, y_train)
    joblib.dump(stacking, stacking_model_path)
    print(f"[SAVE] Stacking model -> {stacking_model_path}")
    evaluate_and_print(stacking, X_test, y_test, title="Stacking")


# ======================
# 单图预测（与现有风格一致）
# ======================
def predict_single_image(image_path, model_path, img_size=None):
    """
    使用保存的集成模型预测单张 PNG：返回 (label_str, confidence)
    """
    if img_size is None:
        img_size = image_size

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    x = np.array(img, dtype=np.uint8).flatten().reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        pred = int(np.argmax(proba))
        conf = float(proba[pred])
    else:
        pred = int(model.predict(x)[0])
        conf = 1.0

    label = "Malware" if pred == 1 else "Benign"
    print(f"[Predict] {os.path.basename(image_path)} -> {label} (confidence={conf:.2f})")
    return label, conf


if __name__ == "__main__":
    main()
    # # 用法示例（训练后）：
    # img_path = r"D:\NotFatDog\Project\GeneratedRGBImages\Test\unknown.png"
    # predict_single_image(img_path, voting_model_path, img_size=image_size)   # 投票模型
    # predict_single_image(img_path, stacking_model_path, img_size=image_size) # 堆叠模型
