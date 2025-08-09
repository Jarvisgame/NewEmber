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

import Utils  # 你的配置读取模块


# ======================
# 读取配置 & 路径处理
# ======================
config = Utils.load_config()

IMG_DIR = config.get('ensemble_classifier', {},).get('input_dir',
            config.get('three_gram_byte_plot', {}).get('output_dir', './output/three_gram_byte_plot'))
OUT_DIR = config.get('ensemble_classifier', {},).get('output_dir', './output/model/ensemble_model')
IMAGE_SIZE = config.get('three_gram_byte_plot', {}).get('image_size', 128)
TEST_SIZE = config.get('ensemble_classifier', {}).get('test_size', 0.2)
RANDOM_STATE = config.get('ensemble_classifier', {}).get('random_state', 42)

os.makedirs(OUT_DIR, exist_ok=True)
VOTING_PATH = os.path.join(OUT_DIR, 'voting_ensemble.pkl')
STACKING_PATH = os.path.join(OUT_DIR, 'stacking_ensemble.pkl')


# ======================
# 数据加载
# ======================
def load_images_and_labels(image_dir, image_size):
    X, y, paths = [], [], []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if not fname.lower().endswith('.png'):
                continue
            fpath = os.path.join(root, fname)
            try:
                img = Image.open(fpath).convert('RGB')
                img = img.resize((image_size, image_size))
                arr = np.array(img, dtype=np.uint8).flatten()
                X.append(arr)
                label = 1 if 'VirusShare' in os.path.basename(root) or 'VirusShare' in root else 0
                y.append(label)
                paths.append(fpath)
            except Exception as e:
                print(f"[WARN] Failed to read {fpath}: {e}")
    X = np.array(X)
    y = np.array(y)
    return X, y, paths


# ======================
# 构建集成模型
# ======================
def build_voting_model(random_state=42):
    """
    软投票：RF + 线性SVM + MLP
    - SVM / MLP 前置 StandardScaler
    """
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=random_state
    )
    svm = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),   # 稀疏/高维时 with_mean=False 更稳妥
        ('svc', SVC(kernel='linear', probability=True, random_state=random_state))
    ])
    mlp = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128),
                              activation='relu', solver='adam',
                              max_iter=200, early_stopping=True,
                              random_state=random_state))
    ])

    voting = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
        voting='soft', weights=[2, 1, 1], n_jobs=None
    )
    return voting


def build_stacking_model(random_state=42):
    """
    堆叠：同上三基学习器 + LR 作为次级学习器
    """
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=random_state
    )
    svm = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('svc', SVC(kernel='linear', probability=True, random_state=random_state))
    ])
    mlp = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128),
                              activation='relu', solver='adam',
                              max_iter=200, early_stopping=True,
                              random_state=random_state))
    ])

    meta = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
    stack = StackingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
        final_estimator=meta,
        stack_method='predict_proba', passthrough=False, n_jobs=None
    )
    return stack


# ======================
# 训练 & 评估
# ======================
def evaluate_and_print(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    print(f"\n[{title}] Confusion Matrix")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(f"\n[{title}] Classification Report")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Benign', 'Malware']))


def main():
    print(f"[INFO] Loading images from: {IMG_DIR}")
    X, y, _ = load_images_and_labels(IMG_DIR, IMAGE_SIZE)
    if len(X) == 0:
        raise RuntimeError(f"No PNG images found under: {IMG_DIR}")
    print(f"[DATA] Total samples: {len(X)} | Malware: {int(y.sum())} | Benign: {len(y) - int(y.sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # === Voting ===
    print("\n[TRAIN] VotingClassifier (soft voting)")
    voting = build_voting_model(random_state=RANDOM_STATE)
    voting.fit(X_train, y_train)
    joblib.dump(voting, VOTING_PATH)
    print(f"[SAVE] Voting model -> {VOTING_PATH}")
    evaluate_and_print(voting, X_test, y_test, title="Voting")

    # === Stacking ===
    print("\n[TRAIN] StackingClassifier")
    stacking = build_stacking_model(random_state=RANDOM_STATE)
    stacking.fit(X_train, y_train)
    joblib.dump(stacking, STACKING_PATH)
    print(f"[SAVE] Stacking model -> {STACKING_PATH}")
    evaluate_and_print(stacking, X_test, y_test, title="Stacking")


# ======================
# 单图预测（投票或堆叠）
# ======================
def predict_single_image(image_path, model_path, image_size=IMAGE_SIZE):
    """
    使用已保存的集成模型预测单张 PNG 的恶意/良性
    返回: (label_str, confidence)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size))
    x = np.array(img, dtype=np.uint8).flatten().reshape(1, -1)

    # 有些模型（如 Voting/Stacking 的元器件）支持 predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        pred = int(np.argmax(proba))
        conf = float(proba[pred])
    else:
        pred = int(model.predict(x)[0])
        # 如果没有 predict_proba，就给个伪置信度
        conf = 1.0

    label = "Malware" if pred == 1 else "Benign"
    print(f"[Predict] {os.path.basename(image_path)} -> {label} (confidence={conf:.2f})")
    return label, conf


if __name__ == "__main__":
    main()
    # # 使用示例（训练完成后）：
    # img_path = r"D:\NotFatDog\Project\GeneratedRGBImages\Test\unknown.png"
    # predict_single_image(img_path, VOTING_PATH, image_size=IMAGE_SIZE)   # 使用投票模型
    # predict_single_image(img_path, STACKING_PATH, image_size=IMAGE_SIZE) # 使用堆叠模型
