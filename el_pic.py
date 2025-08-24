import os
import numpy as np
import joblib
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import Utils


# 配置 & 路径
config = Utils.load_config()

IMG_DIR = config.get("ensemble_classifier", {}).get(
    "input_dir",
    config.get("three_gram_byte_plot", {}).get(
        "output_dir", "./output/three_gram_byte_plot"
    ),
)
OUT_DIR = config.get("ensemble_classifier", {}).get(
    "output_dir", "./output/model/ensemble_model"
)
IMAGE_SIZE = config.get("three_gram_byte_plot", {}).get("image_size", 128)
TEST_SIZE = config.get("ensemble_classifier", {}).get("test_size", 0.2)
RANDOM_STATE = config.get("ensemble_classifier", {}).get("random_state", 42)

Utils.check_directory_exists(OUT_DIR)
VOTING_PATH = os.path.join(OUT_DIR, "voting_ensemble.pkl")
STACKING_PATH = os.path.join(OUT_DIR, "stacking_ensemble.pkl")




# 集成模型构建
def build_voting_model(random_state=RANDOM_STATE):
    # 软投票：RF + 线性SVM + MLP
    # rf
    n_estimators = config.get("ensemble_classifier", {}).get("rf", {}).get("n_estimators", 300)
    n_jobs = config.get("ensemble_classifier", {}).get("rf", {}).get("n_jobs", -1)
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None, n_jobs=n_jobs, random_state=random_state
    )
    # svm
    svm = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),   # 不做均值中心化，避免巨大内存开销
        ('svc', SVC(kernel='linear', probability=True, random_state=random_state))
    ])
    # mlp
    hidden_layer_sizes = tuple(config.get("ensemble_classifier", {}).get("mlp", {}).get("hidden_layer_sizes", (256, 128)))
    max_iter = config.get("ensemble_classifier", {}).get("mlp", {}).get("max_iter", 200)
    early_stopping = config.get("ensemble_classifier", {}).get("mlp", {}).get("early_stopping", True)
    mlp = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('mlp', MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                              activation='relu', solver='adam',
                              max_iter=max_iter, early_stopping=early_stopping,
                              random_state=random_state))
    ])
    voting = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
        voting='soft', weights=[2, 1, 1]
        # sklearn 的 VotingClassifier 在部分版本没有 n_jobs 统一调度；各基学习器内部各自并行
    )
    return voting

def build_stacking_model(random_state=42):
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
        stack_method='predict_proba',
        passthrough=False
        # 同理：并行度主要由基学习器自身控制
    )
    return stack

# 评估工具
def evaluate_and_print(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    print(f"\n[{title}] Confusion Matrix")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(f"\n[{title}] Classification Report")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Benign', 'Malware']))

# 主流程（索引缓存 + memmap）
def main():
    print(f"[INFO] Using images from: {IMG_DIR}")

    # 1. 载入/生成：索引与划分（缓存到 OUT_DIR 下 .npy）
    tr_paths, te_paths, tr_labels, te_labels = Utils.load_or_build_index_split(
        image_dir=IMG_DIR,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        output_dir=OUT_DIR
    )
    print(f"[DATA] Train: {len(tr_paths)} | Test: {len(te_paths)}")
    print(f"[DATA] Malware ratio (all): {np.mean(np.r_[tr_labels, te_labels]):.3f}")

    # 2. 载入/生成：memmap（将像素分批写入 .mmap，再只读加载，降低峰值内存）
    Xtr_mm, ytr_mm = Utils.load_or_build_memmap(
        paths=tr_paths, labels=tr_labels, image_size=IMAGE_SIZE,
        out_X_path=os.path.join(OUT_DIR, 'X_train.mmap'),
        out_y_path=os.path.join(OUT_DIR, 'y_train.mmap'),
        dtype=np.uint8, batch_size=1024
    )
    Xte_mm, yte_mm = Utils.load_or_build_memmap(
        paths=te_paths, labels=te_labels, image_size=IMAGE_SIZE,
        out_X_path=os.path.join(OUT_DIR, 'X_test.mmap'),
        out_y_path=os.path.join(OUT_DIR, 'y_test.mmap'),
        dtype=np.uint8, batch_size=1024
    )

    # 3. 训练 Voting
    print("\n[TRAIN] VotingClassifier (soft voting)")
    voting = build_voting_model(random_state=RANDOM_STATE)
    voting.fit(Xtr_mm, ytr_mm)
    joblib.dump(voting, VOTING_PATH)
    print(f"[SAVE] Voting model -> {VOTING_PATH}")
    evaluate_and_print(voting, Xte_mm, yte_mm, title="Voting")

    # 4. 训练 Stacking
    print("\n[TRAIN] StackingClassifier")
    stacking = build_stacking_model(random_state=RANDOM_STATE)
    stacking.fit(Xtr_mm, ytr_mm)
    joblib.dump(stacking, STACKING_PATH)
    print(f"[SAVE] Stacking model -> {STACKING_PATH}")
    evaluate_and_print(stacking, Xte_mm, yte_mm, title="Stacking")

# 单图预测（投票或堆叠）
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
    Utils.notice_bark('集成模型（彩色图像）训练完毕！')