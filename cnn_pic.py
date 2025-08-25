import os
import random
from typing import Tuple, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from sklearn.metrics import classification_report, confusion_matrix
import joblib

import Utils

def main():
    
    # 配置与随机种子
    config = Utils.load_config()
    IMAGE_DIR   = config['cnn_classifier']['pic_input_dir']         # 彩色图像根目录
    IMAGE_SIZE  = config['three_gram_byte_plot']['image_size']         # 例如 256
    BATCH_SIZE  = config['cnn_classifier']['batch_size']
    LR          = config['cnn_classifier']['lr']
    EPOCHS      = config['cnn_classifier']['epochs']
    MODEL_DIR   = config['cnn_classifier']['pic_output_dir']               # 模型输出目录
    SEED        = config['cnn_classifier']['seed']
    TEST_SIZE   = config['cnn_classifier'].get('test_size', 0.2)       # 若无则默认0.2

    Utils.check_directory_exists(MODEL_DIR)
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_malware_best.pt")
    LABEL_MAP_PATH  = os.path.join(MODEL_DIR, "label_map.pkl")         # 0=Benign,1=Malware

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # 提速（若要完全可复现可改为 False 并启用 deterministic）



# 数据集类与工具函数
class MalwareRGBDataset(Dataset):
    """
    通过文件路径列表读取 PNG 图像：
    - 父目录名包含 'VirusShare' -> label=1（恶意）
    - 否则 -> label=0（良性）
    """
    def __init__(self, file_paths: List[str], image_size: int, transform=None):
        self.file_paths = list(file_paths)
        self.image_size = image_size
        self.transform = transform if transform is not None else T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        p = self.file_paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        label = 1 if "VirusShare" in os.path.dirname(p) else 0
        return img, label
    



# 轻量 CNN 模型
class MalwareCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> 256 x 1 x 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



# 训练/验证函数
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / total
    acc = correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return avg_loss, acc, all_preds, all_labels


# 主训练流程（使用索引缓存 + 划分缓存）
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. 载入或建立：索引 + 划分（结果缓存到 MODEL_DIR）
    #    复用你在 utils.py 中的工具函数：load_or_build_index_split
    tr_paths, te_paths, tr_labels, te_labels = Utils.load_or_build_index_split(
        image_dir=IMAGE_DIR,
        test_size=TEST_SIZE,
        random_state=SEED,
        output_dir=MODEL_DIR
    )
    print(f"[DATA] Train: {len(tr_paths)} | Val: {len(te_paths)}")
    print(f"[DATA] Malware ratio (all): {np.mean(np.r_[tr_labels, te_labels]):.3f}")

    # 2. Transforms（训练有轻微增强；验证仅Resize+ToTensor）
    train_tfms = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ])
    val_tfms = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])

    train_ds = MalwareRGBDataset(tr_paths, IMAGE_SIZE, transform=train_tfms)
    val_ds   = MalwareRGBDataset(te_paths, IMAGE_SIZE, transform=val_tfms)

    # 3. 类不平衡处理：WeightedRandomSampler（按训练标签统计）
    class_sample_count = np.bincount(tr_labels, minlength=2)  # [count_benign, count_malware]
    class_weights = 1.0 / (class_sample_count + 1e-6)
    sample_weights = class_weights[tr_labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(tr_labels),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # 4. 模型/损失/优化器
    model = MalwareCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        print(f"Train  | loss: {train_loss:.4f} | acc: {train_acc:.4f}")
        print(f"Val    | loss: {val_loss:.4f} | acc: {val_acc:.4f}")

        # 保存最优
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[SAVE] Best model updated: {BEST_MODEL_PATH} (acc={best_val_acc:.4f})")

    # 5. 最终评估（加载最佳权重）
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    print("\n[Confusion Matrix]")
    print(confusion_matrix(val_labels, val_preds, labels=[0, 1]))
    print("\n[Classification Report]")
    print(classification_report(val_labels, val_preds, labels=[0, 1], target_names=["Benign", "Malware"]))

    # 6. 备份标签映射
    joblib.dump({"Benign": 0, "Malware": 1}, LABEL_MAP_PATH)
    print(f"[INFO] Label map saved to: {LABEL_MAP_PATH}")


# 单图预测
def load_cnn(model_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MalwareCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


@torch.no_grad()
def predict_image(image_path: str,
                  model_path: str = None,
                  image_size: int = IMAGE_SIZE) -> Tuple[str, float]:
    """
    返回 (label_str, confidence)
    label_str in {"Benign", "Malware"}
    """
    if model_path is None:
        model_path = BEST_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件：{model_path}")

    model, device = load_cnn(model_path)

    tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  # shape: [1,3,H,W]

    logits = model(x)
    prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(prob))

    label = "Malware" if pred == 1 else "Benign"
    confidence = float(prob[pred])
    print(f"[Predict] {os.path.basename(image_path)} -> {label} ({confidence:.2f})")
    return label, confidence

if __name__ == "__main__":
    main()
    Utils.notice_bark('CNN模型（彩色图像）训练完毕！')
