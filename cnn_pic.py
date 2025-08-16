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
from sklearn.model_selection import train_test_split
import joblib

import Utils  # 你的配置模块：提供 load_config()


# =========================
# 配置与随机种子
# =========================
config = Utils.load_config()
IMAGE_DIR = config['three_gram_byte_plot']['output_dir']
IMAGE_SIZE = config['three_gram_byte_plot']['image_size']  # 例如 256
BATCH_SIZE = config['cnn_classifier']['batch_size']
LR = config['cnn_classifier']['lr']
EPOCHS = config['cnn_classifier']['epochs']
MODEL_DIR = config['cnn_classifier']['output_dir']  # 模型输出目录
Utils.check_directory_exists(MODEL_DIR)
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_malware_best.pt")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")  # 备份标签映射（虽然这里是固定的）

SEED = config['cnn_classifier']['seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =========================
# 数据集定义
# =========================
class MalwareRGBDataset(Dataset):
    """
    从根目录递归读取 PNG 图像：
    - 子文件夹名包含 'VirusShare' -> label=1（恶意）
    - 其他子文件夹 -> label=0（良性）
    """
    def __init__(self, file_paths: List[str], transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        p = self.file_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 标签规则
        label = 1 if "VirusShare" in os.path.dirname(p) else 0
        return img, label


def list_all_pngs(root_dir: str) -> List[str]:
    paths = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".png"):
                paths.append(os.path.join(r, f))
    return paths


# =========================
# 简单轻量 CNN 模型
# =========================
class MalwareCNN(nn.Module):
    """
    一个小而快的 CNN：
    输入：RGB (3, IMAGE_SIZE, IMAGE_SIZE)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),  # -> 32 x H x W
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # -> 32 x H/2 x W/2

            nn.Conv2d(32, 64, 3, stride=1, padding=1), # -> 64 x H/2 x W/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # -> 64 x H/4 x W/4

            nn.Conv2d(64, 128, 3, stride=1, padding=1),# -> 128 x H/4 x W/4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # -> 128 x H/8 x W/8

            nn.Conv2d(128, 256, 3, stride=1, padding=1),# -> 256 x H/8 x W/8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)                    # -> 256 x 1 x 1
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


# =========================
# 训练/验证工具函数
# =========================
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


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
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


# =========================
# 主训练流程
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) 列出全部 PNG 路径并做 train/val 划分
    all_paths = list_all_pngs(IMAGE_DIR)
    if len(all_paths) == 0:
        raise RuntimeError(f"在 {IMAGE_DIR} 下未找到 PNG 文件。")

    # 构造标签数组用于 stratify
    all_labels = np.array([1 if "VirusShare" in os.path.dirname(p) else 0 for p in all_paths])
    train_paths, val_paths = train_test_split(
        all_paths, test_size=0.2, random_state=SEED, stratify=all_labels
    )

    print(f"[DATA] Total: {len(all_paths)} | Train: {len(train_paths)} | Val: {len(val_paths)}")
    print(f"[DATA] Malware ratio (all): {all_labels.mean():.3f}")

    # 2) Transforms（注意：不做归一化到 mean/std，仅将像素映射到 [0,1]）
    train_tfms = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),                    # [0,255] -> [0,1]
    ])
    val_tfms = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])

    train_ds = MalwareRGBDataset(train_paths, transform=train_tfms)
    val_ds = MalwareRGBDataset(val_paths, transform=val_tfms)

    # 3) 类不平衡处理：WeightedRandomSampler（按训练集标签统计）
    train_labels = np.array([1 if "VirusShare" in os.path.dirname(p) else 0 for p in train_paths])
    class_sample_count = np.bincount(train_labels, minlength=2)  # [count_benign, count_malware]
    class_weights = 1.0 / (class_sample_count + 1e-6)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                    num_samples=len(train_labels),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 4) 模型/损失/优化器
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

    # 5) 最终评估报告（使用最优模型）
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    print("\n[Confusion Matrix]")
    print(confusion_matrix(val_labels, val_preds, labels=[0, 1]))
    print("\n[Classification Report]")
    print(classification_report(val_labels, val_preds, labels=[0, 1], target_names=["Benign", "Malware"]))

    # 6) 备份标签映射（虽然这里固定：0=Benign, 1=Malware）
    joblib.dump({"Benign": 0, "Malware": 1}, LABEL_MAP_PATH)
    print(f"[INFO] Label map saved to: {LABEL_MAP_PATH}")


# =========================
# 单图预测（推理）
# =========================
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
    Utils.notice_bark('CNN模型训练完毕！')
    # # 示例：训练后进行单图预测（可取消注释使用）
    # test_img = r"D:\NotFatDog\Project\GeneratedRGBImages\Test\unknown.png"
    # predict_image(test_img, BEST_MODEL_PATH, IMAGE_SIZE)
