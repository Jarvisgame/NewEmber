import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np

# ==== 配置 ====
BATCH_SIZE = 16
EPOCHS = 10
IMAGE_SIZE = 128  # 可以是 128 或 256，根据你生成图像决定
DATA_DIR = 'dataset'  # 改成你的图像文件夹
MODEL_PATH = 'cnn_malware_model.pth'

# ==== 数据增强与预处理 ====
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # 将像素归一化到 [-1, 1]
])

# ==== 加载数据 ====
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 类别映射
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
print(f"[✔] 类别标签映射: {idx_to_class}")

# ==== 定义简单CNN模型 ====
class MalwareCNN(nn.Module):
    def __init__(self):
        super(MalwareCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 输入为RGB
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 二分类
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier(x)

# ==== 初始化模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MalwareCNN().to(device)

# ==== 损失函数与优化器 ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==== 训练 ====
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")

# ==== 保存模型 ====
torch.save(model.state_dict(), MODEL_PATH)
print(f"[✔] 模型已保存至: {MODEL_PATH}")
