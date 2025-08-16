import os
import json
import glob
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import Utils  # 你的配置模块

# =========================
# 配置 & 路径
# =========================
config = Utils.load_config()

pe_meta_dir = config['cnn_classifier_jsonl']['input_dir']
output_dir  = config['cnn_classifier_jsonl']['output_dir']
Utils.check_directory_exists(output_dir)

model_path  = os.path.join(output_dir, 'cnn1d_model_jsonl.pt')
meta_path   = os.path.join(output_dir, 'cnn1d_model_meta.pkl')

batch_size  = config['cnn_classifier_jsonl'].get('batch_size', 64)
epochs      = config['cnn_classifier_jsonl'].get('epochs', 20)
lr          = config['cnn_classifier_jsonl'].get('lr', 1e-3)
test_size   = config['cnn_classifier_jsonl'].get('test_size', 0.2)
random_state= config['cnn_classifier_jsonl'].get('random_state', 42)
num_workers = config['cnn_classifier_jsonl'].get('num_workers', 4)

# =========================
# 特征工程（与前面保持一致，530维）
# =========================
def _safe_get(d: Dict[str, Any], key: str, default=0):
    v = d.get(key, default)
    if isinstance(v, bool): return int(v)
    if v is None: return default
    return v

def extract_feature_vector(sample: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """
    530维：hist(256) + byteentropy(256) + strings(8) + general(10)
    """
    y = int(_safe_get(sample, "label", 0))

    hist = sample.get("histogram", [0]*256)
    hist = (hist[:256] + [0]*256)[:256]

    byteent = sample.get("byteentropy", [0]*(16*16))
    byteent = (byteent[:256] + [0]*256)[:256]

    s = sample.get("strings", {})
    strings_feats = [
        float(_safe_get(s, "numstrings", 0)),
        float(_safe_get(s, "avlength", 0.0)),
        float(_safe_get(s, "printables", 0)),
        float(_safe_get(s, "entropy", 0.0)),
        float(_safe_get(s, "paths", 0)),
        float(_safe_get(s, "urls", 0)),
        float(_safe_get(s, "registry", 0)),
        float(_safe_get(s, "MZ", 0)),
    ]

    g = sample.get("general", {})
    general_feats = [
        float(_safe_get(g, "size", 0)),
        float(_safe_get(g, "vsize", 0)),
        float(_safe_get(g, "has_debug", 0)),
        float(_safe_get(g, "exports", 0)),
        float(_safe_get(g, "imports", 0)),
        float(_safe_get(g, "has_relocations", 0)),
        float(_safe_get(g, "has_resources", 0)),
        float(_safe_get(g, "has_signature", 0)),
        float(_safe_get(g, "has_tls", 0)),
        float(_safe_get(g, "symbols", 0)),
    ]

    vec = np.array(hist + byteent + strings_feats + general_feats, dtype=np.float32)
    return vec, y

def feature_names() -> List[str]:
    names = []
    names += [f"hist_{i}" for i in range(256)]
    names += [f"byteent_{i}" for i in range(256)]
    names += ["str_numstrings","str_avlength","str_printables","str_entropy",
              "str_paths","str_urls","str_registry","str_MZ"]
    names += ["gen_size","gen_vsize","gen_has_debug","gen_exports","gen_imports",
              "gen_has_relocations","gen_has_resources","gen_has_signature","gen_has_tls","gen_symbols"]
    return names  # 530

# =========================
# 数据加载
# =========================
def load_dataset_from_jsonl_dir(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    files = glob.glob(os.path.join(root_dir, "**", "*.jsonl"), recursive=True)
    if not files:
        raise RuntimeError(f"No jsonl files found under: {root_dir}")

    X_list, y_list, bad = [], [], 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    vec, lab = extract_feature_vector(obj)
                    X_list.append(vec); y_list.append(lab)
                except Exception:
                    bad += 1
                    continue

    if not X_list:
        raise RuntimeError(f"All jsonl lines failed to parse under: {root_dir}")

    X = np.vstack(X_list)  # (N, 530)
    y = np.array(y_list, dtype=np.int64)
    print(f"[DATA] samples={len(y)} | bad_lines={bad} | pos={y.sum()} | neg={len(y)-y.sum()}")
    return X, y

# =========================
# Dataset / Dataloader
# =========================
class VectorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray):
        """
        X: (N, 530) float32
        y: (N,) int64 in {0,1}
        mean, std: 用训练集统计值做标准化
        """
        self.X = (X - mean) / np.clip(std, 1e-6, None)
        self.y = y

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        # Conv1d 输入: (batch, channels=1, length=530)
        x = torch.from_numpy(self.X[idx]).float().unsqueeze(0)  # (1, 530)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y

# =========================
# 1D CNN 模型
# =========================
class CNN1D(nn.Module):
    def __init__(self, seq_len: int = 530, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),           # 530 -> 265

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),           # 265 -> 132

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)    # -> (128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),              # 128
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x

# =========================
# 训练 / 评估
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * xb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss_sum += loss.item() * xb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(yb.cpu().numpy())
    import numpy as np
    return loss_sum / total, correct / total, np.concatenate(all_pred), np.concatenate(all_true)

# =========================
# 主流程
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # 1) 读数据
    X, y = load_dataset_from_jsonl_dir(pe_meta_dir)

    # 2) 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3) 标准化（按训练集统计）
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)

    # 4) DataLoader
    train_ds = VectorDataset(X_train, y_train, mean, std)
    test_ds  = VectorDataset(X_test,  y_test,  mean, std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # 5) 模型/损失/优化器（类不平衡 → class weight）
    cls_count = np.bincount(y_train, minlength=2)
    total = cls_count.sum()
    weights = torch.tensor(total / np.clip(cls_count, 1, None), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = CNN1D(seq_len=X.shape[1], num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {te_loss:.4f} acc {te_acc:.4f}")

        # 保存最好
        if te_acc >= best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), model_path)
            print(f"[SAVE] best @ acc={best_acc:.4f} -> {model_path}")

    # 最终评估（加载最优权重）
    model.load_state_dict(torch.load(model_path, map_location=device))
    _, _, y_pred, y_true = evaluate(model, test_loader, criterion, device)

    from sklearn.metrics import classification_report, confusion_matrix
    print("\n[Confusion Matrix]")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["Benign","Malware"]))

    # 保存元信息（用于推理时做标准化）
    payload = {
        "feature_names": feature_names(),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "meta": {
            "version": "cnn1d_from_jsonl_v1",
            "feature_dim": X.shape[1],
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "test_size": test_size,
            "random_state": random_state,
        }
    }
    joblib.dump(payload, meta_path)
    print(f"[SAVE] meta -> {meta_path}")

# =========================
# 单样本 JSON 推理
# =========================
@torch.no_grad()
def predict_one_json(sample: Dict[str, Any], model_path: str, meta_path: str) -> Tuple[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载元信息
    payload = joblib.load(meta_path)
    mean = payload["mean"]; std = payload["std"]

    vec, _ = extract_feature_vector(sample)  # (530,)
    x = (vec - mean) / np.clip(std, 1e-6, None)
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,530)

    model = CNN1D(seq_len=vec.shape[0], num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(prob.argmax())
    label = "Malware" if pred == 1 else "Benign"
    conf  = float(prob[pred])
    return label, conf

if __name__ == "__main__":
    main()
