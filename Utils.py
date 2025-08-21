import requests
import dotenv
import os
import json
import glob
import yaml
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple

def feature_names() -> List[str]:
    names = []
    names += [f"hist_{i}" for i in range(256)]
    names += [f"byteent_{i}" for i in range(256)]
    names += ["str_numstrings", "str_avlength", "str_printables", "str_entropy",
              "str_paths", "str_urls", "str_registry", "str_MZ"]
    names += ["gen_size", "gen_vsize", "gen_has_debug", "gen_exports", "gen_imports",
              "gen_has_relocations", "gen_has_resources", "gen_has_signature", "gen_has_tls", "gen_symbols"]
    return names  # len=530

# 特征工程
def _safe_get(d: Dict[str, Any], key: str, default=0):
    v = d.get(key, default)
    # 将 True/False -> 1/0
    if isinstance(v, bool):
        return int(v)
    # 过滤 None
    if v is None:
        return default
    return v

def extract_feature_vector(sample: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """
    将单条 JSON（一个 PE 文件）转换为固定长度向量，并返回 (X, y)
    选用特征：
      - histogram: 256 维
      - byteentropy: 256 维
      - strings: [numstrings, avlength, printables, entropy, paths, urls, registry, MZ] -> 8 维
      - general: [size, vsize, has_debug, exports, imports, has_relocations, has_resources, has_signature, has_tls, symbols] -> 10 维
    共 530 维
    """
    # 1) label
    y = int(_safe_get(sample, "label", 0))

    # 2) histogram (list 长度应为 256)
    hist = sample.get("histogram", [0]*256)
    if len(hist) != 256:
        # 兜底：过长则截断，过短则补零
        hist = (hist[:256] + [0]*256)[:256]

    # 3) byteentropy (16x16=256)
    byteent = sample.get("byteentropy", [0]*(16*16))
    if len(byteent) != 256:
        byteent = (byteent[:256] + [0]*256)[:256]

    # 4) strings（若不存在则给默认）
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

    # 5) general
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

    # 拼接为 1D 向量
    vec = np.array(hist + byteent + strings_feats + general_feats, dtype=np.float32)
    return vec, y

# 数据加载
def load_dataset_from_jsonl_dir(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 root_dir 递归读取所有 *.jsonl，组装为 (X, y)
    """
    jsonl_files = glob.glob(os.path.join(root_dir, "**", "*.jsonl"), recursive=True)
    if not jsonl_files:
        raise RuntimeError(f"No jsonl files found under: {root_dir}")

    X_list, y_list = [], []
    total, bad = 0, 0

    for fp in jsonl_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    vec, lab = extract_feature_vector(obj)
                    X_list.append(vec)
                    y_list.append(lab)
                except Exception as e:
                    bad += 1
                    # 可按需打印：print(f"[WARN] parse error in {fp}: {e}")
                    continue

    if not X_list:
        raise RuntimeError(f"All jsonl lines failed to parse under: {root_dir}")

    X = np.vstack(X_list)       # shape: (N, 530)
    y = np.array(y_list, dtype=np.int64)
    print(f"[DATA] samples={len(y)} | bad_lines={bad} | pos(Malware)={y.sum()} | neg(Benign)={len(y)-y.sum()}")
    return X, y

def index_images(image_dir):
    """仅建立文件路径与标签索引，不读像素"""
    paths, labels = [], []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith('.png'):
                p = os.path.join(root, f)
                label = 1 if 'VirusShare' in root else 0
                paths.append(p)
                labels.append(label)
    return np.array(paths), np.array(labels, dtype=np.int64)

def batch_image_loader(paths, labels, image_size, batch_size=512, dtype=np.uint8):
    """
    分批加载像素的生成器：按batch读磁盘→resize→展平→yield
    dtype可用uint8（节省内存）或float32（部分模型需要）
    """
    N = len(paths)
    D = 3 * image_size * image_size
    i = 0
    while i < N:
        j = min(i + batch_size, N)
        Xb = np.empty((j - i, D), dtype=dtype)
        yb = labels[i:j]
        for k, p in enumerate(paths[i:j]):
            img = Image.open(p).convert('RGB').resize((image_size, image_size))
            Xb[k] = np.asarray(img, dtype=np.uint8).reshape(-1)
        yield Xb, yb
        i = j

def build_memmap(paths, labels, image_size, out_X_path, out_y_path, dtype=np.uint8, batch_size=1024):
    """
    将整套数据写入磁盘memmap文件（分批写，低峰值内存）
    返回：memmap对象（只读模式再打开时可用）
    """
    N = len(paths)
    D = 3 * image_size * image_size
    X_mm = np.memmap(out_X_path, mode='w+', dtype=dtype, shape=(N, D))
    y_mm = np.memmap(out_y_path, mode='w+', dtype=np.int64, shape=(N,))
    y_mm[:] = labels  # 一次写入标签

    offset = 0
    for Xb, yb in batch_image_loader(paths, labels, image_size, batch_size=batch_size, dtype=dtype):
        n = len(yb)
        X_mm[offset:offset+n] = Xb
        offset += n

    # 刷盘并以只读方式重新打开，避免意外写入
    del X_mm; del y_mm
    X_mm = np.memmap(out_X_path, mode='r', dtype=dtype, shape=(N, D))
    y_mm = np.memmap(out_y_path, mode='r', dtype=np.int64, shape=(N,))
    return X_mm, y_mm

def load_config():
    """加载配置文件"""
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config

def check_directory_exists(directory):
    """检查目录是否存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

def notice_bark(message):
    """使用 Bark API 发送通知"""
    dotenv.load_dotenv()  # Load environment variables from .env file
    url = os.getenv("URL")
    if not url:
        print("No URL found in .env file. Cannot send notification.")
        return
    data = {
        'title': message,
    }
    headers = {
        'Content-Type': 'application/json',
    }
    print(f"Sending notification to {url} with message: {message}")
    print(f"Data: {data}")
    try:
        response = requests.get(url, headers=headers, json=data)
        if response.status_code == 200:
            print("Notification sent successfully.")
        else:
            print(f"Failed to send notification: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error sending notification: {e}")

def notice_ntfy(message):
    """使用 ntfy API 发送通知"""
    dotenv.load_dotenv()  # Load environment variables from .env file
    url = os.getenv("NTFY_URL")
    if not url:
        print("No NTFY URL found in .env file. Cannot send notification.")
        return
    print(f"Sending notification to {url} with message: {message}")
    try:
        response = requests.post(url, data=message.encode(encoding='utf-8'))
        if response.status_code == 200:
            print("Notification sent successfully.")
        else:
            print(f"Failed to send notification: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error sending notification: {e}")

        # 