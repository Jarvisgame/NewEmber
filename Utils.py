import requests
import dotenv
import os
import yaml
import numpy as np
from PIL import Image

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