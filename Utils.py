import requests
import dotenv
import os
import yaml
import numpy as np
from PIL import Image

def load_images_and_labels(image_dir):

    config = load_config()
    image_size = config['three_gram_byte_plot']['image_size']
    
    X = []
    y = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)

                # 加载图像为RGB并展平
                img = Image.open(full_path).convert('RGB')
                img = img.resize((image_size, image_size))  # 保证尺寸一致
                img_array = np.array(img, dtype=np.uint8).flatten()

                X.append(img_array)

                # 标签判断：路径中是否包含 'VirusShare'
                label = 1 if 'VirusShare' in root else 0
                y.append(label)
    
    return np.array(X), np.array(y)

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