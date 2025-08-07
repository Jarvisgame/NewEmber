import requests
import dotenv
import os
import yaml

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