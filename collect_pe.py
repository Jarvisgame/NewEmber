import os
import shutil
import hashlib
from pathlib import Path

# 目标保存路径
DEST_DIR = r"C:\PE_Samples_Benign"

# 要收集的扩展名（典型 PE 文件）
PE_EXTENSIONS = ('.exe', '.dll', '.sys', '.ocx', '.cpl')

# 要遍历的系统路径（可根据实际系统添加更多）
SCAN_DIRS = [
    r"C:\Windows\System32",
    r"C:\Windows\SysWOW64",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
]

def get_sha256(file_path):
    """用于文件去重命名"""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha256.update(data)
    except Exception as e:
        print(f"[Hash Error] {file_path}: {e}")
        return None
    return sha256.hexdigest()

def collect_pe_files():
    Path(DEST_DIR).mkdir(parents=True, exist_ok=True)
    collected = 0
    seen_hashes = set()

    for scan_dir in SCAN_DIRS:
        for root, dirs, files in os.walk(scan_dir):
            for file in files:
                if file.lower().endswith(PE_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    file_hash = get_sha256(full_path)
                    if file_hash and file_hash not in seen_hashes:
                        seen_hashes.add(file_hash)
                        try:
                            dest_path = os.path.join(DEST_DIR, file_hash)  # 无扩展名保存
                            shutil.copy2(full_path, dest_path)
                            collected += 1
                        except Exception as e:
                            print(f"[Copy Error] {full_path} -> {dest_path}: {e}")
    print(f"✅ 完成收集，共采集 {collected} 个 PE 文件。")

if __name__ == "__main__":
    collect_pe_files()
