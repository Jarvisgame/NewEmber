import Utils
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

config = Utils.load_config()


def generate_three_3grams(data):
    # 生成3-gram
    ngrams = {}
    for i in range(len(data) - 2):
        ngram = (data[i], data[i + 1], data[i + 2])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return ngrams


def data_cleaning(data):
    """对数据进行清洗"""
    if not isinstance(data, bytearray):
        data = bytearray(data)

    # 移除无效字节
    cleaned_data = [byte for byte in data if 0 <= byte <= 255]

    # 移除无关指令 #0x90是NOP指令
    cleaned_data = [byte for byte in cleaned_data if byte != 0x90]

    return bytearray(cleaned_data)


def ngram_to_rgb_image(ngram_dict, image_size=256):
    """
    将频率最高的 n 个3-gram 映射为RGB图像
    :param ngram_dict: {(b1, b2, b3): freq}
    :return: np.ndarray of shape (image_size, image_size, 3)
    """
    sorted_ngrams = sorted(ngram_dict.items(), key=lambda x: x[1], reverse=True)
    max_count = image_size * image_size
    rgb_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    for idx, (ngram, _) in enumerate(sorted_ngrams[:max_count]):
        r, g, b = ngram
        row = idx // image_size
        col = idx % image_size
        rgb_array[row, col] = [r, g, b]

    return rgb_array


def process_file_rgb(file_path, input_dir, output_dir, image_size=256):
    """处理单个文件并生成RGB图像，保持子目录结构"""
    try:
        with open(file_path, "rb") as f:
            data = bytearray(f.read())

        data = data_cleaning(data)
        ngram_dict = generate_three_3grams(data)
        rgb_img = ngram_to_rgb_image(ngram_dict, image_size=image_size)

        # 构建输出路径，保持目录结构
        rel_path = os.path.relpath(file_path, input_dir)
        rel_dir = os.path.dirname(rel_path)
        output_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(output_subdir, exist_ok=True)

        output_path = os.path.join(output_subdir, os.path.basename(file_path) + ".png")
        img = Image.fromarray(rgb_img, mode="RGB")
        img.save(output_path)
        print(f"Saved RGB image: {output_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def process_files_rgb_parallel(input_dir, output_dir, image_size=256, max_workers=4):
    """并行处理 input_dir 下所有子目录中的文件，输出到对应的 output_dir 子目录中"""
    file_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file_rgb, fp, input_dir, output_dir, image_size)
            for fp in file_paths
        ]
        _ = [f.result() for f in futures]


if __name__ == "__main__":
    # 加载配置文件
    input_dir = config["three_gram_byte_plot"]["input_dir"]
    output_dir = config["three_gram_byte_plot"]["output_dir"]
    image_size = config["three_gram_byte_plot"]["image_size"]
    # 如果没有这两个目录则创建
    Utils.check_directory_exists(input_dir)
    Utils.check_directory_exists(output_dir)
    process_files_rgb_parallel(
        input_dir, output_dir, image_size=image_size, max_workers=4
    )
