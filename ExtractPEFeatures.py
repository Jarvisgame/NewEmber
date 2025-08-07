import os
import lief
import hashlib
import json
import math
import re
from pathlib import Path
from datetime import datetime



def calculate_sha256(file_path):
    """计算文件的 SHA256 哈希值"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def get_file_time(file_path):
    """获取文件的最后修改时间，格式为 YYYY-MM"""
    mtime = os.path.getmtime(file_path)
    return datetime.fromtimestamp(mtime).strftime("%Y-%m")

def calculate_histogram(file_path):
    """计算字节直方图（256 个 bin）"""
    histogram = [0] * 256
    with open(file_path, "rb") as f:
        data = f.read()
        for byte in data:
            histogram[byte] += 1
    return histogram

def calculate_byte_entropy(file_path):
    """计算字节熵直方图（16x16 bin）"""
    window_size = 2048
    stride = 1024
    bins = [0] * (16 * 16)
    
    def shannon_entropy(window):
        """计算窗口的 Shannon 熵"""
        counts = [0] * 256
        for byte in window:
            counts[byte] += 1
        entropy = 0
        for count in counts:
            if count > 0:
                p = count / len(window)
                entropy -= p * math.log2(p)
        return entropy

    with open(file_path, "rb") as f:
        data = f.read()
    
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        if len(window) == window_size:
            entropy = shannon_entropy(window)
            entropy_bin = min(int(entropy * 2), 15)  # 量化为 0-15
            for byte in window:
                byte_bin = min(int(byte / 16), 15)  # 量化为 0-15
                bins[entropy_bin * 16 + byte_bin] += 1
    
    return bins

def extract_strings(file_path):
    """提取字符串信息"""
    printable_pattern = re.compile(b'[\x20-\x7f]{5,}')
    strings = []
    with open(file_path, "rb") as f:
        data = f.read()
        strings = printable_pattern.findall(data)
    
    numstrings = len(strings)
    avlength = sum(len(s) for s in strings) / numstrings if numstrings > 0 else 0
    printables = sum(len(s) for s in strings)
    
    # 可打印字符分布（0x20 到 0x7f，96 个字符）
    printabledist = [0] * 96
    for s in strings:
        for c in s:
            if 0x20 <= c <= 0x7f:
                printabledist[c - 0x20] += 1
    
    # 计算字符串熵
    entropy = 0
    if printables > 0:
        for count in printabledist:
            if count > 0:
                p = count / printables
                entropy -= p * math.log2(p)
    
    # 统计特定字符串
    paths = sum(1 for s in strings if s.lower().startswith(b'c:\\'))
    urls = sum(1 for s in strings if s.lower().startswith(b'http://') or s.lower().startswith(b'https://'))
    registry = sum(1 for s in strings if b'HKEY_' in s)
    mz = sum(1 for s in strings if b'MZ' in s)
    
    return {
        "numstrings": numstrings,
        "avlength": avlength,
        "printabledist": printabledist,
        "printables": printables,
        "entropy": entropy,
        "paths": paths,
        "urls": urls,
        "registry": registry,
        "MZ": mz
    }

def _resolve_entry_section(pe: lief.PE.Binary):
    """
    返回入口点所在的 lief.PE.Section 对象，找不到则返回 None。
    兼容没有 entrypoint_section 属性的旧版 LIEF。
    """
    # LIEF ≥0.12 直接有
    try:
        sec = pe.entrypoint_section            # type: ignore # 可能为 None
        if sec is not None:
            return sec
    except Exception as e:
        print(f"Error parsing: {e}")
        pass

    # —— 手动计算（保险起见）——
    rva = pe.entrypoint
    for s in pe.sections:
        start = s.virtual_address
        end = start + max(s.virtual_size, s.size)
        if start <= rva < end:
            return s
    return None

def extract_pe_metadata(file_path):
    """提取 PE 文件的元数据"""

    try:
        pe: lief.PE.Binary = lief.parse(file_path) # type: ignore
        if not pe:
            return None

        # 初始化 JSON 对象
        metadata = {
            "sha256": calculate_sha256(file_path),
            "appeared": get_file_time(file_path),
            "label": 1,  # 确认为恶意软件
            "histogram": calculate_histogram(file_path),
            "byteentropy": calculate_byte_entropy(file_path),
            "strings": extract_strings(file_path),
            "general": {},
            "header": {"coff": {}, "optional": {}},
            "section": {"entry": "", "sections": []},
            "imports": {},
            "exports": [],
            "datadirectories": []
        }

        # 通用文件信息
        metadata["general"] = {
            "size": os.path.getsize(file_path),
            "vsize": pe.virtual_size,
            "has_debug": int(getattr(pe, "has_debug", False)),
            "exports": len(pe.exported_functions) if getattr(pe, "has_exports", False) else 0,
            "imports": (sum(len(lib.entries) for lib in pe.imports)
                        if getattr(pe, "has_imports", False) else 0),
            "has_relocations": int(getattr(pe, "has_relocations", False)),
            "has_resources": int(getattr(pe, "has_resources", False)),
            # verify_signature() 可能在新旧版本行为不同，故多做一次 has_signatures 兜底
            "has_signature": int((getattr(pe, "verify_signature", lambda: False)() or
                                  getattr(pe, "has_signatures", False))),
            "has_tls": int(getattr(pe, "has_tls", False)),
            "symbols": len(pe.symbols) if getattr(pe, "symbols", None) else 0
        }

        # 头信息
        metadata["header"]["coff"] = {
            "timestamp": getattr(pe.header, "time_date_stamps",
                                 getattr(pe.header, "time_date_stamp", 0)),
            "machine": str(pe.header.machine),
            "characteristics": [str(c) for c in pe.header.characteristics_list]
        }
        oh = pe.optional_header
        metadata["header"]["optional"] = {
            "subsystem": str(oh.subsystem),
            "dll_characteristics": [str(d) for d in oh.dll_characteristics_lists],
            "magic": str(oh.magic),
            "major_image_version": oh.major_image_version,
            "minor_image_version": oh.minor_image_version,
            "major_linker_version": oh.major_linker_version,
            "minor_linker_version": oh.minor_linker_version,
            "major_operating_system_version": oh.major_operating_system_version,
            "minor_operating_system_version": oh.minor_operating_system_version,
            "major_subsystem_version": oh.major_subsystem_version,
            "minor_subsystem_version": oh.minor_subsystem_version,
            "sizeof_code": oh.sizeof_code,
            "sizeof_headers": oh.sizeof_headers,
            "sizeof_heap_commit": oh.sizeof_heap_commit
        }

        # 段信息
        ep_sec = _resolve_entry_section(pe)
        metadata["section"]["entry"] = ep_sec.name if ep_sec else ""

        sections_out = []
        for s in pe.sections:
            chars = getattr(s, "characteristics_lists", [])
            sections_out.append(
                {
                    "name": s.name,
                    "size": s.size,
                    "entropy": getattr(s, "entropy", 0.0),
                    "vsize": s.virtual_size,
                    "props": [str(p) for p in chars]
                }
            )
        metadata["section"]["sections"] = sections_out

        # 导入导出函数
        if getattr(pe, "has_imports", False):
            metadata["imports"] = {
                lib.name: [entry.name for entry in lib.entries]
                for lib in pe.imports
            }
        if getattr(pe, "has_exports", False):
            metadata["exports"] = [exp.name for exp in pe.exported_functions]

        # 数据目录
        metadata["datadirectories"] = [
            {
                "name": str(directory.type).split(".")[-1],
                "size": directory.size,
                "virtual_address": directory.rva
            }
            for directory in pe.data_directories
        ]

        return metadata
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None
    
def bytes_to_str(o):
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")
    raise TypeError

def scan_directory(scan_dir):
    # 扫描scan_dir目录下的所有PE文件夹，并根据文件夹名字生成对应的jsonl文件,例如文件夹下有命名为01的文件夹，则会单独生成一个01.jsonl文件
    scan_dir = Path(scan_dir)
    if not scan_dir.is_dir():
        print(f"Directory {scan_dir} does not exist or is not a directory.")
        return
    for subdir in scan_dir.iterdir():
        if subdir.is_dir():
            jsonl_file = subdir / f"{subdir.name}.jsonl"
            with jsonl_file.open("w", encoding="utf-8") as f:
                # 其中PE文件只有文件名没有.exe后缀
                for pe_file in subdir.glob("*"):
                    metadata = extract_pe_metadata(pe_file)
                    if metadata:
                        metadata["file_name"] = pe_file.stem

if __name__ == "__main__":
    scan_dir = input("Enter directory to scan (default: ./scan_dir): ") or "./scan_dir"
    scan_directory(scan_dir)
    #scan_directory()
    #finish()