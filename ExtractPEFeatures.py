import os
import hashlib
import json
import math
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import lief 
import Utils 


# ========================
# 工具函数
# ========================
def calculate_sha256(file_path: str) -> str:
    """计算文件 SHA256"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_file_time(file_path: str) -> str:
    """文件最后修改时间：YYYY-MM"""
    mtime = os.path.getmtime(file_path)
    return datetime.fromtimestamp(mtime).strftime("%Y-%m")


def calculate_histogram(file_path: str):
    """字节直方图（256 bins）"""
    histogram = [0] * 256
    with open(file_path, "rb") as f:
        data = f.read()
        for byte in data:
            histogram[byte] += 1
    return histogram


def calculate_byte_entropy(file_path: str):
    """字节熵直方图（16x16 bins）"""
    window_size = 2048
    stride = 1024
    bins = [0] * (16 * 16)

    def shannon_entropy(window: bytes) -> float:
        counts = [0] * 256
        for b in window:
            counts[b] += 1
        ent = 0.0
        total = len(window)
        for c in counts:
            if c:
                p = c / total
                ent -= p * math.log2(p)
        return ent

    with open(file_path, "rb") as f:
        data = f.read()

    for i in range(0, max(0, len(data) - window_size + 1), stride):
        window = data[i:i + window_size]
        if len(window) != window_size:
            continue
        entropy = shannon_entropy(window)
        entropy_bin = min(int(entropy * 2), 15)  # 量化到 0-15
        for b in window:
            byte_bin = min(int(b / 16), 15)       # 量化到 0-15
            bins[entropy_bin * 16 + byte_bin] += 1

    return bins


def extract_strings(file_path: str) -> Dict[str, Any]:
    """提取可打印字符串统计"""
    printable_pattern = re.compile(b'[\x20-\x7f]{5,}')
    with open(file_path, "rb") as f:
        data = f.read()

    strings = printable_pattern.findall(data)
    numstrings = len(strings)
    avlength = (sum(len(s) for s in strings) / numstrings) if numstrings else 0.0
    printables = sum(len(s) for s in strings)

    printabledist = [0] * 96  # 0x20-0x7f
    for s in strings:
        for c in s:
            if 0x20 <= c <= 0x7f:
                printabledist[c - 0x20] += 1

    entropy = 0.0
    if printables:
        for count in printabledist:
            if count:
                p = count / printables
                entropy -= p * math.log2(p)

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
        "MZ": mz,
    }


def _resolve_entry_section(pe: "lief.PE.Binary"):
    """返回入口点所在的 Section；找不到返回 None。"""
    try:
        sec = pe.entrypoint_section  # 0.12+ 版本可能直接可用
        if sec is not None:
            return sec
    except Exception:
        pass

    rva = pe.entrypoint
    for s in pe.sections:
        start = s.virtual_address
        end = start + max(s.virtual_size, s.size)
        if start <= rva < end:
            return s
    return None


# ========================
# 主提取逻辑
# ========================
def extract_pe_metadata(file_path: str) -> Optional[Dict[str, Any]]:
    """提取单个 PE 文件的元数据（鲁棒，尽量不抛异常）"""
    try:
        pe: "lief.PE.Binary" = lief.parse(file_path)  # type: ignore
        if not pe:
            return None

        # 标签：路径中含 VirusShare 视为恶意
        label = 1 if "VirusShare" in os.path.dirname(file_path) or "VirusShare" in file_path else 0

        metadata: Dict[str, Any] = {
            "file_name": Path(file_path).stem,
            "sha256": calculate_sha256(file_path),
            "appeared": get_file_time(file_path),
            "label": label,
            "histogram": calculate_histogram(file_path),
            "byteentropy": calculate_byte_entropy(file_path),
            "strings": extract_strings(file_path),
            "general": {},
            "header": {"coff": {}, "optional": {}},
            "section": {"entry": "", "sections": []},
            "imports": {},
            "exports": [],
            "datadirectories": [],
        }

        # 通用信息
        metadata["general"] = {
            "size": os.path.getsize(file_path),
            "vsize": getattr(pe, "virtual_size", 0),
            "has_debug": int(getattr(pe, "has_debug", False)),
            "exports": len(getattr(pe, "exported_functions", [])) if getattr(pe, "has_exports", False) else 0,
            "imports": (sum(len(lib.entries) for lib in getattr(pe, "imports", []))
                        if getattr(pe, "has_imports", False) else 0),
            "has_relocations": int(getattr(pe, "has_relocations", False)),
            "has_resources": int(getattr(pe, "has_resources", False)),
            "has_signature": int((getattr(pe, "verify_signature", lambda: False)() or
                                  getattr(pe, "has_signatures", False))),
            "has_tls": int(getattr(pe, "has_tls", False)),
            "symbols": len(getattr(pe, "symbols", [])) if getattr(pe, "symbols", None) else 0,
        }

        # 头信息
        header = getattr(pe, "header", None)
        if header:
            ts = getattr(header, "time_date_stamps", None)
            if ts is None:
                ts = getattr(header, "time_date_stamp", 0)
            metadata["header"]["coff"] = {
                "timestamp": ts,
                "machine": str(getattr(header, "machine", "")),
                "characteristics": [str(c) for c in getattr(header, "characteristics_list", [])],
            }

        oh = getattr(pe, "optional_header", None)
        if oh:
            metadata["header"]["optional"] = {
                "subsystem": str(getattr(oh, "subsystem", "")),
                "dll_characteristics": [str(d) for d in getattr(oh, "dll_characteristics_lists", [])],
                "magic": str(getattr(oh, "magic", "")),
                "major_image_version": getattr(oh, "major_image_version", 0),
                "minor_image_version": getattr(oh, "minor_image_version", 0),
                "major_linker_version": getattr(oh, "major_linker_version", 0),
                "minor_linker_version": getattr(oh, "minor_linker_version", 0),
                "major_operating_system_version": getattr(oh, "major_operating_system_version", 0),
                "minor_operating_system_version": getattr(oh, "minor_operating_system_version", 0),
                "major_subsystem_version": getattr(oh, "major_subsystem_version", 0),
                "minor_subsystem_version": getattr(oh, "minor_subsystem_version", 0),
                "sizeof_code": getattr(oh, "sizeof_code", 0),
                "sizeof_headers": getattr(oh, "sizeof_headers", 0),
                "sizeof_heap_commit": getattr(oh, "sizeof_heap_commit", 0),
            }

        # 段信息
        entry_sec = _resolve_entry_section(pe)
        metadata["section"]["entry"] = getattr(entry_sec, "name", "") if entry_sec else ""

        sections_out = []
        for s in getattr(pe, "sections", []):
            chars = getattr(s, "characteristics_lists", [])
            sections_out.append({
                "name": getattr(s, "name", ""),
                "size": getattr(s, "size", 0),
                "entropy": getattr(s, "entropy", 0.0),
                "vsize": getattr(s, "virtual_size", 0),
                "props": [str(p) for p in chars],
            })
        metadata["section"]["sections"] = sections_out

        # 导入导出
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
                "size": getattr(directory, "size", 0),
                "virtual_address": getattr(directory, "rva", 0),
            }
            for directory in getattr(pe, "data_directories", [])
        ]

        return metadata
    except Exception as e:
        print(f"[ERROR] Parsing {file_path}: {e}")
        return None


def bytes_to_str(o):
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")
    raise TypeError

def is_pe_file(file_path: str) -> bool:
    """快速检查文件是否可能是 PE 文件（前两个字节为 MZ）"""
    try:
        with open(file_path, "rb") as f:
            magic = f.read(2)
        return magic == b"MZ"
    except Exception:
        return False


# ========================
# 扫描目录（按 config 输入输出）
# ========================
def scan_directory_by_config():
    """
    输入：
      - 默认从 config['path']['binary_files_dir'] 读取，
        该目录下**每个子文件夹**各生成一个同名 jsonl。
        例如：binary_files/01/*  -> 输出 pe_metadata/01.jsonl
    输出：
      - {config['path']['output_dir']}/pe_metadata/{subdir}.jsonl
      - 若 config 中提供 'pe_metadata_extractor.output_dir'，优先使用它。
    """
    config = Utils.load_config()

    # 输入目录：binary_files_dir
    input_root = Path(config['pe_static_features']['input_dir'])

    # 输出根目录：优先 pe_metadata_extractor.output_dir，否则 path.output_dir/pe_metadata
    pe_cfg = config['pe_static_features']['output_dir']
    output_root = Path(os.path.join(pe_cfg, 'pe_metadata'))
    

    # 确保输出目录存在
    Utils.check_directory_exists(output_root)

    for subdir in input_root.iterdir():
        if not subdir.is_dir():
            continue

        jsonl_path = output_root / f"{subdir.name}.jsonl"
        count_ok, count_fail, count_skip = 0, 0, 0

        with jsonl_path.open("w", encoding="utf-8") as f:
            for pe_file in subdir.glob("*"):
                if not pe_file.is_file():
                    continue

                # 先判断是否可能是 PE 文件
                if not is_pe_file(str(pe_file)):
                    count_skip += 1
                    continue

                md = extract_pe_metadata(str(pe_file))
                if md is None:
                    count_fail += 1
                    continue

                line = json.dumps(md, ensure_ascii=False, default=bytes_to_str)
                f.write(line + "\n")
                count_ok += 1

        print(f"[DONE] {subdir.name}: wrote {count_ok} items -> {jsonl_path} "
            f"(failed: {count_fail}, skipped_non_pe: {count_skip})")

# ========================
# main
# ========================
if __name__ == "__main__":
    # 直接使用 config 路径，不再交互输入
    scan_directory_by_config()
