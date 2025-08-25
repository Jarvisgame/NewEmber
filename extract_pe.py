# pe_extract_parallel.py
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

import Utils
import lief
from concurrent.futures import ProcessPoolExecutor, as_completed


# ========= 你已有的工具函数（精简版占位，直接复用你的实现） =========
import math, re, hashlib

def calculate_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def get_file_time(file_path: str) -> str:
    mtime = os.path.getmtime(file_path)
    return datetime.fromtimestamp(mtime).strftime("%Y-%m")

def calculate_histogram(file_path: str):
    hist = [0]*256
    with open(file_path, "rb") as f:
        data = f.read()
        for b in data:
            hist[b] += 1
    return hist

def calculate_byte_entropy(file_path: str):
    window_size = 2048
    stride = 1024
    bins = [0]*(16*16)
    def shannon_entropy(win: bytes) -> float:
        cnt = [0]*256
        for x in win:
            cnt[x]+=1
        total = len(win)
        ent = 0.0
        for c in cnt:
            if c:
                p = c/total
                ent -= p*math.log2(p)
        return ent
    with open(file_path, "rb") as f:
        data = f.read()
    L = len(data)
    for i in range(0, max(0, L - window_size + 1), stride):
        win = data[i:i+window_size]
        if len(win) != window_size:
            continue
        e = shannon_entropy(win)
        ebin = min(int(e*2), 15)
        for b in win:
            bbin = min(int(b/16), 15)
            bins[ebin*16+bbin]+=1
    return bins

def extract_strings(file_path: str):
    patt = re.compile(b'[\x20-\x7f]{5,}')
    with open(file_path, "rb") as f:
        data = f.read()
    ss = patt.findall(data)
    numstrings = len(ss)
    avlength = (sum(len(s) for s in ss)/numstrings) if numstrings else 0.0
    printables = sum(len(s) for s in ss)
    dist = [0]*96
    for s in ss:
        for c in s:
            if 0x20 <= c <= 0x7f:
                dist[c-0x20]+=1
    ent = 0.0
    if printables:
        for c in dist:
            if c:
                p = c/printables
                ent -= p*math.log2(p)
    paths = sum(1 for s in ss if s.lower().startswith(b'c:\\'))
    urls = sum(1 for s in ss if s.lower().startswith(b'http://') or s.lower().startswith(b'https://'))
    registry = sum(1 for s in ss if b'HKEY_' in s)
    mz = sum(1 for s in ss if b'MZ' in s)
    return {
        "numstrings": numstrings,
        "avlength": avlength,
        "printabledist": dist,
        "printables": printables,
        "entropy": ent,
        "paths": paths,
        "urls": urls,
        "registry": registry,
        "MZ": mz
    }

def _resolve_entry_section(pe: "lief.PE.Binary"):
    try:
        sec = pe.entrypoint_section
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

def bytes_to_str(o):
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")
    raise TypeError

# ========= 新增：快速魔数检测 =========
def is_pe_file(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            return f.read(2) == b"MZ"
    except Exception:
        return False


# ========= 单文件任务：在子进程中执行 =========
def process_one_file(file_path: str) -> Tuple[str, Optional[str]]:
    """
    返回: (status, json_line_or_none)
      status in {"ok", "skip", "fail"}
    """
    p = str(file_path)
    # 1) 非PE直接跳过
    if not is_pe_file(p):
        return "skip", None

    try:
        pe: "lief.PE.Binary" = lief.parse(p)  # type: ignore
        if not pe:
            return "fail", None

        label = 1 if "VirusShare" in os.path.dirname(p) or "VirusShare" in p else 0

        md = {
            "file_name": Path(p).stem,
            "sha256": calculate_sha256(p),
            "appeared": get_file_time(p),
            "label": label,
            "histogram": calculate_histogram(p),
            "byteentropy": calculate_byte_entropy(p),
            "strings": extract_strings(p),
            "general": {},
            "header": {"coff": {}, "optional": {}},
            "section": {"entry": "", "sections": []},
            "imports": {},
            "exports": [],
            "datadirectories": [],
        }

        md["general"] = {
            "size": os.path.getsize(p),
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

        header = getattr(pe, "header", None)
        if header:
            ts = getattr(header, "time_date_stamps", None)
            if ts is None:
                ts = getattr(header, "time_date_stamp", 0)
            md["header"]["coff"] = {
                "timestamp": ts,
                "machine": str(getattr(header, "machine", "")),
                "characteristics": [str(c) for c in getattr(header, "characteristics_list", [])],
            }

        oh = getattr(pe, "optional_header", None)
        if oh:
            md["header"]["optional"] = {
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

        entry_sec = _resolve_entry_section(pe)
        md["section"]["entry"] = getattr(entry_sec, "name", "") if entry_sec else ""

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
        md["section"]["sections"] = sections_out

        if getattr(pe, "has_imports", False):
            md["imports"] = {
                lib.name: [entry.name for entry in lib.entries]
                for lib in pe.imports
            }
        if getattr(pe, "has_exports", False):
            md["exports"] = [exp.name for exp in pe.exported_functions]

        md["datadirectories"] = [
            {
                "name": str(directory.type).split(".")[-1],
                "size": getattr(directory, "size", 0),
                "virtual_address": getattr(directory, "rva", 0),
            }
            for directory in getattr(pe, "data_directories", [])
        ]

        line = json.dumps(md, ensure_ascii=False, default=bytes_to_str)
        return "ok", line

    except Exception:
        # 任意异常都视为解析失败（可能加壳/损坏）
        return "fail", None


# ========= 主流程：并行执行，边收边写 =========
def scan_directory_parallel():
    config = Utils.load_config()

    # 输入目录：binary_files_dir
    input_root = Path(config['pe_static_features']['input_dir'])

    # 输出根目录：优先 pe_metadata_extractor.output_dir，否则 path.output_dir/pe_metadata
    pe_cfg = config['pe_static_features']['output_dir']
    output_root = Path(os.path.join(pe_cfg, 'pe_metadata'))
    max_workers = config['system_info']['max_workers']

    Utils.check_directory_exists(str(output_root))

    # 遍历一级子目录：每个子目录生成单独 jsonl
    for subdir in input_root.iterdir():
        if not subdir.is_dir():
            continue

        jsonl_path = output_root / f"{subdir.name}.jsonl"
        files = [p for p in subdir.glob("*") if p.is_file()]
        if not files:
            print(f"[SKIP] empty dir: {subdir}")
            continue

        ok = fail = skip = 0
        print(f"[START] {subdir.name} | files={len(files)} | workers={max_workers}")

        with open(jsonl_path, "w", encoding="utf-8") as fout:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(process_one_file, str(p)): str(p) for p in files}
                for fut in as_completed(futures):
                    status, line = fut.result()
                    if status == "ok" and line is not None:
                        fout.write(line + "\n")
                        ok += 1
                    elif status == "skip":
                        skip += 1
                    else:
                        fail += 1

        print(f"[DONE] {subdir.name}: wrote={ok}, failed={fail}, skipped_non_pe={skip} -> {jsonl_path}")


if __name__ == "__main__":
    scan_directory_parallel()
