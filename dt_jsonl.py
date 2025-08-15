import os
import json
import glob
import numpy as np
from typing import List, Dict, Any, Tuple

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import Utils

# 加载配置
config = Utils.load_config()
pe_meta_dir = config['decision_tree_classifier_jsonl']['input_dir']
output_dir = config['decision_tree_classifier_jsonl']['output_dir']
Utils.check_directory_exists(output_dir)
model_path = os.path.join(output_dir, 'dt_model_jsonl.pkl')

# 评估/随机种子
test_size = config['decision_tree_classifier_jsonl']['test_size']
random_state = config['decision_tree_classifier_jsonl']['random_state']

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


def feature_names() -> List[str]:
    names = []
    names += [f"hist_{i}" for i in range(256)]
    names += [f"byteent_{i}" for i in range(256)]
    names += ["str_numstrings", "str_avlength", "str_printables", "str_entropy",
              "str_paths", "str_urls", "str_registry", "str_MZ"]
    names += ["gen_size", "gen_vsize", "gen_has_debug", "gen_exports", "gen_imports",
              "gen_has_relocations", "gen_has_resources", "gen_has_signature", "gen_has_tls", "gen_symbols"]
    return names  # len=530

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


# 训练与保存
def main():
    print(f"[INFO] Loading JSONL from: {pe_meta_dir}")
    X, y = load_dataset_from_jsonl_dir(pe_meta_dir)

    # 训练/验证划分（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 决策树（先用默认参数，保证快速跑通；后续可调 max_depth/min_samples_split 等）
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    print("\n[Confusion Matrix]")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Benign", "Malware"]))

    # 模型 + 附加信息一并保存
    payload = {
        "model": clf,
        "feature_names": feature_names(),
        "meta": {
            "version": "dt_from_jsonl_v1",
            "feature_dim": X.shape[1],
            "test_size": test_size,
            "random_state": random_state,
        }
    }
    joblib.dump(payload, model_path)
    print(f"[SAVE] model -> {model_path}")


if __name__ == "__main__":
    main()
