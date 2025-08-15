import os
import json
import glob
import numpy as np
from typing import List, Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import Utils

# =========================
# 加载配置 & 路径
# =========================
config = Utils.load_config()

# 路径与参数从 random_forest_classifier_jsonl 读取
pe_meta_dir = config['random_forest_classifier_jsonl']['input_dir']
output_dir = config['random_forest_classifier_jsonl']['output_dir']
Utils.check_directory_exists(output_dir)
model_path = os.path.join(output_dir, 'rf_model_jsonl.pkl')

# 评估/随机种子/模型超参
test_size = config['random_forest_classifier_jsonl']['test_size']
random_state = config['random_forest_classifier_jsonl']['random_state']
n_estimators = config['random_forest_classifier_jsonl'].get('n_estimators', 300)
max_depth = config['random_forest_classifier_jsonl'].get('max_depth', None)
n_jobs = config['random_forest_classifier_jsonl'].get('n_jobs', -1)
class_weight = config['random_forest_classifier_jsonl'].get('class_weight', 'balanced_subsample')

# =========================
# 特征工程（与DT版本一致）
# =========================
def _safe_get(d: Dict[str, Any], key: str, default=0):
    v = d.get(key, default)
    if isinstance(v, bool):
        return int(v)
    if v is None:
        return default
    return v

def extract_feature_vector(sample: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """
    选用特征：
      - histogram: 256
      - byteentropy: 256
      - strings: 8（numstrings, avlength, printables, entropy, paths, urls, registry, MZ）
      - general: 10（size, vsize, has_debug, exports, imports, has_relocations, has_resources, has_signature, has_tls, symbols）
    共 530 维
    """
    y = int(_safe_get(sample, "label", 0))

    hist = sample.get("histogram", [0]*256)
    if len(hist) != 256:
        hist = (hist[:256] + [0]*256)[:256]

    byteent = sample.get("byteentropy", [0]*(16*16))
    if len(byteent) != 256:
        byteent = (byteent[:256] + [0]*256)[:256]

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
    return names  # 530

# =========================
# 数据加载
# =========================
def load_dataset_from_jsonl_dir(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    jsonl_files = glob.glob(os.path.join(root_dir, "**", "*.jsonl"), recursive=True)
    if not jsonl_files:
        raise RuntimeError(f"No jsonl files found under: {root_dir}")

    X_list, y_list = [], []
    bad = 0
    for fp in jsonl_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    vec, lab = extract_feature_vector(obj)
                    X_list.append(vec)
                    y_list.append(lab)
                except Exception:
                    bad += 1
                    continue

    if not X_list:
        raise RuntimeError(f"All jsonl lines failed to parse under: {root_dir}")

    X = np.vstack(X_list)  # (N, 530)
    y = np.array(y_list, dtype=np.int64)
    print(f"[DATA] samples={len(y)} | bad_lines={bad} | pos(Malware)={y.sum()} | neg(Benign)={len(y)-y.sum()}")
    return X, y

# =========================
# 训练与保存
# =========================
def main():
    print(f"[INFO] Loading JSONL from: {pe_meta_dir}")
    X, y = load_dataset_from_jsonl_dir(pe_meta_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 随机森林
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight
    )
    rf.fit(X_train, y_train)

    # 评估
    y_pred = rf.predict(X_test)
    print("\n[Confusion Matrix]")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Benign", "Malware"]))

    # 保存模型 + 元信息
    payload = {
        "model": rf,
        "feature_names": feature_names(),
        "meta": {
            "version": "rf_from_jsonl_v1",
            "feature_dim": X.shape[1],
            "test_size": test_size,
            "random_state": random_state,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_jobs": n_jobs,
            "class_weight": class_weight
        }
    }
    joblib.dump(payload, model_path)
    print(f"[SAVE] model -> {model_path}")

if __name__ == "__main__":
    main()
    Utils.notice_ntfy('随机森林模型（JSONL）训练完毕！')