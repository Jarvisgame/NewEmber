import os
import json
import glob
import numpy as np
from typing import List, Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import Utils

# =========================
# 加载配置 & 路径
# =========================
config = Utils.load_config()

pe_meta_dir = config['ensemble_classifier_jsonl']['input_dir']
output_dir  = config['ensemble_classifier_jsonl']['output_dir']
Utils.check_directory_exists(output_dir)
voting_path   = os.path.join(output_dir, 'voting_ensemble_jsonl.pkl')
stacking_path = os.path.join(output_dir, 'stacking_ensemble_jsonl.pkl')

# 通用
test_size    = config['ensemble_classifier_jsonl']['test_size']
random_state = config['ensemble_classifier_jsonl']['random_state']

# RF 超参
rf_params = config['ensemble_classifier_jsonl'].get('rf', {})
rf_n_estimators = rf_params.get('n_estimators', 300)
rf_max_depth    = rf_params.get('max_depth', None)
rf_n_jobs       = rf_params.get('n_jobs', -1)
rf_class_weight = rf_params.get('class_weight', 'balanced_subsample')

# SVM 超参
svm_params = config['ensemble_classifier_jsonl'].get('svm', {})
svm_C        = svm_params.get('C', 1.0)
svm_max_iter = svm_params.get('max_iter', 2000)
svm_class_weight = svm_params.get('class_weight', 'balanced')

# MLP 超参
mlp_params = config['ensemble_classifier_jsonl'].get('mlp', {})
mlp_hidden_sizes     = tuple(mlp_params.get('hidden_layer_sizes', [256, 128]))
mlp_max_iter         = mlp_params.get('max_iter', 200)
mlp_alpha            = mlp_params.get('alpha', 1e-4)
mlp_early_stopping   = mlp_params.get('early_stopping', True)
mlp_validation_frac  = mlp_params.get('validation_fraction', 0.1)
mlp_learning_rate    = mlp_params.get('learning_rate', 'adaptive')

# =========================
# 特征工程（与其它模型一致）
# =========================
def _safe_get(d: Dict[str, Any], key: str, default=0):
    v = d.get(key, default)
    if isinstance(v, bool): return int(v)
    if v is None: return default
    return v

def extract_feature_vector(sample: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    y = int(_safe_get(sample, "label", 0))

    hist = sample.get("histogram", [0]*256)
    hist = (hist[:256] + [0]*256)[:256]

    byteent = sample.get("byteentropy", [0]*(16*16))
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
    names += ["str_numstrings","str_avlength","str_printables","str_entropy",
              "str_paths","str_urls","str_registry","str_MZ"]
    names += ["gen_size","gen_vsize","gen_has_debug","gen_exports","gen_imports",
              "gen_has_relocations","gen_has_resources","gen_has_signature","gen_has_tls","gen_symbols"]
    return names  # 530

# =========================
# 数据加载
# =========================
def load_dataset_from_jsonl_dir(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    jsonl_files = glob.glob(os.path.join(root_dir, "**", "*.jsonl"), recursive=True)
    if not jsonl_files:
        raise RuntimeError(f"No jsonl files found under: {root_dir}")

    X_list, y_list, bad = [], [], 0
    for fp in jsonl_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    vec, lab = extract_feature_vector(obj)
                    X_list.append(vec); y_list.append(lab)
                except Exception:
                    bad += 1
                    continue

    if not X_list:
        raise RuntimeError(f"All jsonl lines failed to parse under: {root_dir}")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    print(f"[DATA] samples={len(y)} | bad_lines={bad} | pos={y.sum()} | neg={len(y)-y.sum()}")
    return X, y

# =========================
# 模型构建
# =========================
def build_voting(rs=42):
    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators, max_depth=rf_max_depth,
        n_jobs=rf_n_jobs, class_weight=rf_class_weight, random_state=rs
    )
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', C=svm_C, probability=True,
                    class_weight=svm_class_weight, random_state=rs,
                    max_iter=svm_max_iter))
    ])
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=mlp_hidden_sizes, activation='relu', solver='adam',
            alpha=mlp_alpha, max_iter=mlp_max_iter, random_state=rs,
            early_stopping=mlp_early_stopping, validation_fraction=mlp_validation_frac,
            learning_rate=mlp_learning_rate
        ))
    ])

    return VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
        voting='soft', weights=[2, 1, 1]
    )

def build_stacking(rs=42):
    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators, max_depth=rf_max_depth,
        n_jobs=rf_n_jobs, class_weight=rf_class_weight, random_state=rs
    )
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', C=svm_C, probability=True,
                    class_weight=svm_class_weight, random_state=rs,
                    max_iter=svm_max_iter))
    ])
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=mlp_hidden_sizes, activation='relu', solver='adam',
            alpha=mlp_alpha, max_iter=mlp_max_iter, random_state=rs,
            early_stopping=mlp_early_stopping, validation_fraction=mlp_validation_frac,
            learning_rate=mlp_learning_rate
        ))
    ])
    meta = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=rs)

    return StackingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
        final_estimator=meta,
        stack_method='predict_proba',
        passthrough=False
    )

# =========================
# 训练与保存
# =========================
def main():
    print(f"[INFO] Loading JSONL from: {pe_meta_dir}")
    X, y = load_dataset_from_jsonl_dir(pe_meta_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Voting
    print("\n[TRAIN] VotingClassifier")
    voting = build_voting(rs=random_state)
    voting.fit(X_train, y_train)
    joblib.dump({"model": voting, "feature_names": feature_names(),
                 "meta": {"version": "voting_jsonl_v1"}}, voting_path)
    print(f"[SAVE] Voting -> {voting_path}")
    yp = voting.predict(X_test)
    print("\n[Voting] Confusion Matrix")
    print(confusion_matrix(y_test, yp, labels=[0, 1]))
    print("\n[Voting] Classification Report")
    print(classification_report(y_test, yp, labels=[0, 1], target_names=["Benign","Malware"]))

    # Stacking
    print("\n[TRAIN] StackingClassifier")
    stacking = build_stacking(rs=random_state)
    stacking.fit(X_train, y_train)
    joblib.dump({"model": stacking, "feature_names": feature_names(),
                 "meta": {"version": "stacking_jsonl_v1"}}, stacking_path)
    print(f"[SAVE] Stacking -> {stacking_path}")
    yp2 = stacking.predict(X_test)
    print("\n[Stacking] Confusion Matrix")
    print(confusion_matrix(y_test, yp2, labels=[0, 1]))
    print("\n[Stacking] Classification Report")
    print(classification_report(y_test, yp2, labels=[0, 1], target_names=["Benign","Malware"]))

if __name__ == "__main__":
    main()
    Utils.notice_ntfy('集成模型（JSONL）训练完毕！')
