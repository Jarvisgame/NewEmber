import os
import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import Utils  # 配置加载模块

# 加载配置
config = Utils.load_config()
image_dir = config['decision_tree_classifier']['pic_input_dir']
Utils.check_directory_exists(image_dir)
image_size = config['three_gram_byte_plot']['image_size']
output_dir = config['decision_tree_classifier']['pic_output_dir']
Utils.check_directory_exists(output_dir)
model_path = os.path.join(output_dir, 'dt_model.pkl')  # 模型保存路径
test_size = config['decision_tree_classifier']['test_size']
random_state = config['decision_tree_classifier']['random_state']

# 1. 建索引 + 划分
tr_paths, te_paths, tr_labels, te_labels = Utils.load_or_build_index_split(
    image_dir=image_dir,
    test_size=test_size,
    random_state=random_state,
    output_dir=output_dir
)

# 2. 为训练集与测试集分别构建memmap（避免一次性像素进内存）
Xtr_mm, ytr_mm = Utils.load_or_build_memmap(
    paths=tr_paths, labels=tr_labels, image_size=image_size,
    out_X_path=os.path.join(output_dir, 'X_train.mmap'),
    out_y_path=os.path.join(output_dir, 'y_train.mmap'),
    dtype=np.uint8, batch_size=1024
)

Xte_mm, yte_mm = Utils.load_or_build_memmap(
    paths=te_paths, labels=te_labels, image_size=image_size,
    out_X_path=os.path.join(output_dir, 'X_test.mmap'),
    out_y_path=os.path.join(output_dir, 'y_test.mmap'),
    dtype=np.uint8, batch_size=1024
)

# 3. 初始化决策树并训练
clf = DecisionTreeClassifier(random_state=random_state)
clf.fit(Xtr_mm, ytr_mm)

# 4. 保存模型
joblib.dump(clf, model_path)
print(f"模型已保存至: {model_path}")


# 5. 预测并评估
y_pred = clf.predict(Xte_mm)

print("Confusion Matrix:\n", confusion_matrix(yte_mm, y_pred))
print("\nClassification Report:\n", classification_report(yte_mm, y_pred, target_names=['Benign','Malware']))


Utils.notice_bark('决策树模型（彩色图像）训练完毕！')