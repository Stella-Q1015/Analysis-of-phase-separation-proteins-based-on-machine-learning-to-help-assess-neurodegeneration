import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('C:\\Users\\18374\\Desktop\data2.xlsx')  # 替换为你的文件路径

# 选择特征列和目标列
X = df[['PTM', 'Phos', 'Ac', 'Me', 'SUMO', 'partner']]  # 特征列
y = df['material_state']  # 目标列

# 编码目标变量
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建SVM分类器
svm_model = SVC(probability=True, kernel='linear', random_state=42)

# 训练模型
svm_model.fit(X_train, y_train)

# 预测测试集结果
y_pred = svm_model.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 预测概率
y_pred_prob = svm_model.predict_proba(X_test)

# 计算ROC曲线和AUC分数
n_classes = len(np.unique(y_encoded))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制ROC曲线
plt.figure(figsize=(14, 8))

colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
             label=f'ROC curve of class {encoder.classes_[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()