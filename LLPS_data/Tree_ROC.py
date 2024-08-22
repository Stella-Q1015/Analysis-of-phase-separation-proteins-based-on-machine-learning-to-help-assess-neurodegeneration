import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1. 数据加载
df = pd.read_excel('C:\\Users\\18374\\Desktop\data3.xlsx', engine='openpyxl')

# 2. 数据预处理
# 将分类特征转换为数值型
label_encoders = {col: LabelEncoder() for col in df.columns}
for col in df.columns:
    df[col] = label_encoders[col].fit_transform(df[col])

# 假设最后一列是目标变量
X = df.drop('partner', axis=1)
y = df['partner']

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 模型训练
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. 预测并计算概率
y_prob = model.predict_proba(X_test)[:, 1]

# 6. 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


