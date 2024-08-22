import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl


# 步骤1: 读取Excel文件
excel_file = 'C:\\Users\\18374\\Desktop\data3.xlsx'  # 替换为你的Excel文件路径
data = pd.read_excel(excel_file)

# 步骤2: 数据预处理
# 假设最后一列是目标变量
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 目标变量

# 如果目标变量是分类的字符串，需要进行编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 检查是否有分类特征，如果有，使用LabelEncoder进行编码
categorical_features = X.select_dtypes(include=['object']).columns
for col in categorical_features:
    X[col] = LabelEncoder().fit_transform(X[col])

# 步骤3: 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 步骤4: 构建决策树模型
clf = DecisionTreeClassifier(random_state=42)

# 步骤5: 训练模型
clf.fit(X_train, y_train)

# 步骤6: 预测测试集结果
y_pred = clf.predict(X_test)

# 步骤7: 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 步骤8: 可视化决策树
from sklearn import tree
import matplotlib.pyplot as plt

# 使用plot_tree函数绘制训练好的决策树模型
plt.figure(figsize=(20,10))  # 可以调整图形的大小以更好地展示树
tree.plot_tree(clf, filled=True)  # filled=True 将根据多数类为节点着色

# 显示图形
plt.show()

# 可选：保存图形为文件
# plt.savefig('decision_tree.png')
