from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
data = pd.read_csv('data.csv')
# 分离特征和目标变量
X = data.iloc[:, :-1]  # 所有行，除了最后一列的所有列
y = data.iloc[:, -1]   # 所有行，最后一列
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建决策树分类器实例
clf = DecisionTreeClassifier()
# 训练模型
clf.fit(X_train, y_train)
# 预测测试集结果
y_pred = clf.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
#输入数据的形式，以下是一些基本要求：
#数值型数据：决策树算法通常处理数值型数据。如果你的数据包含分类变量（如字符串），你可能需要进行编码，将它们转换为数值型数据。可以使用 pandas 的 get_dummies 函数或 scikit-learn 的 OneHotEncoder 进行编码。

#特征和目标变量：数据应该被分为特征（X）和目标变量（y）。特征是模型用来进行预测的输入变量，而目标变量是模型试图预测的结果。
#数据清洗：在将数据输入模型之前，通常需要进行数据清洗，包括处理缺失值、异常值等。
#数据类型：scikit-learn 的大多数算法都期望输入数据为 numpy 数组或 pandas DataFrame。确保你的数据是这些格式之一。
#训练集和测试集：为了评估模型的性能，应该将数据集分为训练集和测试集。通常，训练集用于训练模型，测试集用于评估模型的泛化能力。