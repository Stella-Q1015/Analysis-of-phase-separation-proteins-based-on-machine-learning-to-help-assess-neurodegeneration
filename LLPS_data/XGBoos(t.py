import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件
df = pd.read_excel('C:\\Users\\18374\\Desktop\data2.xlsx')  # 替换为你的Excel文件路径

# 假设Excel文件中的目标列是'Target'，特征列是'Feature1', 'Feature2', ...
X = df[['Phos', 'partner','Ac','Me','SUMO']]  # 替换为实际的特征列名
y = df['material_state']  # 替换为实际的目标列名

# 编码目标变量
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# 训练XGBoost模型
model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
#可视化
xgb.plot_importance(model)
plt.title('Feature Importance')
plt.show()