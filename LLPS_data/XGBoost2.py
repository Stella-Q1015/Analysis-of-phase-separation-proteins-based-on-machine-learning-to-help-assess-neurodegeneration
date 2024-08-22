import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. 数据加载
df = pd.read_excel('C:\\Users\\18374\\Desktop\data2.xlsx')

# 2. 数据预处理
# 假设所有列除了'partner'都是特征，'partner'是目标变量
X = df.drop('partner', axis=1)
y = df['partner']

# 将分类特征转换为数值型
label_encoders = {col: LabelEncoder() for col in X.columns}
X = X.apply(lambda x: label_encoders[x.name].fit_transform(x))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 模型训练
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 4. 模型评估
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# 使用 plot_tree 方法可视化树模型
xgb.plot_tree(model, num_trees=0, figsize=(20, 10))  # 您可以根据需要更改 num_trees 的值
plt.show()