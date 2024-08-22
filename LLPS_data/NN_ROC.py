import pandas as pd
import numpy as np  # Add this line to import NumPy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder

# 假设df是包含数据的Pandas DataFrame
df = pd.read_excel('C:\\Users\\18374\\Desktop\data2.xlsx')  # 加载数据

# 选择特征列和目标列
feature_cols = ['Phos', 'Ac', 'Me', 'SUMO', 'partner']
X = df[feature_cols].values
y = df['material_state'].values

# 对分类特征进行One-Hot编码
encoder = OneHotEncoder(sparse=False)
X_partner_encoded = encoder.fit_transform(X[['partner']])

# 合并数值特征和编码后的特征
X_encoded = np.column_stack((X[['Phos', 'Ac', 'Me', 'SUMO']], X_partner_encoded))

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建神经网络模型
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # material_state是二分类问题

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# 确保模型输入层的维度与X_train的特征数量匹配
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# 预测概率
y_pred_prob = model.predict(X_test)

# 计算ROC曲线和AUC值
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 可视化ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()