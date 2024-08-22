import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('font', family='Times New Roman')
# 读取Excel文件
df = pd.read_excel('C:\\Users\\18374\\Desktop\data2.xlsx')  # 替换为你的Excel文件路径

# 假设Excel文件中的目标列是'material_state'，特征列是'Phos', 'partner','Ac','Me','SUMO'
X = df[['Phos', 'partner','Ac','Me','SUMO']]  # 替换为实际的特征列名
y = df['material_state']  # 替换为实际的目标列名

# 编码目标变量，确保它是二分类的
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# 训练XGBoost模型
model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False)  # 修改为二分类
model.fit(X_train, y_train)

# 预测
y_pred_encoded = model.predict(X_test)

# 将预测结果转换回原始类别标签
y_pred = encoder.inverse_transform(y_pred_encoded)

# 评估模型
accuracy = accuracy_score(y_test, y_pred_encoded)
print(f"Accuracy: {accuracy}")

# 可视化特征重要性
xgb.plot_importance(model)
plt.title('Feature Importance')
plt.show()

# 输出预测结果
for i, pred in enumerate(y_pred):
    if pred == encoder.classes_[1]:  # 假设1代表"是"
        print(f"Sample {i+1}: 是")
    else:
        print(f"Sample {i+1}: 否")