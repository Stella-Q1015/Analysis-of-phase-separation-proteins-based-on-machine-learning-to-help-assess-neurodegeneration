import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from graphviz import Source

# 导入matplotlib.pyplot
import matplotlib.pyplot as plt

# 设置matplotlib的样式为蓝色系
blue_style = {
    'axes.facecolor': '#E8F4FF',
    'axes.edgecolor': '#82CAFF',
    'axes.grid': True,
    'grid.color': '#B3E2FF',
    'grid.linestyle': '--',
    'text.color': '#0C2C66',
    'xtick.color': '#0C2C66',
    'ytick.color': '#0C2C66',
    'lines.solid_capstyle': 'round'
}

plt.style.use(blue_style)

# 1. 数据加载
df = pd.read_excel('C:\\Users\\18374\\Desktop\data2.xlsx')

# 2. 数据预处理
# 假设所有列除了'partner'都是特征，'partner'是目标变量
X = df.drop('partner', axis=1)
y = df['partner']

# 将分类特征转换为数值型
label_encoders = {col: LabelEncoder() for col in X.select_dtypes(include=['object']).columns}
X = X.apply(lambda x: label_encoders[x.name].fit_transform(x) if x.name in label_encoders else x)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 模型训练
model = xgb.XGBClassifier(use_label_encoder=True, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 4. 模型评估
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# 5. 可视化决策树
# 将XGBoost模型转换为dot格式的字符串
dot_data = xgb.to_graphviz(model, num_trees=1, rankfirst=False)

# 使用graphviz包渲染图形
graph = Source(dot_data)

# 保存图形到文件系统
graph.render("xgb_tree", format='png', cleanup=True)
