import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='Times New Roman')
import matplotlib.pyplot as plt

# 假设你有一个图像文件路径列表和对应的标签列表
image_paths = ["C:\\Users\\18374\\Desktop\\训练集（正常）\\G3BP1.png", "C:\\Users\\18374\\Desktop\\训练集（正常）\\cGAS.png","C:\\Users\\18374\\Desktop\\训练集（正常）\\Dcp2.png","C:\\Users\\18374\\Desktop\\训练集（正常）\\G3BP1.png","C:\\Users\\18374\\Desktop\\训练集（正常）\\Hsp27.png","C:\\Users\\18374\\Desktop\\训练集（正常）\\MeCP2.png","C:\\Users\\18374\\Desktop\\训练集（正常）\\POLR2B.png","C:\\Users\\18374\\Desktop\\训练集（正常）\\Pumilio 1.png","C:\\Users\\18374\\Desktop\\训练集（正常）\\RING1b.png","C:\\Users\\18374\\Desktop\\训练集（异常）\\ANXA11.png","C:\\Users\\18374\\Desktop\\训练集（异常）\\Hdj2.png","C:\\Users\\18374\\Desktop\\训练集（异常）\\HSF1.png","C:\\Users\\18374\\Desktop\\训练集（异常）\\Nup133.png","C:\\Users\\18374\\Desktop\\训练集（异常）\\PLK4.png","C:\\Users\\18374\\Desktop\\训练集（异常）\\SOP-2.png","C:\\Users\\18374\\Desktop\\训练集（异常）\\STING1.png","C:\\Users\\18374\\Desktop\\训练集（异常）\\TDP-43.png","C:\\Users\\18374\\Desktop\\训练集（异常）\\UBQLN2.png"]
labels = [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]  # 0 代表正常相分离，1 代表异常相分离

# 定义图像的目标尺寸和批量大小
target_size = (64, 64)  # 例如 (64, 64) 根据你的图像大小来设置
batch_size = 32

# 初始化X_train和y_train
X_train = np.zeros((len(image_paths), *target_size, 3), dtype=np.float32)
y_train = np.array(labels, dtype=np.int8)  # 标签编码为0或1

# 加载和预处理图像，填充到X_train数组中
for i, path in enumerate(image_paths):
    img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # 将图像转换为数组
    img_array /= 255.0  # 归一化
    X_train[i] = img_array

# 将标签转换为独热编码，如果使用二元交叉熵损失函数则不需要这一步
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)

# 假设你已经准备好了验证集和测试集的图像路径列表
val_image_paths = ["C:\\Users\\18374\\Desktop\\验证集（正常\\FIB1.png","C:\\Users\\18374\\Desktop\\验证集（正常\\NPM1.png","C:\\Users\\18374\\Desktop\\验证集（正常\\ORC6.png","C:\\Users\\18374\\Desktop\\验证集（异常）\\APC.png","C:\\Users\\18374\\Desktop\\验证集（异常）\\hnRNPA2.png","C:\\Users\\18374\\Desktop\\验证集（异常）\\RBM14.png"]
test_image_paths = ["C:\\Users\\18374\\Desktop\\测试集（正常）\\FMRP.png","C:\\Users\\18374\\Desktop\\测试集（正常）\\FXR1.png","C:\\Users\\18374\\Desktop\\测试集（正常）\\SynGAP.png","C:\\Users\\18374\Desktop\\测试集（异常）\\C9orf72.png","C:\\Users\\18374\\Desktop\\测试集（异常）\\FUS.png","C:\\Users\\18374\\Desktop\\测试集（异常）\\NUP62.png"]

# 假设你也有对应的验证集和测试集标签
val_labels = [0,0,0,1,1,1]
test_labels = [0,0,0,1,1,1]

# 初始化验证集和测试集数据数组
X_val = np.zeros((len(val_image_paths), *target_size, 3), dtype=np.float32)
X_test = np.zeros((len(test_image_paths), *target_size, 3), dtype=np.float32)
y_val = np.array(val_labels, dtype=np.int8)
y_test = np.array(test_labels, dtype=np.int8)

# 加载和预处理验证集图像数据
for i, path in enumerate(val_image_paths):
    img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array /= 255.0
    X_val[i] = img_array

# 加载和预处理测试集图像数据
for i, path in enumerate(test_image_paths):
    img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array /= 255.0
    X_test[i] = img_array

# 定义CNN架构
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 使用sigmoid函数因为这是一个二分类问题
])

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 使用ImageDataGenerator进行数据增强
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

# 训练模型
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=50  # 训练的epoch数量
)

# 假设X_test和y_test是测试集的特征和标签
test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest accuracy:', test_acc)

# 保存模型
model.save('protein_separation_cnn_model.h5')

# 假设你已经训练了模型并进行了保存，现在加载模型
model = tf.keras.models.load_model('protein_separation_cnn_model.h5')

# 假设X_test和y_test是你的测试集特征和标签
# 确保X_test已经进行了与训练集相同的预处理步骤

# 评估模型在测试集上的性能
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# 计算额外的性能指标
from sklearn.metrics import classification_report, confusion_matrix

# 获取预测结果
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # 根据你的模型输出阈值来设定

# 打印分类报告
print(classification_report(y_test, y_pred))

# 使用numpy的array打印混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(np.array2string(conf_matrix, separator=', ', prefix='[ ', suffix=' ]'))

# 如果你希望以更美观的格式打印，可以使用prettytable库（需要先安装：pip install prettytable）
from prettytable import PrettyTable

# 创建prettytable对象
pt = PrettyTable()
pt.field_names = ["Actual 0", "Actual 1", "Total"]

# 填充数据
for i in range(len(conf_matrix)):
    pt.add_row([conf_matrix[i, 0], conf_matrix[i, 1], np.sum(conf_matrix[i])])

# 创建一个新的图和轴对象
fig, ax = plt.subplots(figsize=(10, 8))

# 在指定的轴对象上绘制热力图
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['正常相分离', '异常相分离'],
            yticklabels=['正常相分离', '异常相分离'],
            cbar=True,  # 显示颜色条
            ax=ax)  # 指定绘制热力图的轴对象

# 设置图表标题和坐标轴标签的字号为30
ax.set_xlabel('Predicted Label', fontsize=25)
ax.set_ylabel('True Label', fontsize=25)
ax.set_title('Heatmap', fontsize=25)

# 显示图表
plt.show()