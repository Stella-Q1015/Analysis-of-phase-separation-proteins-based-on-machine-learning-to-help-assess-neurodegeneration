import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras import layers, models
from openpyxl import load_workbook

# 读取Excel文件
def read_excel_data(file_path):
    wb = load_workbook(file_path)
    sheet = wb.active
    data = sheet.values  # 这里 data 是一个生成器
    
    images = []
    labels = []
    
    for row in data:
        if not row:  # 跳过空行
            continue
        label = row[0]
        # 确保图像数据列不为空并且可以转换为字符串
        if row[1] and row[2]:
            image_data_1_str = row[1].replace(" ", "")  # 移除所有空格
            image_data_2_str = row[2].replace(" ", "")
            image_data_1 = np.fromstring(image_data_1_str.encode(), sep=' ').astype('float32')
            image_data_2 = np.fromstring(image_data_2_str.encode(), sep=' ').astype('float32')
            images.append([image_data_1, image_data_2])
            labels.append(label)
    
    # 将图像列表转换为数组，并添加通道维度
    images = np.array(images).astype('float32') / 255.0  # 归一化
    # 确保标签是整数类型
    labels = np.array(labels, dtype='int32')
    
    return images, labels

# 预处理数据
def preprocess_data(images, labels):
    # 将图像数据转换为适合CNN的格式 (28, 28, 2)
    images = np.array(images).astype('float32') / 255.0
    labels = np.array(labels).astype('int32')
    
    return images, labels

# 构建CNN模型
def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # 假设有10个类别
    return model

# 主函数
def main():
    file_path = 'C:\\Users\\18374\\Desktop\data2.xlsx'  # 替换为你的Excel文件路径
    images, labels = read_excel_data(file_path)
    images, labels = preprocess_data(images, labels)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # 构建模型
    input_shape = X_train.shape[1:]  # 获取输入形状，排除批次维度
    model = build_model(input_shape)
    
    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 训练模型
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    
    # 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc}")
    
    # 数据可视化
    def plot_history(history):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='train accuracy')
        plt.plot(history.history['val_accuracy'], label='validation accuracy')
        plt.title('Accuracy over epochs')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.title('Loss over epochs')
        plt.legend()
        
        plt.show()
    
    plot_history(history)
    
    # 混淆矩阵可视化
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # 打印分类报告
    print(classification_report(y_test, y_pred_classes))

if __name__ == "__main__":
    main()