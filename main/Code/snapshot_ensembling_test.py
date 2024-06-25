from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf
from keras.layers import InputLayer, Conv1D, LSTM, Dense, Dropout, Flatten, Reshape, MaxPooling1D, BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.keras.regularizers import l2


def create_model(seed=None):
    np.random.seed(seed)  # 设置随机种子影响权重初始化
    model = Sequential()
    model.add(InputLayer(input_shape=(4115, 1)))

    # 添加L2正则化和批量归一化
    model.add(
        Conv1D(filters=32, kernel_size=30, strides=2, padding='SAME', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='SAME'))

    model.add(
        Conv1D(filters=64, kernel_size=25, strides=2, padding='SAME', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=1, padding='SAME'))

    model.add(
        Conv1D(filters=128, kernel_size=20, strides=2, padding='SAME', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='SAME'))

    model.add(
        Conv1D(filters=64, kernel_size=15, strides=2, padding='SAME', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='SAME'))

    model.add(
        Conv1D(filters=16, kernel_size=10, strides=1, padding='SAME', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='SAME'))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())

    model.add(Reshape((128, 1)))
    model.add(LSTM(64))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


models_dict = {}
models_path = 'E:\\2024\\Code\\result\\5.16\\snapshot ensembling\\'
for model_name in os.listdir(models_path):
    if model_name.endswith('.h5'):
        if 'model' in model_name:
            weight_file_path = os.path.join(models_path, model_name)
            model = create_model()
            model.load_weights(weight_file_path)
            models_dict[model_name] = model

data_path = 'E:\\2024\\Data\\slices_5.16\\test\\'

data_list = []
test_label_list = []

for filename in os.listdir(data_path):
    if filename.endswith('.csv'):
        file_path = data_path + filename

        test_label = filename.split('_')[1]
        pattern = r'(baseline|N1|N2|N3|N4|N5|N6)'  # Neutral|Enjoyment|Disgust|Surprise|Anger|Sadness|Fear

        if re.search(pattern, test_label):
            data = pd.read_csv(file_path)['data'].values
            data_list.append(data)
            labels = 0
            if test_label == 'baseline':
                labels = 0
            if test_label == 'N1':
                labels = 1
            if test_label == 'N2':
                labels = 2
            if test_label == 'N3':
                labels = 3
            if test_label == 'N4':
                labels = 4
            if test_label == 'N5':
                labels = 5
            if test_label == 'N6':
                labels = 6
            test_label_list.append(labels)

scaler = StandardScaler()
X_test = scaler.fit_transform(data_list)

X_test = np.array(X_test).reshape((-1, 4115, 1))
# print(len(np.array(test_label_list)))
y_test = to_categorical(test_label_list)


# 使用集成模型进行预测
models = []
keys = models_dict.keys()
for key in keys:
    if 'model' in key:
        model = models_dict[key]
        prediction = model.predict(X_test)
        y_predictions = np.argmax(prediction, axis=1)

        # 计算预测准确率
        test_accuracy = accuracy_score(test_label_list, y_predictions)
        print(key, test_accuracy)

        # 计算混淆矩阵
        test_confusion = confusion_matrix(test_label_list, y_predictions)

        # 绘制混淆矩阵
        plt.figure(figsize=(7, 6))
        seaborn.heatmap(test_confusion, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Calm', 'Happy', 'Disgust', 'Surprise', 'Anger', 'Sad', 'Fear'],
                        yticklabels=['Calm', 'Happy', 'Disgust', 'Surprise', 'Anger', 'Sad', 'Fear'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('E:\\2024\\Code\\result\\5.16\\snapshot ensembling\\' + str(key.split('.')[0]) + '_cm.png')
        plt.close()
        # plt.show()

'''
snapshot_model_1_060.h5 0.5237362637362637
snapshot_model_1_120.h5 0.7112087912087912
snapshot_model_1_180.h5 0.833076923076923
snapshot_model_1_240.h5 0.8685714285714285
snapshot_model_1_300.h5 0.8850549450549451

snapshot_model_2_060.h5 0.33087912087912086
snapshot_model_2_120.h5 0.6249450549450549
snapshot_model_2_180.h5 0.8368131868131868
snapshot_model_2_240.h5 0.8762637362637362
snapshot_model_2_300.h5 0.899010989010989

snapshot_model_3_060.h5 0.44164835164835164
snapshot_model_3_120.h5 0.6907692307692308
snapshot_model_3_180.h5 0.8072527472527472
snapshot_model_3_240.h5 0.8406593406593407
snapshot_model_3_300.h5 0.8612087912087912

snapshot_model_4_060.h5 0.36604395604395606
snapshot_model_4_120.h5 0.6158241758241758
snapshot_model_4_180.h5 0.8012087912087912
snapshot_model_4_240.h5 0.8503296703296703
snapshot_model_4_300.h5 0.873956043956044

snapshot_model_5_060.h5 0.5271428571428571
snapshot_model_5_120.h5 0.7506593406593407
snapshot_model_5_180.h5 0.8406593406593407
snapshot_model_5_240.h5 0.8713186813186813
snapshot_model_5_300.h5 0.8834065934065934
'''
