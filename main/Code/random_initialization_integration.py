from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import InputLayer, Conv1D, MaxPooling1D
from keras.layers import Dropout, Flatten, Dense, LSTM, Reshape, BatchNormalization
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import CSVLogger


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
    model.summary()

    return model


# 定义模型训练过程图绘制函数
def plot_performance(hist, i):
    hist_ = hist.history
    epochs = hist.epoch

    plt.plot(epochs, hist_['accuracy'], label='Training Accuracy')
    plt.plot(epochs, hist_['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('E:\\2024\\Code\\result\\' + 'Model_' + str(i+1) + '_Training Accuracy.png')
    plt.close()
    # plt.show()

    plt.figure()
    plt.plot(epochs, hist_['loss'], label='Training loss')
    plt.plot(epochs, hist_['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('E:\\2024\\Code\\result\\' + 'Model_' + str(i+1) + '_Training Loss.png')
    plt.close()
    # plt.show()


def ensemble_predict(models, X_test):
    # 软投票
    # 获取所有模型的预测概率分布
    predictions = [model.predict(X_test) for model in models]
    # 对所有模型的预测概率分布取平均
    averaged_predictions = np.mean(predictions, axis=0)
    # 根据平均概率分布做预测，选择概率最高的类别作为预测结果
    soft_voting_predictions = np.argmax(averaged_predictions, axis=1)

    # 硬投票
    # 获取所有模型的预测类别
    predictions_for_hard = [np.argmax(model.predict(X_test), axis=1) for model in models]
    # 对所有模型的预测类别进行统计，选择出现次数最多的类别作为最终预测结果
    hard_voting_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions_for_hard)

    return soft_voting_predictions, hard_voting_predictions


path = 'E:\\2024\\Data\\slices_5.16\\train\\'

# 初始化存储数据的列表
train_data_list = []
train_label_list = []

input_shape = (4115, 1)
num_classes = 7
epochs = 300
batch_size = 512
num_models = 5


# train_data
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        file_path = path + filename

        train_label = filename.split('_')[1]
        pattern = r'(baseline|N1|N2|N3|N4|N5|N6)'  # Neutral|Enjoyment|Disgust|Surprise|Anger|Sadness|Fear

        if re.search(pattern, train_label):
            data = pd.read_csv(file_path)['data'].values
            padding_length = input_shape[0] - len(data)
            padding_value = 0
            padded_data = np.concatenate((data, np.full(padding_length, padding_value)))
            train_data_list.append(padded_data)

            labels = 0
            if train_label == 'baseline':
                labels = 0
            if train_label == 'N1':
                labels = 1
            if train_label == 'N2':
                labels = 2
            if train_label == 'N3':
                labels = 3
            if train_label == 'N4':
                labels = 4
            if train_label == 'N5':
                labels = 5
            if train_label == 'N6':
                labels = 6
            train_label_list.append(labels)


N = len(train_label_list)
print(train_label_list.count(0) / N, train_label_list.count(1) / N, train_label_list.count(2) / N,
      train_label_list.count(3) / N,
      train_label_list.count(4) / N, train_label_list.count(5) / N, train_label_list.count(6) / N)

train_label_list = np.array(train_label_list)
train_data_list = np.array(train_data_list)

x_train, x_val, y_train, y_val = train_test_split(train_data_list, np.array(train_label_list), test_size=0.25, random_state=24)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)


# 创建并训练不同初始化的模型
models = [create_model(i * 3) for i in range(num_models)]

for i, model in enumerate(models):
    print(f"Training model {i + 1}...")
    checkpoint_filepath = 'E:\\2024\\Code\\result\\5.16\\random initialization integration\\random_model_' + str(i+1) + '.h5'  # 创建检查点回调，设置保存路径及文件名格式
    csv_log_file = 'E:\\2024\\Code\\result\\5.16\\random initialization integration\\training_history_model_' + str(i+1) + '.csv'  # 指定CSV日志文件的名称
    # 创建CSVLogger回调
    # 创建CSVLogger回调
    csv_logger = CSVLogger(csv_log_file)
    # 创建检查点回调，设置保存路径及文件名格式
    checkpoint_callback = ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_accuracy',  # 根据验证集准确率决定何时保存
        save_best_only=True,  # 是否只保存最优模型，这里我们希望保存所有周期结束的模型
        verbose=1
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, csv_logger]
    )

    plot_performance(history, i)
