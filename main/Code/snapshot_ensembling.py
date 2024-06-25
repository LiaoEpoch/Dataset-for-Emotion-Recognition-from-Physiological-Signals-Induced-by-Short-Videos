from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import InputLayer, Conv1D, MaxPooling1D
from keras.layers import Dropout, Flatten, Dense, LSTM, Reshape, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os
import re
import numpy as np
import pandas as pd
import seaborn
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger
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
    model.summary()

    return model


class DisplayLR(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print('Learning rate: {}'.format(K.get_value(self.model.optimizer.learning_rate)))


def cosine_annealing_warm_restarts_schedule(epoch, T_0, T_mult=1, eta_max=0.001, eta_min=0, verbose=0):
    """
    epoch: 当前的训练 epoch 数（从1开始）
    T_0: 第一个周期的长度（以 epoch 为单位）
    T_mult: 后续周期长度的乘数
    eta_max: 最大学习率
    eta_min: 最小学习率（可以设为0）
    """
    x = (epoch / T_0) % 1
    alpha = 0.5 * (1 + np.cos(np.pi * x))
    new_lr = eta_min + (eta_max - eta_min) * alpha

    # 确保新的学习率始终大于等于eta_min，并转换为float类型
    new_lr = max(float(new_lr), float(eta_min))

    return new_lr


path = 'E:\\2024\\Data\\slices_5.16\\train\\'

# 初始化存储数据的列表
train_data_list = []
train_label_list = []

input_shape = (4115, 1)
num_classes = 7
epochs = 300
batch_size = 512
num_models = 5
T_0 = 60  # 第一个周期的迭代次数
eta_max = 0.001  # 最大学习率
eta_min = 0  # 最小学习率（也可以设置一个非零的小值）


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
models = [create_model(i * 3) for i in range(num_models)]  # 这里的随机初始化还可以采用其他策略，比如Xavier初始化和He初始化等

for i, model in enumerate(models):
    print(f"Training model {i + 1}...")
    checkpoint_filepath = 'E:\\2024\\Code\\result\\5.16\\snapshot ensembling\\snapshot_model_' + str(i+1) + '_{epoch:03d}.h5'  # 创建检查点回调，设置保存路径及文件名格式
    csv_log_file = 'E:\\2024\\Code\\result\\5.16\\snapshot ensembling\\training_history_model_' + str(i+1) + '.csv'  # 指定CSV日志文件的名称
    # 创建CSVLogger回调
    csv_logger = CSVLogger(csv_log_file)
    # 创建检查点回调，设置保存路径及文件名格式
    checkpoint_callback = ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_accuracy',  # 根据验证集准确率决定何时保存
        save_best_only=False,  # 是否只保存最优模型，这里我们希望保存所有周期结束的模型
        # save_weights_only=False,
        verbose=1,
        period=60,  # 修改这里，设置为每25个epoch保存一次
    )

    # 创建实例并加入回调列表
    display_lr_callback = DisplayLR()
    # 创建学习率调度器回调
    # 创建一个带有参数的匿名函数，并将其传给LearningRateScheduler
    lr_scheduler_callback = LearningRateScheduler(lambda epoch: cosine_annealing_warm_restarts_schedule(epoch=epoch, T_0=T_0, eta_max=eta_max, eta_min=eta_min))

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[lr_scheduler_callback, display_lr_callback, checkpoint_callback, csv_logger]
    )
