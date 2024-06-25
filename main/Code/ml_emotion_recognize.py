import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from collections import Counter

# 设置Matplotlib的字体为支持中文的字体
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def features_variety(path):
    for file_name in os.listdir(path):
        if file_name.endswith('.csv'):
            csv_data = pd.read_csv(path + file_name)
            features_num = csv_data.keys().to_list().index('Label')
            header = csv_data.columns.to_list()
            features_name_list = header[1:features_num]

            happy_greater_ratio = []
            happy_less_ratio = []
            disgust_greater_ratio = []
            disgust_less_ratio = []
            surprise_greater_ratio = []
            surprise_less_ratio = []
            anger_greater_ratio = []
            anger_less_ratio = []
            sad_greater_ratio = []
            sad_less_ratio = []
            fear_greater_ratio = []
            fear_less_ratio = []

            for feature_name in features_name_list:
                features = csv_data[feature_name].to_list()
                label_list = csv_data['Label'].to_list()

                features_of_calm = []
                features_of_happy = []
                features_of_disgust = []
                features_of_surprise = []
                features_of_anger = []
                features_of_sad = []
                features_of_fear = []

                for feature, label in zip(features, label_list):
                    # label = label_list[features.index(feature)]
                    if label == 'Calm':
                        features_of_calm.append(feature)

                    if label == 'Happy':
                        features_of_happy.append(feature)

                    if label == 'Disgust':
                        features_of_disgust.append(feature)

                    if label == 'Surprise':
                        features_of_surprise.append(feature)

                    if label == 'Anger':
                        features_of_anger.append(feature)

                    if label == 'Sad':
                        features_of_sad.append(feature)

                    if label == 'Fear':
                        features_of_fear.append(feature)

                n = len(np.array(features_of_calm))

                less_count = 0
                greater_count = 0
                for feature_value_of_calm, feature_value_of_happy in zip(features_of_calm, features_of_happy):
                    if feature_value_of_calm > feature_value_of_happy:
                        less_count += 1
                    elif feature_value_of_calm < feature_value_of_happy:
                        greater_count += 1
                happy_less_ratio.append((less_count / n))
                happy_greater_ratio.append((greater_count / n))

                less_count = 0
                greater_count = 0
                for feature_value_of_calm, feature_value_of_disgust in zip(features_of_calm, features_of_disgust):
                    if feature_value_of_calm > feature_value_of_disgust:
                        less_count += 1
                    elif feature_value_of_calm < feature_value_of_disgust:
                        greater_count += 1
                disgust_less_ratio.append((less_count / n))
                disgust_greater_ratio.append((greater_count / n))

                less_count = 0
                greater_count = 0
                for feature_value_of_calm, feature_value_of_surprise in zip(features_of_calm,
                                                                            features_of_surprise):
                    if feature_value_of_calm > feature_value_of_surprise:
                        less_count += 1
                    elif feature_value_of_calm < feature_value_of_surprise:
                        greater_count += 1
                surprise_less_ratio.append((less_count / n))
                surprise_greater_ratio.append((greater_count / n))

                less_count = 0
                greater_count = 0
                for feature_value_of_calm, feature_value_of_anger in zip(features_of_calm, features_of_anger):
                    if feature_value_of_calm > feature_value_of_anger:
                        less_count += 1
                    elif feature_value_of_calm < feature_value_of_anger:
                        greater_count += 1
                anger_less_ratio.append((less_count / n))
                anger_greater_ratio.append((greater_count / n))

                less_count = 0
                greater_count = 0
                for feature_value_of_calm, feature_value_of_sad in zip(features_of_calm, features_of_sad):
                    if feature_value_of_calm > feature_value_of_sad:
                        less_count += 1
                    elif feature_value_of_calm < feature_value_of_sad:
                        greater_count += 1
                sad_less_ratio.append((less_count / n))
                sad_greater_ratio.append((greater_count / n))

                less_count = 0
                greater_count = 0
                for feature_value_of_calm, feature_value_of_fear in zip(features_of_calm, features_of_fear):
                    if feature_value_of_calm > feature_value_of_fear:
                        less_count += 1
                    elif feature_value_of_calm < feature_value_of_fear:
                        greater_count += 1
                fear_less_ratio.append((less_count / n))
                fear_greater_ratio.append((greater_count / n))

            df = pd.DataFrame({'feature_name': np.array(features_name_list).flatten(),
                               'happy_less_ratio': np.array(happy_less_ratio).flatten(),
                               'disgust_less_ratio': np.array(disgust_less_ratio).flatten(),
                               'surprise_less_ratio': np.array(surprise_less_ratio).flatten(),
                               'anger_less_ratio': np.array(anger_less_ratio).flatten(),
                               'sad_less_ratio': np.array(sad_less_ratio).flatten(),
                               'fear_less_ratio': np.array(fear_less_ratio).flatten(),

                               'happy_greater_ratio': np.array(happy_greater_ratio).flatten(),
                               'disgust_greater_ratio': np.array(disgust_greater_ratio).flatten(),
                               'surprise_greater_ratio': np.array(surprise_greater_ratio).flatten(),
                               'anger_greater_ratio': np.array(anger_greater_ratio).flatten(),
                               'sad_greater_ratio': np.array(sad_greater_ratio).flatten(),
                               'fear_greater_ratio': np.array(fear_greater_ratio).flatten(),
                               })
            df.to_csv('E:\\2024\\Code\\result\\changed_features\\' + file_name.replace('all_', ''), index=False)


def feature_selection(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)

        key = file_name.split('.')[0]

        # 定义分类器
        svm = SVC(probability=True)
        knn = KNeighborsClassifier(n_neighbors=5)
        dt = DecisionTreeClassifier()
        ab = AdaBoostClassifier()
        rf = RandomForestClassifier()

        csv_data = pd.read_csv(file_path)  # X是特征矩阵，y是标签
        features_num = csv_data.keys().to_list().index('P')
        all_features = csv_data.iloc[:, :features_num]
        labels = csv_data['class']

        one_object_k_best_hard_voting_acc, one_object_k_best_soft_voting_acc = [], []
        ab_acc_list, rf_acc_list = [], []
        ab_f1_list, rf_f1_list = [], []
        selected_feature_dict_list = []

        for k in range(1, 125):
            k_best = SelectKBest(score_func=f_classif, k=k)
            X_selected = k_best.fit_transform(all_features, labels)

            # 获取选中特征的索引
            selected_feature_dict = {'k': None,
                                     'selected_feature_indices': None}
            selected_feature_indices = k_best.get_support(indices=True)
            selected_feature_dict['k'] = k
            selected_feature_dict['selected_feature_indices'] = selected_feature_indices
            selected_feature_dict_list.append(selected_feature_dict)

            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X_selected, labels, test_size=0.2, random_state=2023)

            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            svm_acc = accuracy_score(y_test, y_pred)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            knn_acc = accuracy_score(y_test, y_pred)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            dt_acc = accuracy_score(y_test, y_pred)

            ab.fit(X_train, y_train)
            y_pred = ab.predict(X_test)
            ab_acc = accuracy_score(y_test, y_pred)
            ab_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            ab_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            ab_f1 = 2 * ab_precision * ab_recall / (ab_precision + ab_recall)
            ab_acc_list.append(ab_acc)
            ab_f1_list.append(ab_f1)

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            rf_acc = accuracy_score(y_test, y_pred)
            rf_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rf_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            rf_f1 = 2 * rf_precision * rf_recall / (rf_precision + rf_recall)
            rf_acc_list.append(rf_acc)
            rf_f1_list.append(rf_f1)

        # key = 'K_Best Vote ' + key
        ab_max_acc = max(ab_acc_list)
        ab_max_acc_index = ab_acc_list.index(ab_max_acc) + 1
        ab_max_acc_f1 = ab_f1_list[ab_max_acc_index - 1]
        rf_max_acc = max(rf_acc_list)
        rf_max_acc_index = rf_acc_list.index(rf_max_acc) + 1
        rf_max_acc_f1 = ab_f1_list[rf_max_acc_index - 1]

        for dictionary in selected_feature_dict_list:
            # print(dictionary)
            if dictionary['k'] == ab_max_acc_index:
                print(
                    f'{key},AB,{dictionary["selected_feature_indices"]},k:{ab_max_acc_index},max_acc:{ab_max_acc},f1:{ab_max_acc_f1}')

            if dictionary['k'] == rf_max_acc_index:
                print(
                    f'{key},RF,{dictionary["selected_feature_indices"]},k={rf_max_acc_index},max_acc:{rf_max_acc},f1:{rf_max_acc_f1}')


def emotion_classify(path):
    # 定义分类器
    svm = SVC(kernel='poly', probability=True)
    knn = KNeighborsClassifier(n_neighbors=5)
    dt = DecisionTreeClassifier()
    ab = AdaBoostClassifier()
    rf = RandomForestClassifier()
    nb = GaussianNB()

    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)

        key = file_name.split('.')[0]

        svm_acc_list, knn_acc_list, dt_acc_list, ab_acc_list, rf_acc_list, nb_acc_list = [], [], [], [], [], []
        svm_f1_list, knn_f1_list, dt_f1_list, ab_f1_list, rf_f1_list, nb_f1_list = [], [], [], [], [], []

        csv_data = pd.read_csv(file_path)  # X是特征矩阵，y是标签
        features_num = csv_data.keys().to_list().index('Label')
        all_features = csv_data.iloc[:, :features_num]
        labels = csv_data['Num']

        for k in range(1, 126):
            k_best = SelectKBest(score_func=f_classif, k=k)
            X_selected = k_best.fit_transform(all_features, labels)

            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X_selected, labels, test_size=0.2, random_state=2023)

            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            svm_acc = accuracy_score(y_test, y_pred)
            svm_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            svm_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            svm_f1 = 2 * svm_precision * svm_recall / (svm_precision + svm_recall)
            svm_acc_list.append(svm_acc)
            svm_f1_list.append(svm_f1)

            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            knn_acc = accuracy_score(y_test, y_pred)
            knn_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            knn_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            knn_f1 = 2 * knn_precision * knn_recall / (knn_precision + knn_recall)
            knn_acc_list.append(knn_acc)
            knn_f1_list.append(knn_f1)

            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            dt_acc = accuracy_score(y_test, y_pred)
            dt_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            dt_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            dt_f1 = 2 * dt_precision * dt_recall / (dt_precision + dt_recall)
            dt_acc_list.append(dt_acc)
            dt_f1_list.append(dt_f1)

            ab.fit(X_train, y_train)
            y_pred = ab.predict(X_test)
            ab_acc = accuracy_score(y_test, y_pred)
            ab_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            ab_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            ab_f1 = 2 * ab_precision * ab_recall / (ab_precision + ab_recall)
            ab_acc_list.append(ab_acc)
            ab_f1_list.append(ab_f1)

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            rf_acc = accuracy_score(y_test, y_pred)
            rf_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rf_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            rf_f1 = 2 * rf_precision * rf_recall / (rf_precision + rf_recall)
            rf_acc_list.append(rf_acc)
            rf_f1_list.append(rf_f1)

            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)
            nb_acc = accuracy_score(y_test, y_pred)
            nb_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            nb_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            nb_f1 = 2 * nb_precision * nb_recall / (nb_precision + nb_recall)
            nb_acc_list.append(nb_acc)
            nb_f1_list.append(nb_f1)

        print(f'{key}:'
              f'max_svm_acc:{max(svm_acc_list)},max_svm_f1:{max(svm_f1_list)},'
              f'max_knn_acc:{max(knn_acc_list)},max_knn_f1:{max(knn_f1_list)},'
              f'max_dt_acc:{max(dt_acc_list)},max_dt_f1:{max(dt_f1_list)},'
              f'max_ab_acc:{max(ab_acc_list)},max_ab_f1:{max(ab_f1_list)},'
              f'max_rf_acc:{max(rf_acc_list)},max_rf_f1:{max(rf_f1_list)},'
              f'max_nb_acc:{max(nb_acc_list)},max_nb_f1:{max(nb_f1_list)}')


if __name__ == '__main__':
    '''
    Emotion recognition - Switch file paths as needed
    path: The location where the data (feature) to be identified is stored
    '''
    # 1. emotion classification
    path = 'E:\\2024\\Code\\result\\features\\feature_groups\\Binary_classification\\'
    # path = 'E:\\2024\\Code\\result\\features\\feature_groups\\Three_classification\\'
    emotion_classify(path)
    '''
    result: 0--calm; 1--happy; 2--disgust; 3--surprise; 4--anger; 5--sad; 6--fear
    class_01:max_svm_acc:0.625,max_svm_f1:0.7010135135135136,max_knn_acc:0.75,max_knn_f1:0.7928571428571428,max_dt_acc:0.875,max_dt_f1:0.8886718750000001,max_ab_acc:0.875,max_ab_f1:0.8886718750000001,max_rf_acc:0.875,max_rf_f1:0.8886718750000001,max_nb_acc:0.875,max_nb_f1:0.8886718750000001
    class_02:max_svm_acc:0.75,max_svm_f1:0.7545180722891566,max_knn_acc:0.75,max_knn_f1:0.75,max_dt_acc:1.0,max_dt_f1:1.0,max_ab_acc:1.0,max_ab_f1:1.0,max_rf_acc:1.0,max_rf_f1:1.0,max_nb_acc:0.6875,max_nb_f1:0.6864567526555386
    class_03:max_svm_acc:0.625,max_svm_f1:0.6313451776649746,max_knn_acc:0.6875,max_knn_f1:0.7087004405286343,max_dt_acc:0.8125,max_dt_f1:0.8396840148698884,max_ab_acc:1.0,max_ab_f1:1.0,max_rf_acc:1.0,max_rf_f1:1.0,max_nb_acc:0.6875,max_nb_f1:0.6864567526555386
    class_04:max_svm_acc:0.75,max_svm_f1:0.7865853658536586,max_knn_acc:1.0,max_knn_f1:1.0,max_dt_acc:0.9375,max_dt_f1:0.9406146179401993,max_ab_acc:0.9375,max_ab_f1:0.9406146179401993,max_rf_acc:1.0,max_rf_f1:1.0,max_nb_acc:0.8125,max_nb_f1:0.8135403329065302
    class_05:max_svm_acc:0.8125,max_svm_f1:0.8352803738317757,max_knn_acc:0.6875,max_knn_f1:0.6913841807909604,max_dt_acc:0.8125,max_dt_f1:0.8135403329065302,max_ab_acc:0.875,max_ab_f1:0.8886718750000001,max_rf_acc:0.8125,max_rf_f1:0.8163875598086124,max_nb_acc:0.8125,max_nb_f1:0.8135403329065302
    class_06:max_svm_acc:1.0,max_svm_f1:1.0,max_knn_acc:1.0,max_knn_f1:1.0,max_dt_acc:0.9375,max_dt_f1:0.9406146179401993,max_ab_acc:1.0,max_ab_f1:1.0,max_rf_acc:1.0,max_rf_f1:1.0,max_nb_acc:1.0,max_nb_f1:1.0
    class_12:max_svm_acc:0.6875,max_svm_f1:0.7391141141141141,max_knn_acc:0.8125,max_knn_f1:0.8352803738317757,max_dt_acc:0.875,max_dt_f1:0.8862179487179488,max_ab_acc:0.9375,max_ab_f1:0.9406146179401993,max_rf_acc:1.0,max_rf_f1:1.0,max_nb_acc:0.875,max_nb_f1:0.8862179487179488
    class_14:max_svm_acc:0.6875,max_svm_f1:0.7391141141141141,max_knn_acc:0.75,max_knn_f1:0.76171875,max_dt_acc:0.75,max_dt_f1:0.76171875,max_ab_acc:0.8125,max_ab_f1:0.8396840148698884,max_rf_acc:0.8125,max_rf_f1:0.8396840148698884,max_nb_acc:0.75,max_nb_f1:0.7865853658536586
    class_15:max_svm_acc:0.8125,max_svm_f1:0.8352803738317757,max_knn_acc:0.8125,max_knn_f1:0.8135403329065302,max_dt_acc:0.8125,max_dt_f1:0.8135403329065302,max_ab_acc:0.8125,max_ab_f1:0.8396840148698884,max_rf_acc:0.8125,max_rf_f1:0.8163875598086124,max_nb_acc:0.8125,max_nb_f1:0.8135403329065302
    class_16:max_svm_acc:0.9375,max_svm_f1:0.9413900414937759,max_knn_acc:0.9375,max_knn_f1:0.9406146179401993,max_dt_acc:0.875,max_dt_f1:0.8862179487179488,max_ab_acc:0.9375,max_ab_f1:0.9406146179401993,max_rf_acc:1.0,max_rf_f1:1.0,max_nb_acc:0.9375,max_nb_f1:0.9413900414937759
    class_32:max_svm_acc:0.625,max_svm_f1:0.6552768166089965,max_knn_acc:0.8125,max_knn_f1:0.8163875598086124,max_dt_acc:0.8125,max_dt_f1:0.8135403329065302,max_ab_acc:0.8125,max_ab_f1:0.8163875598086124,max_rf_acc:0.8125,max_rf_f1:0.8135403329065302,max_nb_acc:0.6875,max_nb_f1:0.6864567526555386
    class_34:max_svm_acc:0.625,max_svm_f1:0.6552768166089965,max_knn_acc:0.6875,max_knn_f1:0.6864567526555386,max_dt_acc:0.75,max_dt_f1:0.76171875,max_ab_acc:0.8125,max_ab_f1:0.8396840148698884,max_rf_acc:0.875,max_rf_f1:0.8886718750000001,max_nb_acc:0.6875,max_nb_f1:0.6864567526555386
    class_35:max_svm_acc:0.5625,max_svm_f1:0.6052631578947368,max_knn_acc:0.75,max_knn_f1:0.76171875,max_dt_acc:0.75,max_dt_f1:0.7545180722891566,max_ab_acc:0.8125,max_ab_f1:0.8135403329065302,max_rf_acc:0.75,max_rf_f1:0.7545180722891566,max_nb_acc:0.625,max_nb_f1:0.7010135135135136
    class_36:max_svm_acc:0.875,max_svm_f1:0.875,max_knn_acc:0.875,max_knn_f1:0.8886718750000001,max_dt_acc:0.8125,max_dt_f1:0.8135403329065302,max_ab_acc:0.8125,max_ab_f1:0.8396840148698884,max_rf_acc:0.875,max_rf_f1:0.875,max_nb_acc:0.875,max_nb_f1:0.8886718750000001

    class_012:max_svm_acc:0.5217391304347826,max_svm_f1:0.5185296625043879,max_knn_acc:0.6521739130434783,max_knn_f1:0.6929347826086957,max_dt_acc:0.782608695652174,max_dt_f1:0.782608695652174,max_ab_acc:0.8260869565217391,max_ab_f1:0.8403298350824588,max_rf_acc:0.9130434782608695,max_rf_f1:0.9161385408990419,max_nb_acc:0.6956521739130435,max_nb_f1:0.7025287484843932
    class_014:max_svm_acc:0.5217391304347826,max_svm_f1:0.4911185262148338,max_knn_acc:0.5652173913043478,max_knn_f1:0.6242171189979122,max_dt_acc:0.6956521739130435,max_dt_f1:0.7079655834656423,max_ab_acc:0.8260869565217391,max_ab_f1:0.8311028978738094,max_rf_acc:0.782608695652174,max_rf_f1:0.808306294613887,max_nb_acc:0.7391304347826086,max_nb_f1:0.7673574810897238
    class_015:max_svm_acc:0.5652173913043478,max_svm_f1:0.620169082125604,max_knn_acc:0.6086956521739131,max_knn_f1:0.6513717311086155,max_dt_acc:0.6956521739130435,max_dt_f1:0.7370102523888356,max_ab_acc:0.6956521739130435,max_ab_f1:0.6919053938022012,max_rf_acc:0.7391304347826086,max_rf_f1:0.7854418655333988,max_nb_acc:0.6956521739130435,max_nb_f1:0.7287784679089027
    class_016:max_svm_acc:0.6086956521739131,max_svm_f1:0.5237613751263903,max_knn_acc:0.6956521739130435,max_knn_f1:0.707856598016781,max_dt_acc:0.782608695652174,max_dt_f1:0.8159111933395005,max_ab_acc:0.782608695652174,max_ab_f1:0.8056265984654731,max_rf_acc:0.8260869565217391,max_rf_f1:0.8594642072902944,max_nb_acc:0.782608695652174,max_nb_f1:0.7929515418502202
    class_032:max_svm_acc:0.391304347826087,max_svm_f1:0.47445652173913044,max_knn_acc:0.43478260869565216,max_knn_f1:0.47749072323741515,max_dt_acc:0.7391304347826086,max_dt_f1:0.7854418655333988,max_ab_acc:0.8695652173913043,max_ab_f1:0.880300590445518,max_rf_acc:0.8695652173913043,max_rf_f1:0.8695652173913043,max_nb_acc:0.6086956521739131,max_nb_f1:0.6376811594202898
    class_034:max_svm_acc:0.6086956521739131,max_svm_f1:0.6172688303735456,max_knn_acc:0.6521739130434783,max_knn_f1:0.6593406593406593,max_dt_acc:0.6956521739130435,max_dt_f1:0.7352865806782267,max_ab_acc:0.7391304347826086,max_ab_f1:0.7814694115465118,max_rf_acc:0.8695652173913043,max_rf_f1:0.8716736193732021,max_nb_acc:0.6521739130434783,max_nb_f1:0.6599843790679512
    class_035:max_svm_acc:0.4782608695652174,max_svm_f1:0.453505304763886,max_knn_acc:0.5652173913043478,max_knn_f1:0.5737812911725954,max_dt_acc:0.6956521739130435,max_dt_f1:0.7154282390577287,max_ab_acc:0.6521739130434783,max_ab_f1:0.6877968578735841,max_rf_acc:0.782608695652174,max_rf_f1:0.8134118397772429,max_nb_acc:0.5217391304347826,max_nb_f1:0.5881422924901185
    class_036:max_svm_acc:0.782608695652174,max_svm_f1:0.7867276887871854,max_knn_acc:0.8695652173913043,max_knn_f1:0.8907741251325556,max_dt_acc:0.782608695652174,max_dt_f1:0.8003147747393272,max_ab_acc:0.782608695652174,max_ab_f1:0.8003147747393272,max_rf_acc:0.8695652173913043,max_rf_f1:0.880300590445518,max_nb_acc:0.8260869565217391,max_nb_f1:0.8410290978541737
    '''

    # 2. Look for significant change features
    path = 'E:\\2024\\Code\\result\\features\\'
    features_variety(path)

    # 3. Feature selection
    path = 'E:\\2024\\Code\\result\\features\\feature_groups\\Binary_classification\\'
    # path = 'E:\\2024\\Code\\result\\features\\feature_groups\\Three_classification\\'
    feature_selection(path)
