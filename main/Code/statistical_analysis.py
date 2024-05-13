import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
import statsmodels.stats.multicomp as mc
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests


# 1: data
file_dir = 'E:\\2024\\Code\\ml_code\\BSPC\\result\\features\\emotion_features'
emotions = ['Calm', 'Happy', 'Disgust', 'Surprise', 'Anger', 'Sad', 'Fear']
file_names = {emotion: f"{emotion}.csv" for emotion in emotions}

dataframes = {}
for emotion, file_name in file_names.items():
    file_path = os.path.join(file_dir, file_name)
    dataframes[emotion] = pd.read_csv(file_path)

# 2.1: t-test
results = {emotion: {} for emotion in emotions if emotion != 'Calm'}

for feature in dataframes['Calm'].columns:
    for emotion in emotions:
        if emotion != 'Calm':
            t_stat, p_value = stats.ttest_rel(dataframes['Calm'][feature], dataframes[emotion][feature])

            results[emotion][feature] = {'t_stat': t_stat, 'p_value': p_value}


for emotion in results:
    results[emotion] = pd.DataFrame(results[emotion]).T

# 2.2: Bonferroni
num_tests = len(dataframes['Calm'].columns) * (len(emotions) - 1)


bonferroni_alpha = 0.1 / num_tests

for emotion in results:
    results[emotion]['adj_p_value'] = results[emotion]['p_value'] * num_tests

    results[emotion]['significant'] = results[emotion]['adj_p_value'] < 0.1


# 2.3: Cohen's d
def cohens_d(x, y):

    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


for emotion in results:
    for feature in dataframes['Calm'].columns:
        d = cohens_d(dataframes['Calm'][feature], dataframes[emotion][feature])
        results[emotion].loc[feature, 'Cohen_d'] = d


# 2.4: save result
for emotion in results:
    save_dir = 'E:\\2024\\Code\\result\\statistical_analysis_reslut\\t-test'
    results[emotion].to_csv(save_dir + '\\' + f'results_{emotion}.csv')


# 3.1 sign-test
def sign_test(data1, data2):
    differences = np.array(data2) - np.array(data1)
    positive_diffs = np.sum(differences > 0)
    negative_diffs = np.sum(differences < 0)
    n = positive_diffs + negative_diffs
    p_value = 2 * binom.cdf(min(positive_diffs, negative_diffs), n, 0.5)

    return positive_diffs, negative_diffs, p_value


sign_test_results = {}

for emotion in emotions:
    if emotion != 'Calm':
        results = {}
        for feature in dataframes['Calm'].columns:
            data_Calm = dataframes['Calm'][feature]
            data_emotion = dataframes[emotion][feature]

            positive_diffs, negative_diffs, p_value = sign_test(data_Calm, data_emotion)
            results[feature] = {'Positive Differences': positive_diffs,
                                'Negative Differences': negative_diffs,
                                'P-value': p_value}

        sign_test_results[emotion] = pd.DataFrame(results).T

all_p_values = []
for emotion_results in sign_test_results.values():
    all_p_values.extend(emotion_results['P-value'].tolist())

# Bonferroni
corrected_results = multipletests(all_p_values, alpha=0.1, method='bonferroni')
corrected_p_values = corrected_results[1]
reject = corrected_results[0]


i = 0
for emotion, df in sign_test_results.items():
    for feature in df.index:
        df.at[feature, 'Corrected P-Value'] = corrected_p_values[i]
        df.at[feature, 'Significant'] = reject[i]
        i += 1

    save_dir = 'E:\\2024\\Code\\result\\statistical_analysis_reslut\\sign-test'
    df.to_csv(os.path.join(save_dir, f'sign_test_corrected_results_{emotion}.csv'))


# 4.1 Wilcoxon
wilcoxon_results = {emotion: {} for emotion in emotions if emotion != 'Calm'}
for feature in dataframes['Calm'].columns:
    for emotion in emotions:
        if emotion != 'Calm':
            Calm_data = dataframes['Calm'][feature].dropna()
            emotion_data = dataframes[emotion][feature].dropna()
            if len(Calm_data) == len(emotion_data):
                w_stat, w_p_value = wilcoxon(Calm_data, emotion_data)
                wilcoxon_results[emotion][feature] = {'W_stat': w_stat, 'W_p_value': w_p_value}


all_w_p_values = [result['W_p_value'] for emotion_results in wilcoxon_results.values() for result in emotion_results.values()]


corrected_results = multipletests(all_w_p_values, alpha=0.1, method='bonferroni')
corrected_p_values = corrected_results[1]
reject = corrected_results[0]


i = 0
for emotion in wilcoxon_results:
    for feature in wilcoxon_results[emotion]:
        wilcoxon_results[emotion][feature]['Corrected W_p_value'] = corrected_p_values[i]
        wilcoxon_results[emotion][feature]['Significant'] = reject[i]
        i += 1


for emotion in wilcoxon_results:
    wilcoxon_results_df = pd.DataFrame(wilcoxon_results[emotion]).T
    save_dir = 'E:\\2024\\Code\\result\\statistical_analysis_result\\Wilcoxon'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    wilcoxon_results_df.to_csv(os.path.join(save_dir, f'wilcoxon_results_{emotion}.csv'))


# 5.1 Bootstrap
# Finally, the Bootstrap method was selected to explain the association between emotion and physiological signals
def bootstrap_test(data1, data2, n_bootstrap=1000):
    """ Significance of mean differences between two data sets using Bootstrap. """
    size = len(data1)
    bootstrap_means_diff = []

    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=size, replace=True)
        sample2 = np.random.choice(data2, size=size, replace=True)
        diff = np.mean(sample1) - np.mean(sample2)
        bootstrap_means_diff.append(diff)

    observed_diff = np.mean(data1) - np.mean(data2)

    lower, upper = np.percentile(bootstrap_means_diff, [2.5, 97.5])

    p_value = np.sum(np.abs(bootstrap_means_diff) >= np.abs(observed_diff)) / n_bootstrap

    return observed_diff, lower, upper, p_value


bootstrap_results = {emotion: {} for emotion in emotions if emotion != 'Calm'}

for feature in dataframes['Calm'].columns:
    for emotion in emotions:
        if emotion != 'Calm':
            Calm_data = dataframes['Calm'][feature].dropna()
            emotion_data = dataframes[emotion][feature].dropna()

            observed_diff, lower, upper, p_value = bootstrap_test(emotion_data, Calm_data)
            bootstrap_results[emotion][feature] = {
                'Observed Difference': observed_diff,
                'CI Lower': lower,
                'CI Upper': upper,
                'P-Value': p_value
            }

for emotion in bootstrap_results:
    bootstrap_df = pd.DataFrame(bootstrap_results[emotion]).T
    bootstrap_df['Significant'] = (bootstrap_df['CI Lower'] > 0) | (bootstrap_df['CI Upper'] < 0)

    bootstrap_df['Significant_by_p_value'] = bootstrap_df['P-Value'] < 0.1

    save_dir = os.path.join('E:\\2024\\Code\\result\\statistical_analysis_result\\Bootstrap')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    bootstrap_df.to_csv(os.path.join(save_dir, f'bootstrap_results_{emotion}.csv'))
