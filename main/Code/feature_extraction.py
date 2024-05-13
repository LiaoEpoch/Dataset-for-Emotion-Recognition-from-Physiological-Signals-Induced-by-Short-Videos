import os
import pywt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import simps
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.stats import entropy


def get_eeg_features(input_path, output_path):
    dim_p, dim_a, dim_d = [], [], []
    labels = []
    all_name = []

    all_mean = []
    all_var = []
    all_std = []
    all_data_amplitude = []
    all_mean_data_amplitude = []
    all_mean_autocorrelation = []

    all_mean_hurst_values = []
    all_total_power = []
    all_freq_in_max_fft = []

    # all_diff_entropy_of_delta = []
    all_diff_entropy_of_theta = []
    all_diff_entropy_of_alpha = []
    all_diff_entropy_of_beta = []
    all_diff_entropy_of_gamma = []

    all_power_of_delta = []
    all_power_of_theta = []
    all_power_of_alpha = []
    all_power_of_beta = []
    all_power_of_gamma = []

    all_power_ratio_of_delta = []
    all_power_ratio_of_theta = []
    all_power_ratio_of_alpha = []
    all_power_ratio_of_beta = []
    all_power_ratio_of_gamma = []

    all_total_energy = []
    all_level_0_ratio = []
    all_level_1_ratio = []
    all_level_2_ratio = []
    all_level_3_ratio = []
    all_level_4_ratio = []

    for filename in os.listdir(input_path):
        if filename.endswith('.csv'):
            print(filename)
            csv_path = os.path.join(input_path, filename)
            raw_data = pd.read_csv(csv_path)
            data = np.array(raw_data['data']).flatten()
            all_name.append(filename)

            p = raw_data['P'][:1].values
            a = raw_data['A'][:1].values
            d = raw_data['D'][:1].values
            label = raw_data['label'][:1].values

            # 时域特征
            data_mean = np.mean(data)
            data_std = np.std(data)
            data_var = np.var(data)

            data_amplitude = np.ptp(data)  # 峰-谷振幅（Peak-to-Peak Amplitude）
            mean_data_amplitude = np.mean(np.abs(data))  # 平均振幅（Mean Amplitude）

            # 计算时域自相关
            max_lag = 64  # 最大滞后值，可以根据需求调整
            autocorrelation_features = []
            for lag in range(32, max_lag + 1, 2):
                N = len(data)
                mean = np.mean(data)
                autocorr = 0
                for t in range(N - lag):
                    autocorr += (data[t] - mean) * (data[t + lag] - mean)
                autocorr /= ((N - lag) * np.var(data))
                autocorrelation_features.append(autocorr)
            mean_autocorrelation = np.mean(autocorrelation_features)  # 平均自相关特征

            # Hurst指数
            max_lag = 512
            hurst_values = []
            for lag in range(128, max_lag + 1, 16):
                lags = range(1, lag)
                segments = [data[i::lag] for i in lags]
                # 计算每个子序列的标准差
                std_devs = [np.std(segment) for segment in segments]
                # 计算累积偏差
                cum_deviation = np.cumsum(std_devs)
                # 计算Hurst指数
                hurst = np.log(cum_deviation[-1] / cum_deviation[0]) / np.log(lag)
                hurst_values.append(hurst)
            mean_hurst_values = np.mean(hurst_values)  # 平均Hurst指数

            fs = 512  # 采样率
            # 进行傅里叶变换
            fft_result = np.fft.fft(data)
            frequencies = np.fft.fftfreq(len(data), 1 / fs)

            max_fft = max(np.abs(fft_result))
            max_fft_index = np.where(np.abs(fft_result) == max_fft)
            freq_in_max_fft = np.abs(frequencies[max_fft_index])[0]  # 最大频谱幅度所在的频率

            # 计算频谱特征
            freq_bands = [(0.1, 3), (3, 7), (7, 13), (13, 30), (30, 60)]  # 不同频段的边界
            frequencies, power_spectrum = welch(data, fs=fs, nperseg=256)

            # 计算每个频段的微分熵和功率谱密度
            diff_entropy_by_band = []
            for band in freq_bands:
                low, high = band
                band_indices = np.where((frequencies >= low) & (frequencies <= high))[0]

                # 提取当前频段的功率谱密度
                band_power_spectrum = power_spectrum[band_indices]

                # 计算微分熵
                diff_entropy = entropy(band_power_spectrum)
                diff_entropy_by_band.append(diff_entropy)

            # 不同频段的微分熵
            # diff_entropy_of_delta = diff_entropy_by_band[0]
            diff_entropy_of_theta = diff_entropy_by_band[1]
            diff_entropy_of_alpha = diff_entropy_by_band[2]
            diff_entropy_of_beta = diff_entropy_by_band[3]
            diff_entropy_of_gamma = diff_entropy_by_band[4]

            # 计算每个频段的能量值
            power_by_band = []
            for band in freq_bands:
                freq_indices = np.where((frequencies >= band[0]) & (frequencies <= band[1]))[0]
                power_in_band = np.sum(power_spectrum[freq_indices])
                power_by_band.append(power_in_band)

            # 计算每个频段的能量占比
            total_power = np.sum(power_spectrum)
            power_ratio_by_band = [power / total_power for power in power_by_band]

            # 不同频段的总能量
            power_of_delta = power_by_band[0]
            power_of_theta = power_by_band[1]
            power_of_alpha = power_by_band[2]
            power_of_beta = power_by_band[3]
            power_of_gamma = power_by_band[4]

            # 不同频段的能量占比
            power_ratio_of_delta = power_ratio_by_band[0]
            power_ratio_of_theta = power_ratio_by_band[1]
            power_ratio_of_alpha = power_ratio_by_band[2]
            power_ratio_of_beta = power_ratio_by_band[3]
            power_ratio_of_gamma = power_ratio_by_band[4]

            # 小波包分解
            wavelet = 'db4'
            level = 5
            coeffs = pywt.wavedec(data, wavelet, level=level)
            # 提取小波包能量特征
            energy_features = []
            for i in range(1, level + 2):  # 从第1层到第level+1层
                if i < len(coeffs):
                    subband_energy = np.sum(np.square(coeffs[i]))
                    energy_features.append(subband_energy)
            total_energy = np.sum(energy_features)
            level_0_ratio = energy_features[0] / total_energy  # 五个分解级别上的能量占总比
            level_1_ratio = energy_features[1] / total_energy
            level_2_ratio = energy_features[2] / total_energy
            level_3_ratio = energy_features[3] / total_energy
            level_4_ratio = energy_features[4] / total_energy

            dim_p.append(p)
            dim_a.append(a)
            dim_d.append(d)
            labels.append(label)

            all_mean.append(data_mean)
            all_var.append(data_var)
            all_std.append(data_std)
            all_data_amplitude.append(data_amplitude)
            all_mean_data_amplitude.append(mean_data_amplitude)

            all_mean_autocorrelation.append(mean_autocorrelation)
            all_mean_hurst_values.append(mean_hurst_values)
            all_total_power.append(total_power)
            all_freq_in_max_fft.append(freq_in_max_fft)

            # 不同频段的微分熵
            # all_diff_entropy_of_delta.append(diff_entropy_of_delta)
            all_diff_entropy_of_theta.append(diff_entropy_of_theta)
            all_diff_entropy_of_alpha.append(diff_entropy_of_alpha)
            all_diff_entropy_of_beta.append(diff_entropy_of_beta)
            all_diff_entropy_of_gamma.append(diff_entropy_of_gamma)

            # 不同频段的总能量
            all_power_of_delta.append(power_of_delta)
            all_power_of_theta.append(power_of_theta)
            all_power_of_alpha.append(power_of_alpha)
            all_power_of_beta.append(power_of_beta)
            all_power_of_gamma.append(power_of_gamma)

            # 不同频段的能量占比
            all_power_ratio_of_delta.append(power_ratio_of_delta)
            all_power_ratio_of_theta.append(power_ratio_of_theta)
            all_power_ratio_of_alpha.append(power_ratio_of_alpha)
            all_power_ratio_of_beta.append(power_ratio_of_beta)
            all_power_ratio_of_gamma.append(power_ratio_of_gamma)

            all_total_energy.append(total_energy)
            all_level_0_ratio.append(level_0_ratio)
            all_level_1_ratio.append(level_1_ratio)
            all_level_2_ratio.append(level_2_ratio)
            all_level_3_ratio.append(level_3_ratio)
            all_level_4_ratio.append(level_4_ratio)

    dim_p = np.array(dim_p).flatten()
    dim_a = np.array(dim_a).flatten()
    dim_d = np.array(dim_d).flatten()
    labels = np.array(labels).flatten()

    all_mean = np.array(all_mean).flatten()
    all_var = np.array(all_var).flatten()
    all_std = np.array(all_std).flatten()
    all_data_amplitude = np.array(all_data_amplitude).flatten()
    all_mean_data_amplitude = np.array(all_mean_data_amplitude).flatten()
    all_mean_autocorrelation = np.array(all_mean_autocorrelation).flatten()

    all_mean_hurst_values = np.array(all_mean_hurst_values).flatten()
    all_total_power = np.array(all_total_power).flatten()
    all_freq_in_max_fft = np.array(all_freq_in_max_fft).flatten()

    all_power_of_delta = np.array(all_power_of_delta).flatten()
    all_power_of_theta = np.array(all_power_of_theta).flatten()
    all_power_of_alpha = np.array(all_power_of_alpha).flatten()
    all_power_of_beta = np.array(all_power_of_beta).flatten()
    all_power_of_gamma = np.array(all_power_of_gamma).flatten()

    all_power_ratio_of_delta = np.array(all_power_ratio_of_delta).flatten()
    all_power_ratio_of_theta = np.array(all_power_ratio_of_theta).flatten()
    all_power_ratio_of_alpha = np.array(all_power_ratio_of_alpha).flatten()
    all_power_ratio_of_beta = np.array(all_power_ratio_of_beta).flatten()
    all_power_ratio_of_gamma = np.array(all_power_ratio_of_gamma).flatten()

    all_total_energy = np.array(all_total_energy).flatten()
    all_level_0_ratio = np.array(all_level_0_ratio).flatten()
    all_level_1_ratio = np.array(all_level_1_ratio).flatten()
    all_level_2_ratio = np.array(all_level_2_ratio).flatten()
    all_level_3_ratio = np.array(all_level_3_ratio).flatten()
    all_level_4_ratio = np.array(all_level_4_ratio).flatten()

    df = pd.DataFrame({
        'file_name': all_name,
        'eeg_mean': all_mean,
        'eeg_std': all_var,
        'eeg_var': all_std,
        'eeg_amplitude': all_data_amplitude,
        'eeg_mean_amp': all_mean_data_amplitude,
        'eeg_mean_autocorrelation': all_mean_autocorrelation,
        'eeg_mean_hurst_values': all_mean_hurst_values,
        # 'eeg_diff_entropy_of_delta': all_diff_entropy_of_delta,
        'eeg_diff_entropy_of_theta': all_diff_entropy_of_theta,
        'eeg_diff_entropy_of_alpha': all_diff_entropy_of_alpha,
        'eeg_diff_entropy_of_beta': all_diff_entropy_of_beta,
        'eeg_diff_entropy_of_gamma': all_diff_entropy_of_gamma,
        'eeg_total_power': all_total_power,
        'eeg_freq_in_max_fft': all_freq_in_max_fft,
        'eeg_power_of_delta': all_power_of_delta,
        'eeg_power_of_theta': all_power_of_theta,
        'eeg_power_of_alpha': all_power_of_alpha,
        'eeg_power_of_beta': all_power_of_beta,
        'eeg_power_of_gamma': all_power_of_gamma,
        'eeg_power_ratio_of_delta': all_power_ratio_of_delta,
        'eeg_power_ratio_of_theta': all_power_ratio_of_theta,
        'eeg_power_ratio_of_alpha': all_power_ratio_of_alpha,
        'eeg_power_ratio_of_beta': all_power_ratio_of_beta,
        'eeg_power_ratio_of_gamma': all_power_ratio_of_gamma,
        'eeg_total_energy': all_total_energy,
        'eeg_level_0_ratio': all_level_0_ratio,
        'eeg_level_1_ratio': all_level_1_ratio,
        'eeg_level_2_ratio': all_level_2_ratio,
        'eeg_level_3_ratio': all_level_3_ratio,
        'eeg_level_4_ratio': all_level_4_ratio,

        'P': dim_p, 'A': dim_a, 'D': dim_d,
        'Label': labels})
    file_path = output_path + 'all_eeg_features.csv'
    df.to_csv(file_path, index=False)


def get_gsr_features(input_path, output_path):
    dim_p, dim_a, dim_d = [], [], []
    labels = []
    all_name = []

    all_min = []
    all_mean = []
    all_var = []
    all_std = []
    all_max = []

    all_third_moment = []
    all_first_diff_max = []
    all_first_diff_min = []
    all_first_diff_mean = []
    all_first_diff_median = []

    all_second_diff_max = []
    all_second_diff_min = []
    all_second_diff_mean = []
    all_second_diff_median = []
    all_mean_first_diff_negative = []

    # all_negative_diff_ratio = []
    all_local_min_count = []
    all_local_max_count = []
    all_mean_rise_time = []
    all_max_rise_time = []
    # all_min_rise_time = []
    all_mean_rise_speed = []
    all_max_rise_speed = []
    all_min_rise_speed = []

    all_data_amplitude = []
    all_mean_data_amplitude = []
    all_dynamic_range = []
    all_mean_peak_value = []
    all_mean_trough_value = []
    all_data_auc = []

    all_power = []
    all_max_psd = []
    all_min_psd = []
    all_var_psd = []
    for filename in os.listdir(input_path):
        if filename.endswith(".csv"):
            # if 'P01_N1' in filename:
            # print(filename)
            csv_path = os.path.join(input_path, filename)
            raw_data = pd.read_csv(csv_path)
            data = np.array(raw_data['data']).flatten()

            p = raw_data['P'][:1].values
            a = raw_data['A'][:1].values
            d = raw_data['D'][:1].values
            label = raw_data['label'][:1].values
            all_name.append(filename)

            # 统计特征
            data_mean = np.mean(data)  # 均值
            data_variance = np.var(data)  # 方差
            data_std = np.std(data)  # 标准差
            data_min = np.min(data)
            data_max = np.max(data)

            third_moment = np.mean((data - data_mean) ** 3) / data_std ** 3  # 三阶矩

            # 差分
            first_diff = np.diff(data)  # 一阶差分
            first_diff_max = np.max(first_diff)  # 一阶差分的最值
            first_diff_min = np.min(first_diff)
            first_diff_mean = np.mean(first_diff)  # 一阶差分的均值
            first_diff_median = np.median(first_diff)  # 一阶差分的中位数

            second_diff = np.diff(first_diff)  # 二阶差分
            second_diff_max = np.max(second_diff)  # 二阶差分的最值
            second_diff_min = np.min(second_diff)
            second_diff_mean = np.mean(second_diff)  # 二阶差分的均值
            second_diff_median = np.median(second_diff)  # 二阶差分的中位数

            first_diff_negative = []  # 负导数
            for i in first_diff:
                if i < 0:
                    first_diff_negative.append(i)
            mean_first_diff_negative = np.mean(first_diff_negative)  # 负导数的平均值

            # GSR信号中的局部极小值（波谷数）
            local_min_indices = argrelextrema(data, np.less)
            local_min_count = len(local_min_indices[0])  # 波谷数

            # 提取皮电峰值计数特征--单值
            data_peaks, _ = find_peaks(data)  # 寻找峰值点
            local_max_count = len(data_peaks)  # 峰计数

            # 平均上升时间以及平均上升速度
            if np.any(first_diff > 0):
                rise_indices = np.where(first_diff > 0)  # [0]  # 找到皮肤电反应上升的位置
                rise_indices_diff = np.diff(rise_indices)
                mean_rise_time = np.mean(rise_indices_diff)  # 平均上升时间
                max_rise_time = np.max(rise_indices_diff)  # 最大上升时间
                min_rise_time = np.min(rise_indices_diff)  # 最小上升时间
                rise_speeds = first_diff[rise_indices]  # 计算皮肤电反应上升速度
                mean_rise_speed = np.mean(rise_speeds)  # 平均上升速度
                max_rise_speed = np.max(rise_speeds)  # 最大上升速度
                min_rise_speed = np.min(rise_speeds)  # 最小上升速度
            else:
                down_indices = np.where(first_diff < 0)  # [0]  # 找到皮肤电反应上升的位置
                rise_indices_diff = np.diff(down_indices)
                mean_rise_time = np.mean(rise_indices_diff)  # 平均上升时间
                max_rise_time = np.max(rise_indices_diff)  # 最大上升时间
                min_rise_time = np.min(rise_indices_diff)  # 最小上升时间

                rise_speeds = first_diff[down_indices]  # 计算皮肤电反应上升速度
                mean_rise_speed = np.mean(rise_speeds)  # 平均上升速度
                max_rise_speed = np.max(rise_speeds)  # 最大上升速度
                min_rise_speed = np.min(rise_speeds)  # 最小上升速度

            # 振幅与平均振幅
            data_amplitude = np.ptp(data)  # 峰-谷振幅（Peak-to-Peak Amplitude）
            mean_data_amplitude = np.mean(np.abs(data))  # 平均振幅（Mean Amplitude）

            # 计算信号的幅度
            amplitude = np.abs(data)

            # 计算信号的最大幅度和最小幅度
            max_amplitude = np.max(amplitude)
            min_amplitude = np.min(amplitude)

            # 计算信号的动态范围
            dynamic_range = 10 * np.log10(max_amplitude / min_amplitude)

            # 局部最值的平均值
            local_max_indices = argrelextrema(data, np.greater)  # 找到峰值所在位置
            local_max_values = []
            for i in local_max_indices[0]:
                value = data[i]  # 由索引找到具体数据
                local_max_values.append(value)
            local_min_values = []
            for i in local_min_indices[0]:
                value = data[i]
                local_min_values.append(value)
            mean_peak_value = np.mean(local_max_values)  # 平均局部最大值
            mean_trough_value = np.mean(local_min_values)  # 平均局部最小值

            # 计算皮电曲线下面积特征--单值
            data_auc = simps(data)

            # 频谱功率
            fs = 50  # 采样频率
            frequencies, power_spectrum = welch(data, fs, nperseg=128)  # 计算频谱
            frequency_range = (frequencies >= 0) & (frequencies <= 2)  # 提取在0到5 Hz范围内的频谱功率谱
            psd_in_0_to_2Hz = power_spectrum[frequency_range]
            power_in_0_to_2Hz = np.trapz(psd_in_0_to_2Hz)  # 0 到 5 Hz范围内的频谱功率(积分)
            max_psd = np.max(psd_in_0_to_2Hz)  # 最大功率
            min_psd = np.min(psd_in_0_to_2Hz)  # 最小功率
            var_psd = np.var(psd_in_0_to_2Hz)  # 功率的方差

            dim_p.append(p)
            dim_a.append(a)
            dim_d.append(d)
            labels.append(label)

            all_min.append(data_min)
            all_mean.append(data_mean)
            all_var.append(data_variance)
            all_std.append(data_std)
            all_max.append(data_max)

            all_third_moment.append(third_moment)
            all_first_diff_max.append(first_diff_max)
            all_first_diff_min.append(first_diff_min)
            all_first_diff_mean.append(first_diff_mean)
            all_first_diff_median.append(first_diff_median)

            all_second_diff_max.append(second_diff_max)
            all_second_diff_min.append(second_diff_min)
            all_second_diff_mean.append(second_diff_mean)
            all_second_diff_median.append(second_diff_median)
            all_mean_first_diff_negative.append(mean_first_diff_negative)

            # all_negative_diff_ratio.append(negative_diff_ratio)
            all_local_min_count.append(local_min_count)
            all_local_max_count.append(local_max_count)
            all_mean_rise_time.append(mean_rise_time)
            all_max_rise_time.append(max_rise_time)
            # all_min_rise_time.append(min_rise_time)
            all_mean_rise_speed.append(mean_rise_speed)
            all_max_rise_speed.append(max_rise_speed)
            all_min_rise_speed.append(min_rise_speed)

            all_data_amplitude.append(data_amplitude)
            all_mean_data_amplitude.append(mean_data_amplitude)
            all_dynamic_range.append(dynamic_range)
            all_mean_peak_value.append(mean_peak_value)
            all_mean_trough_value.append(mean_trough_value)
            all_data_auc.append(data_auc)

            all_power.append(power_in_0_to_2Hz)
            all_max_psd.append(max_psd)
            all_min_psd.append(min_psd)
            all_var_psd.append(var_psd)

    # 将特征转换为 NumPy 数组
    all_min = np.array(all_min).flatten()
    all_mean = np.array(all_mean).flatten()
    all_var = np.array(all_var).flatten()
    all_std = np.array(all_std).flatten()
    all_max = np.array(all_max).flatten()

    all_third_moment = np.array(all_third_moment).flatten()

    all_first_diff_max = np.array(all_first_diff_max).flatten()
    all_first_diff_min = np.array(all_first_diff_min).flatten()
    all_first_diff_mean = np.array(all_first_diff_mean).flatten()
    all_first_diff_median = np.array(all_first_diff_median).flatten()
    all_second_diff_max = np.array(all_second_diff_max).flatten()

    all_second_diff_min = np.array(all_second_diff_min).flatten()
    all_second_diff_mean = np.array(all_second_diff_mean).flatten()
    all_second_diff_median = np.array(all_second_diff_median).flatten()
    all_mean_first_diff_negative = np.array(all_mean_first_diff_negative).flatten()
    # all_negative_diff_ratio = np.array(all_negative_diff_ratio).flatten()

    all_local_min_count = np.array(all_local_min_count).flatten()
    all_local_max_count = np.array(all_local_max_count).flatten()
    all_mean_rise_time = np.array(all_mean_rise_time).flatten()
    all_max_rise_time = np.array(all_max_rise_time).flatten()
    # all_min_rise_time = np.array(all_min_rise_time).flatten()

    all_mean_rise_speed = np.array(all_mean_rise_speed).flatten()
    all_max_rise_speed = np.array(all_max_rise_speed).flatten()
    all_min_rise_speed = np.array(all_min_rise_speed).flatten()
    all_data_amplitude = np.array(all_data_amplitude).flatten()
    all_mean_data_amplitude = np.array(all_mean_data_amplitude).flatten()

    all_dynamic_range = np.array(all_dynamic_range).flatten()

    all_mean_peak_value = np.array(all_mean_peak_value).flatten()
    all_mean_trough_value = np.array(all_mean_trough_value).flatten()
    all_data_auc = np.array(all_data_auc).flatten()
    all_power = np.array(all_power).flatten()
    all_max_psd = np.array(all_max_psd).flatten()

    all_min_psd = np.array(all_min_psd).flatten()
    all_var_psd = np.array(all_var_psd).flatten()

    # 将标签转换为 NumPy 数组
    dim_p = np.array(dim_p).flatten()
    dim_a = np.array(dim_a).flatten()
    dim_d = np.array(dim_d).flatten()
    labels = np.array(labels).flatten()

    df = pd.DataFrame({
        'file_name': all_name,
        'gsr_min': all_min,
        'gsr_max': all_max,
        'gsr_mean': all_mean,
        'gsr_var': all_var,
        'gsr_std': all_std,

        'gsr_third_moment': all_third_moment,
        'gsr_first_diff_max': all_first_diff_max,
        'gsr_first_diff_min': all_first_diff_min,
        'gsr_first_diff_mean': all_first_diff_mean,
        'gsr_first_diff_median': all_first_diff_median,

        'gsr_second_diff_max': all_second_diff_max,
        'gsr_second_diff_min': all_second_diff_min,
        'gsr_second_diff_mean': all_second_diff_mean,
        'gsr_second_diff_median': all_second_diff_median,
        'gsr_mean_first_diff_negative': all_mean_first_diff_negative,

        # 'gsr_negative_diff_ratio': all_negative_diff_ratio,
        'gsr_local_min_count': all_local_min_count,
        'gsr_local_max_count': all_local_max_count,
        'gsr_mean_rise_time': all_mean_rise_time,
        'gsr_max_rise_time': all_max_rise_time,

        # 'gsr_min_rise_time': all_min_rise_time,
        'gsr_mean_rise_speed': all_mean_rise_speed,
        'gsr_max_rise_speed': all_max_rise_speed,
        'gsr_min_rise_speed': all_min_rise_speed,
        'gsr_amplitude': all_data_amplitude,

        'gsr_mean_amp': all_mean_data_amplitude,
        'gsr_dynamic_range': all_dynamic_range,
        'gsr_mean_peak_value': all_mean_peak_value,
        'gsr_mean_trough_value': all_mean_trough_value,
        'gsr_data_auc': all_data_auc,

        'gsr_power': all_power,
        'gsr_max_psd': all_max_psd,
        'gsr_min_psd': all_min_psd,
        'gsr_var_psd': all_var_psd,

        'P': dim_p, 'A': dim_a, 'D': dim_d,
        'Label': labels})
    file_path = output_path + 'all_gsr_features.csv'
    df.to_csv(file_path, index=False)


def get_hr_features(input_path, output_path):
    # 初始化存储特征和标签的列表
    dim_p, dim_a, dim_d = [], [], []
    labels = []
    all_name = []

    all_mean = []
    all_var = []
    all_std = []
    all_median = []
    all_max = []

    all_min = []
    all_third_moment = []
    all_first_diff_max = []
    all_first_diff_min = []
    all_first_diff_mean = []

    # all_first_diff_median = []
    all_mean_first_diff_negative = []
    # all_negative_ratio = []
    all_local_min_count = []
    all_local_max_count = []

    all_mean_rise_time = []
    all_max_rise_time = []
    # all_min_rise_time = []
    all_mean_rise_speed = []
    all_max_rise_speed = []
    # all_min_rise_speed = []
    all_data_amplitude = []
    all_mean_data_amplitude = []
    all_mean_peak_value = []

    all_mean_trough_value = []
    all_data_auc = []
    all_power = []
    all_max_psd = []
    all_min_psd = []

    all_var_psd = []
    all_spectral_power_0 = []
    all_spectral_power_1 = []
    all_power_ratio_0 = []
    all_power_ratio_1 = []

    all_power_ratio_01 = []

    for filename in os.listdir(input_path):
        if filename.endswith('.csv'):
            # if 'P04_N5' in filename:
            print(filename)
            csv_path = os.path.join(input_path, filename)
            raw_data = pd.read_csv(csv_path)
            data = np.array(raw_data['hr_data']).flatten()

            # p = raw_data['P'][:1].values
            # a = raw_data['A'][:1].values
            # d = raw_data['D'][:1].values
            label = raw_data['Label'][:1].values
            all_name.append(filename)
            # 提取特征
            data_mean = np.mean(data)
            data_std = np.std(data)
            data_var = np.var(data)
            data_median = np.median(data)
            data_min = np.min(data)
            data_max = np.max(data)

            third_moment = np.mean((data - data_mean) ** 3) / data_std ** 3  # 三阶矩

            # 差分
            first_diff = np.diff(data)  # 一阶差分
            first_diff_max = np.max(first_diff)  # 一阶差分的最值
            first_diff_min = np.min(first_diff)
            first_diff_mean = np.mean(first_diff)  # 一阶差分的均值
            first_diff_median = np.median(first_diff)  # 一阶差分的中位数

            first_diff_negative = []  # 负导数
            for i in first_diff:
                if i < 0:
                    first_diff_negative.append(i)
            mean_first_diff_negative = np.mean(first_diff_negative)  # 负导数的平均值

            # GSR信号中的局部极小值（波谷数）
            local_min_indices = argrelextrema(data, np.less)
            local_min_count = len(local_min_indices[0])  # 波谷数

            # 提取皮电峰值计数特征--单值
            data_peaks, _ = find_peaks(data)  # 寻找峰值点
            local_max_count = len(data_peaks)  # 峰计数

            # 平均上升时间以及平均上升速度
            rise_indices = np.where(first_diff > 0)  # [0]  # 找到皮肤电反应上升的位置
            rise_indices_diff = np.diff(rise_indices)
            mean_rise_time = np.mean(rise_indices_diff)  # 平均上升时间
            max_rise_time = np.max(rise_indices_diff)  # 最大上升时间
            # min_rise_time = np.min(rise_indices_diff)  # 最小上升时间

            rise_speeds = first_diff[rise_indices]  # 计算皮肤电反应上升速度
            mean_rise_speed = np.mean(rise_speeds)  # 平均上升速度
            max_rise_speed = np.max(rise_speeds)  # 最大上升速度
            # min_rise_speed = np.min(rise_speeds)  # 最小上升速度

            # 振幅与平均振幅
            data_amplitude = np.ptp(data)  # 峰-谷振幅（Peak-to-Peak Amplitude）
            mean_amplitude = np.mean(np.abs(data))  # 平均振幅（Mean Amplitude）

            # 局部最值的平均值
            local_max_indices = argrelextrema(data, np.greater)  # 找到峰值所在位置
            local_max_values = []
            for i in local_max_indices[0]:
                value = data[i]  # 由索引找到具体数据
                local_max_values.append(value)
            local_min_values = []
            for i in local_min_indices[0]:
                value = data[i]
                local_min_values.append(value)
            mean_peak_value = np.mean(local_max_values)  # 平均局部最大值
            mean_trough_value = np.mean(local_min_values)  # 平均局部最小值

            # 计算皮电曲线下面积特征--单值
            data_auc = simps(data)

            # 频谱功率
            fs = 1  # 采样频率
            frequencies, power_spectrum = welch(data, fs, nperseg=20)  # 计算频谱
            frequency_range_0 = (frequencies >= 0) & (frequencies <= 1)  # 提取在 0 到 1 Hz范围内的频谱功率谱
            frequency_range_1 = (frequencies >= 0) & (frequencies <= 0.15)  # 提取在 0 到 0.15 Hz范围内的频谱功率谱
            frequency_range_2 = (frequencies >= 0.15) & (frequencies <= 1)  # 提取在 0.15 到 1 Hz范围内的频谱功率谱

            psd_in_0_to_1Hz = power_spectrum[frequency_range_0]
            power_in_0_to_1Hz = np.trapz(psd_in_0_to_1Hz)  # 0 到 1 Hz范围内的频谱功率(积分)
            max_psd = np.max(psd_in_0_to_1Hz)  # 最大功率
            min_psd = np.min(psd_in_0_to_1Hz)  # 最小功率
            var_psd = np.var(psd_in_0_to_1Hz)  # 功率的方差

            psd_in_0_to_0_15Hz = power_spectrum[frequency_range_1]
            power_in_0_to_0_15Hz = np.trapz(psd_in_0_to_0_15Hz)  # 0 到 0.15 Hz范围内的频谱功率(积分)
            power_in_0_to_0_15Hz_ratio = power_in_0_to_0_15Hz / power_in_0_to_1Hz  # 0 到 0.15 Hz范围内的频谱功率占总比

            psd_in_0_15_to_1Hz = power_spectrum[frequency_range_2]
            power_in_0_15_to_1Hz = np.trapz(psd_in_0_15_to_1Hz)  # 0.15 到 1 Hz范围内的频谱功率(积分)
            power_in_0_15_to_1Hz_ratio = power_in_0_15_to_1Hz / power_in_0_to_1Hz  # 0.15 到 1 Hz范围内的频谱功率占总比

            power_0_15_to_0_85_ratio = power_in_0_to_0_15Hz / power_in_0_15_to_1Hz  # 0 到 0.15 Hz频谱功率与0.15 到 1 Hz比

            dim_p.append(p)
            dim_a.append(a)
            dim_d.append(d)
            labels.append(label)

            all_mean.append(data_mean)
            all_var.append(data_var)
            all_std.append(data_std)
            all_median.append(data_median)
            all_max.append(data_max)

            all_min.append(data_min)
            all_third_moment.append(third_moment)
            all_first_diff_max.append(first_diff_max)
            all_first_diff_min.append(first_diff_min)
            all_first_diff_mean.append(first_diff_mean)

            # all_first_diff_median.append(first_diff_median)
            all_mean_first_diff_negative.append(mean_first_diff_negative)
            # all_negative_ratio.append(negative_ratio)
            all_local_min_count.append(local_min_count)
            all_local_max_count.append(local_max_count)

            all_mean_rise_time.append(mean_rise_time)
            all_max_rise_time.append(max_rise_time)
            # all_min_rise_time.append(min_rise_time)
            all_mean_rise_speed.append(mean_rise_speed)
            all_max_rise_speed.append(max_rise_speed)
            # all_min_rise_speed.append(min_rise_speed)
            all_data_amplitude.append(data_amplitude)
            all_mean_data_amplitude.append(mean_amplitude)
            all_mean_peak_value.append(mean_peak_value)

            all_mean_trough_value.append(mean_trough_value)
            all_data_auc.append(data_auc)
            all_power.append(power_in_0_to_1Hz)
            all_max_psd.append(max_psd)
            all_min_psd.append(min_psd)

            all_var_psd.append(var_psd)
            all_spectral_power_0.append(power_in_0_to_0_15Hz)
            all_spectral_power_1.append(power_in_0_15_to_1Hz)
            all_power_ratio_0.append(power_in_0_to_0_15Hz_ratio)
            all_power_ratio_1.append(power_in_0_15_to_1Hz_ratio)

            all_power_ratio_01.append(power_0_15_to_0_85_ratio)

    # 将特征转换为 NumPy 数组
    all_mean = np.array(all_mean).flatten()
    all_var = np.array(all_var).flatten()
    all_std = np.array(all_std).flatten()
    all_median = np.array(all_median).flatten()
    all_min = np.array(all_min).flatten()
    all_max = np.array(all_max).flatten()

    all_third_moment = np.array(all_third_moment).flatten()
    all_first_diff_max = np.array(all_first_diff_max).flatten()
    all_first_diff_min = np.array(all_first_diff_min).flatten()
    all_first_diff_mean = np.array(all_first_diff_mean).flatten()
    # all_first_diff_median = np.array(all_first_diff_median).flatten()

    all_mean_first_diff_negative = np.array(all_mean_first_diff_negative).flatten()
    # all_negative_ratio = np.array(all_negative_ratio).flatten()
    all_local_min_count = np.array(all_local_min_count).flatten()
    all_local_max_count = np.array(all_local_max_count).flatten()
    all_mean_rise_time = np.array(all_mean_rise_time).flatten()

    all_max_rise_time = np.array(all_max_rise_time).flatten()
    # all_min_rise_time = np.array(all_min_rise_time).flatten()
    all_mean_rise_speed = np.array(all_mean_rise_speed).flatten()
    all_max_rise_speed = np.array(all_max_rise_speed).flatten()
    # all_min_rise_speed = np.array(all_min_rise_speed).flatten()

    all_data_amplitude = np.array(all_data_amplitude).flatten()
    all_mean_data_amplitude = np.array(all_mean_data_amplitude).flatten()
    all_mean_peak_value = np.array(all_mean_peak_value).flatten()
    all_mean_trough_value = np.array(all_mean_trough_value).flatten()
    all_data_auc = np.array(all_data_auc).flatten()

    all_power = np.array(all_power).flatten()
    all_max_psd = np.array(all_max_psd).flatten()
    all_min_psd = np.array(all_min_psd).flatten()
    all_var_psd = np.array(all_var_psd).flatten()
    all_spectral_power_0 = np.array(all_spectral_power_0).flatten()

    all_spectral_power_1 = np.array(all_spectral_power_1).flatten()
    all_power_ratio_0 = np.array(all_power_ratio_0).flatten()
    all_power_ratio_1 = np.array(all_power_ratio_1).flatten()
    all_power_ratio_01 = np.array(all_power_ratio_01).flatten()

    # 将标签转换为 NumPy 数组
    dim_p = np.array(dim_p).flatten()
    dim_a = np.array(dim_a).flatten()
    dim_d = np.array(dim_d).flatten()
    labels = np.array(labels).flatten()

    df = pd.DataFrame({
        'file_name': all_name,
        'hr_mean': all_mean,
        'hr_var': all_var,
        'hr_std': all_std,
        'hr_median': all_median,
        'hr_min': all_min,
        'hr_max': all_max,

        'hr_third_moment': all_third_moment,
        'hr_first_diff_max': all_first_diff_max,
        'hr_first_diff_min': all_first_diff_min,
        'hr_first_diff_mean': all_first_diff_mean,
        # 'hr_first_diff_median': all_first_diff_median,

        'hr_mean_first_diff_negative': all_mean_first_diff_negative,
        # 'hr_negative_ratio': all_negative_ratio,
        'hr_local_min_count': all_local_min_count,
        'hr_local_max_count': all_local_max_count,
        'hr_mean_rise_time': all_mean_rise_time,

        'hr_max_rise_time': all_max_rise_time,
        # 'hr_min_rise_time': all_min_rise_time,
        'hr_mean_rise_speed': all_mean_rise_speed,
        'hr_max_rise_speed': all_max_rise_speed,
        # 'hr_min_rise_speed': all_min_rise_speed,

        'hr_amplitude': all_data_amplitude,
        'hr_mean_amplitude': all_mean_data_amplitude,
        'hr_mean_peak_value': all_mean_peak_value,
        'hr_mean_trough_value': all_mean_trough_value,
        'hr_data_auc': all_data_auc,

        'hr_power': all_power,
        'hr_max_psd': all_max_psd,
        'hr_min_psd': all_min_psd,
        'hr_var_psd': all_var_psd,
        'hr_spectral_power_0': all_spectral_power_0,

        'hr_spectral_power_1': all_spectral_power_1,
        'hr_power_ratio_0': all_power_ratio_0,
        'hr_power_ratio_1': all_power_ratio_1,
        'hr_power_ratio_01': all_power_ratio_01,

        'P': dim_p, 'A': dim_a, 'D': dim_d,
        'Label': labels})
    file_path = output_path + 'all_hr_features.csv'
    df.to_csv(file_path, index=False)


def get_st_features(input_path, output_path):
    # 初始化存储特征和标签的列表
    dim_p, dim_a, dim_d = [], [], []
    labels = []
    all_name = []

    all_mean = []
    all_var = []
    all_std = []
    all_median = []
    all_max = []

    all_min = []
    all_third_moment = []
    all_first_diff_max = []
    all_first_diff_min = []
    all_first_diff_mean = []

    all_first_diff_median = []
    all_mean_first_diff_negative = []
    # all_negative_ratio = []
    all_local_min_count = []
    all_local_max_count = []

    all_mean_rise_time = []
    all_max_rise_time = []
    # all_min_rise_time = []
    all_mean_rise_speed = []
    all_max_rise_speed = []
    all_min_rise_speed = []
    all_data_amplitude = []
    all_mean_data_amplitude = []
    all_mean_peak_value = []

    all_mean_trough_value = []
    all_data_auc = []
    all_power = []
    all_max_psd = []
    all_min_psd = []

    all_var_psd = []
    all_spectral_power_0 = []
    all_spectral_power_1 = []
    all_power_ratio_0 = []
    all_power_ratio_1 = []

    all_power_ratio_01 = []

    for filename in os.listdir(input_path):
        if filename.endswith('.csv'):
            # if 'P04_N5' in filename:
            print(filename)
            csv_path = os.path.join(input_path, filename)
            raw_data = pd.read_csv(csv_path)
            data = np.array(raw_data['data']).flatten()
            all_name.append(filename)
            p = raw_data['P'][:1].values
            a = raw_data['A'][:1].values
            d = raw_data['D'][:1].values
            label = raw_data['label'][:1].values

            # 提取特征
            data_mean = np.mean(data)
            data_std = np.std(data)
            data_var = np.var(data)
            data_median = np.median(data)
            data_min = np.min(data)
            data_max = np.max(data)

            third_moment = np.mean((data - data_mean) ** 3) / data_std ** 3  # 三阶矩

            # 差分
            first_diff = np.diff(data)  # 一阶差分
            first_diff_max = np.max(first_diff)  # 一阶差分的最值
            first_diff_min = np.min(first_diff)
            first_diff_mean = np.mean(first_diff)  # 一阶差分的均值
            first_diff_median = np.median(first_diff)  # 一阶差分的中位数

            first_diff_negative = []  # 负导数
            for i in first_diff:
                if i < 0:
                    first_diff_negative.append(i)
            mean_first_diff_negative = np.mean(first_diff_negative)  # 负导数的平均值

            # GSR信号中的局部极小值（波谷数）
            local_min_indices = argrelextrema(data, np.less)  # argrelextrema的返回值是局部最值的索引构成的数组，而不是局部最值本身
            local_min_count = len(local_min_indices[0])  # 波谷数

            # 提取皮电峰值计数特征--单值
            data_peaks, _ = find_peaks(data)  # 寻找峰值点
            local_max_count = len(data_peaks)  # 峰计数

            # 平均上升时间以及平均上升速度
            rise_indices = np.where(first_diff > 0)  # [0]  # 找到皮肤电反应上升的位置
            rise_indices_diff = np.diff(rise_indices)
            mean_rise_time = np.mean(rise_indices_diff)  # 平均上升时间
            max_rise_time = np.max(rise_indices_diff)  # 最大上升时间
            # min_rise_time = np.min(rise_indices_diff)  # 最小上升时间

            rise_speeds = first_diff[rise_indices]  # 计算皮肤电反应上升速度
            mean_rise_speed = np.mean(rise_speeds)  # 平均上升速度
            max_rise_speed = np.max(rise_speeds)  # 最大上升速度
            min_rise_speed = np.min(rise_speeds)  # 最小上升速度

            # 振幅与平均振幅
            data_amplitude = np.ptp(data)  # 峰-谷振幅（Peak-to-Peak Amplitude）
            mean_amplitude = np.mean(np.abs(data))  # 平均振幅（Mean Amplitude）

            # 局部最值的平均值
            local_max_indices = argrelextrema(data, np.greater)  # 找到峰值所在位置
            local_max_values = []
            for i in local_max_indices[0]:
                value = data[i]  # 由索引找到具体数据
                local_max_values.append(value)
            local_min_values = []
            for i in local_min_indices[0]:
                value = data[i]
                local_min_values.append(value)
            mean_peak_value = np.mean(local_max_values)  # 平均局部最大值
            mean_trough_value = np.mean(local_min_values)  # 平均局部最小值

            # 计算皮电曲线下面积特征--单值
            data_auc = simps(data)

            # 频谱功率
            fs = 5  # 采样频率
            frequencies, power_spectrum = welch(data, fs, nperseg=20)  # 计算频谱
            frequency_range_0 = (frequencies >= 0) & (frequencies <= 5)  # 提取在 0 到 1 Hz范围内的频谱功率谱
            frequency_range_1 = (frequencies >= 0) & (frequencies <= 1)  # 提取在 0 到 0.15 Hz范围内的频谱功率谱
            frequency_range_2 = (frequencies >= 1) & (frequencies <= 5)  # 提取在 0.15 到 1 Hz范围内的频谱功率谱

            psd_in_0_to_5Hz = power_spectrum[frequency_range_0]
            power_in_0_to_5Hz = np.trapz(psd_in_0_to_5Hz)  # 0 到 1 Hz范围内的频谱功率(积分)
            max_psd = np.max(psd_in_0_to_5Hz)  # 最大功率
            min_psd = np.min(psd_in_0_to_5Hz)  # 最小功率
            var_psd = np.var(psd_in_0_to_5Hz)  # 功率的方差

            psd_in_0_to_1Hz = power_spectrum[frequency_range_1]
            power_in_0_to_1Hz = np.trapz(psd_in_0_to_1Hz)  # 0 到 0.15 Hz范围内的频谱功率(积分)
            power_in_0_to_1Hz_ratio = power_in_0_to_1Hz / power_in_0_to_5Hz  # 0 到 0.15 Hz范围内的频谱功率占总比

            psd_in_0_15_to_1Hz = power_spectrum[frequency_range_2]
            power_in_1_to_5Hz = np.trapz(psd_in_0_15_to_1Hz)  # 0.15 到 1 Hz范围内的频谱功率(积分)
            power_in_1_to_5Hz_ratio = power_in_1_to_5Hz / power_in_0_to_5Hz  # 0.15 到 1 Hz范围内的频谱功率占总比

            power_1_to_4_ratio = power_in_0_to_1Hz / power_in_1_to_5Hz  # 0 到 0.15 Hz频谱功率与0.15 到 1 Hz比

            dim_p.append(p)
            dim_a.append(a)
            dim_d.append(d)
            labels.append(label)

            all_mean.append(data_mean)
            all_var.append(data_var)
            all_std.append(data_std)
            all_median.append(data_median)
            all_max.append(data_max)

            all_min.append(data_min)
            all_third_moment.append(third_moment)
            all_first_diff_max.append(first_diff_max)
            all_first_diff_min.append(first_diff_min)
            all_first_diff_mean.append(first_diff_mean)

            all_first_diff_median.append(first_diff_median)
            all_mean_first_diff_negative.append(mean_first_diff_negative)
            # all_negative_ratio.append(negative_ratio)
            all_local_min_count.append(local_min_count)
            all_local_max_count.append(local_max_count)

            all_mean_rise_time.append(mean_rise_time)
            all_max_rise_time.append(max_rise_time)
            # all_min_rise_time.append(min_rise_time)
            all_mean_rise_speed.append(mean_rise_speed)
            all_max_rise_speed.append(max_rise_speed)
            all_min_rise_speed.append(min_rise_speed)
            all_data_amplitude.append(data_amplitude)
            all_mean_data_amplitude.append(mean_amplitude)
            all_mean_peak_value.append(mean_peak_value)

            all_mean_trough_value.append(mean_trough_value)
            all_data_auc.append(data_auc)
            all_power.append(power_in_0_to_1Hz)
            all_max_psd.append(max_psd)
            all_min_psd.append(min_psd)

            all_var_psd.append(var_psd)
            all_spectral_power_0.append(power_in_0_to_1Hz)
            all_spectral_power_1.append(power_in_1_to_5Hz)
            all_power_ratio_0.append(power_in_0_to_1Hz_ratio)
            all_power_ratio_1.append(power_in_1_to_5Hz_ratio)

            all_power_ratio_01.append(power_1_to_4_ratio)

    # 将特征转换为 NumPy 数组
    all_mean = np.array(all_mean).flatten()
    all_var = np.array(all_var).flatten()
    all_std = np.array(all_std).flatten()
    all_median = np.array(all_median).flatten()
    all_min = np.array(all_min).flatten()
    all_max = np.array(all_max).flatten()

    all_third_moment = np.array(all_third_moment).flatten()
    all_first_diff_max = np.array(all_first_diff_max).flatten()
    all_first_diff_min = np.array(all_first_diff_min).flatten()
    all_first_diff_mean = np.array(all_first_diff_mean).flatten()
    all_first_diff_median = np.array(all_first_diff_median).flatten()

    all_mean_first_diff_negative = np.array(all_mean_first_diff_negative).flatten()
    # all_negative_ratio = np.array(all_negative_ratio).flatten()
    all_local_min_count = np.array(all_local_min_count).flatten()
    all_local_max_count = np.array(all_local_max_count).flatten()
    all_mean_rise_time = np.array(all_mean_rise_time).flatten()

    all_max_rise_time = np.array(all_max_rise_time).flatten()
    # all_min_rise_time = np.array(all_min_rise_time).flatten()
    all_mean_rise_speed = np.array(all_mean_rise_speed).flatten()
    all_max_rise_speed = np.array(all_max_rise_speed).flatten()
    all_min_rise_speed = np.array(all_min_rise_speed).flatten()

    all_data_amplitude = np.array(all_data_amplitude).flatten()
    all_mean_data_amplitude = np.array(all_mean_data_amplitude).flatten()
    all_mean_peak_value = np.array(all_mean_peak_value).flatten()
    all_mean_trough_value = np.array(all_mean_trough_value).flatten()
    all_data_auc = np.array(all_data_auc).flatten()

    all_power = np.array(all_power).flatten()
    all_max_psd = np.array(all_max_psd).flatten()
    all_min_psd = np.array(all_min_psd).flatten()
    all_var_psd = np.array(all_var_psd).flatten()
    all_spectral_power_0 = np.array(all_spectral_power_0).flatten()

    all_spectral_power_1 = np.array(all_spectral_power_1).flatten()
    all_power_ratio_0 = np.array(all_power_ratio_0).flatten()
    all_power_ratio_1 = np.array(all_power_ratio_1).flatten()
    all_power_ratio_01 = np.array(all_power_ratio_01).flatten()

    # 将标签转换为 NumPy 数组
    dim_p = np.array(dim_p).flatten()
    dim_a = np.array(dim_a).flatten()
    dim_d = np.array(dim_d).flatten()
    labels = np.array(labels).flatten()

    df = pd.DataFrame({
        'file_name': all_name,
        'st_mean': all_mean,
        'st_var': all_var,
        'st_std': all_std,
        'st_median': all_median,
        'st_min': all_min,
        'st_max': all_max,

        'st_third_moment': all_third_moment,
        'st_first_diff_max': all_first_diff_max,
        'st_first_diff_min': all_first_diff_min,
        'st_first_diff_mean': all_first_diff_mean,
        'st_first_diff_median': all_first_diff_median,

        'st_mean_first_diff_negative': all_mean_first_diff_negative,
        # 'st_negative_ratio': all_negative_ratio,
        'st_local_min_count': all_local_min_count,
        'st_local_max_count': all_local_max_count,
        'st_mean_rise_time': all_mean_rise_time,

        'st_max_rise_time': all_max_rise_time,
        # 'st_min_rise_time': all_min_rise_time,
        'st_mean_rise_speed': all_mean_rise_speed,
        'st_max_rise_speed': all_max_rise_speed,
        'st_min_rise_speed': all_min_rise_speed,

        'st_amplitude': all_data_amplitude,
        'st_mean_amplitude': all_mean_data_amplitude,
        'st_mean_peak_value': all_mean_peak_value,
        'st_mean_trough_value': all_mean_trough_value,
        'st_data_auc': all_data_auc,

        'st_power': all_power,
        'st_max_psd': all_max_psd,
        'st_min_psd': all_min_psd,
        'st_var_psd': all_var_psd,
        'st_spectral_power_0': all_spectral_power_0,

        'st_spectral_power_1': all_spectral_power_1,
        'st_power_ratio_0': all_power_ratio_0,
        'st_power_ratio_1': all_power_ratio_1,
        'st_power_ratio_01': all_power_ratio_01,

        'P': dim_p, 'A': dim_a, 'D': dim_d,
        'Label': labels})
    file_path = output_path + 'all_st_features.csv'
    df.to_csv(file_path, index=False)


if __name__ == '__main__':
    '''
    Feature extraction - Toggle file paths as needed
    input_path: The location of the feature data to be extracted
    output_path: The address where the attribute is stored
    '''
    input_path = 'E:\\2024\\Data\\EEG\\denoised\\'
    output_path = 'E:\\2024\\Code\\result\\features\\'
    get_eeg_features(input_path, output_path)

    input_path = 'E:\\2024\\Data\\GSR\\denoised\\'
    output_path = 'E:\\2024\\Code\\result\\features\\'
    get_gsr_features(input_path, output_path)

    input_path = 'E:\\2024\\Data\\HR\\raw_data\\'
    output_path = 'E:\\2024\\Code\\result\\features\\'
    get_hr_features(input_path, output_path)

    input_path = 'E:\\2024\\Data\\ST\\denoised\\'
    output_path = 'E:\\2024\\Code\\result\\features\\'
    get_st_features(input_path, output_path)

    # fill empty values
    path = 'E:\\2024\\Code\\result\\features\\'
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path)

            # 查找空值并用0填充
            df.fillna(0, inplace=True)

            # 保存修改后的表格
            df.to_csv(file_path, index=False)
