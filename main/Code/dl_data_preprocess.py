import os
import numpy as np
import pandas as pd


def slice_data(data, frequency):
    # 使用滑动窗口切割数据
    fixed_length = 5 * frequency
    stride = frequency
    data_length = len(data)
    slices = []
    for start in range(0, data_length - fixed_length + 1, stride):
        end = start + fixed_length
        new_slice = data[start:end]
        slices.append(new_slice)
    return slices


eeg_path = 'E:\\2024\\Data\\EEG\\denoised'
gsr_path = 'E:\\2024\\Data\\GSR\\denoised'
st_path = 'E:\\2024\\Data\\ST\\denoised'
hr_path = 'E:\\2024\\Data\\HR\\raw_data'

save_path = 'E:\\2024\\Data\\slices_5.16\\'
for eeg_file_name, gsr_file_name, st_file_name, hr_file_name in zip(os.listdir(eeg_path), os.listdir(gsr_path),
                                                                    os.listdir(st_path), os.listdir(hr_path)):
    if eeg_file_name == gsr_file_name == st_file_name == hr_file_name:
        print(eeg_file_name)
        eeg_file_path = os.path.join(eeg_path, eeg_file_name)
        eeg_csv_data = pd.read_csv(eeg_file_path)
        eeg_data = eeg_csv_data['data'].values
        eeg_data = (eeg_data - eeg_data.min()) / (eeg_data.max() - eeg_data.min())
        eeg_label = eeg_csv_data['label']
        eeg_freq = 512

        gsr_file_path = os.path.join(gsr_path, gsr_file_name)
        gsr_csv_data = pd.read_csv(gsr_file_path)
        gsr_data = gsr_csv_data['data'].values
        gsr_data = (gsr_data - gsr_data.min()) / (gsr_data.max() - gsr_data.min())
        gsr_freq = 300

        st_file_path = os.path.join(st_path, st_file_name)
        st_csv_data = pd.read_csv(st_file_path)
        st_data = st_csv_data['data'].values
        st_data = (st_data - st_data.min()) / (st_data.max() - st_data.min())
        st_freq = 10

        hr_file_path = os.path.join(hr_path, hr_file_name)
        hr_csv_data = pd.read_csv(hr_file_path)
        hr_data = hr_csv_data['hr_data'].values
        hr_data = (hr_data - hr_data.min()) / (hr_data.max() - hr_data.min())
        hr_freq = 1

        # 对每种数据进行切片
        sliced_eeg = slice_data(eeg_data, eeg_freq)
        sliced_gsr = slice_data(gsr_data, gsr_freq)
        sliced_temp = slice_data(st_data, st_freq)
        sliced_hr = slice_data(hr_data, hr_freq)
        min_slices = min(len(sliced_eeg), len(sliced_gsr), len(sliced_temp), len(sliced_hr))

        # 合并数据片段
        combined_slices = [np.concatenate((eeg, gsr, temp, hr)) for eeg, gsr, temp, hr in
                           zip(sliced_eeg, sliced_gsr, sliced_temp, sliced_hr)]

        # 保存每个合并后的数据片段为CSV
        for idx, slices in enumerate(combined_slices):
            # 创建DataFrame
            df = pd.DataFrame({'data': slices,
                               'label': eeg_label[:len(slices)]})
            # 保存为CSV，文件名为 slice_{idx}.csv，其中 {idx} 是数据片段的索引
            df.to_csv(save_path + eeg_file_name.split('.')[0] + '_' + str(idx) + '.csv', index=False)
