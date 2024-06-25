import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from collections import Counter
import shutil
import pywt


def eeg_filter(data):
    coeffs = pywt.wavedec(data=data, wavelet='db4', level=6)
    cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))

    cD1.fill(0)
    cD2.fill(0)
    for i in range(2, len(coeffs) - 1):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet='db4')

    return rdata


def gsr_filter(data):

    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def st_filter(data):
    coeffs = pywt.wavedec(data=data, wavelet='db4', level=4)
    cA4, cD4, cD3, cD2, cD1 = coeffs

    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))

    cD1.fill(0)
    cD2.fill(0)
    cD3.fill(0)
    for i in range(3, len(coeffs)):
        if i < len(coeffs):
            coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet='db4')

    return rdata


if __name__ == '__main__':

    # EEG denoise
    raw_eeg_dir = 'E:\\2024\\Data\\EEG\\raw_data'
    for eeg_file_name in os.listdir(raw_eeg_dir):
        print(eeg_file_name)
        if eeg_file_name.endswith('.csv'):
            eeg_file_path = os.path.join(raw_eeg_dir, eeg_file_name)
            df = pd.read_csv(eeg_file_path)
            raw_eeg = df['eeg_data']

            denoised_eeg = eeg_filter(raw_eeg)[5:-5]

            p = df['P'][:len(denoised_eeg)]
            a = df['A'][:len(denoised_eeg)]
            d = df['D'][:len(denoised_eeg)]
            label = df['Label'][:len(denoised_eeg)]

            df2 = pd.DataFrame({
                'eeg_data': denoised_eeg,
                'P': p,
                'A': a,
                'D': d,
                'Label': label
            })
            eeg_save_dir = 'F:\\GitHub Project\\Data\\EEG'
            df2.to_csv(os.path.join(eeg_save_dir, eeg_file_name), index=False)

    # GSR denoise
    raw_gsr_dir = 'E:\\2024\\Data\\GSR\\raw_data'
    for gsr_file_name in os.listdir(raw_gsr_dir):
        if gsr_file_name.endswith('.csv'):
            gsr_file_path = os.path.join(raw_gsr_dir, gsr_file_name)
            df = pd.read_csv(gsr_file_path)
            raw_gsr = df['gsr_data']

            denoised_gsr = gsr_filter(raw_gsr)[3:-3]

            p = df['P'][:len(denoised_gsr)]
            a = df['A'][:len(denoised_gsr)]
            d = df['D'][:len(denoised_gsr)]
            label = df['Label'][:len(denoised_gsr)]

            df2 = pd.DataFrame({
                'gsr_data': denoised_gsr,
                'P': p,
                'A': a,
                'D': d,
                'Label': label
            })
            gsr_save_dir = 'F:\\GitHub Project\\Data\\GSR'
            df2.to_csv(os.path.join(gsr_save_dir, gsr_file_name), index=False)

    # ST denoise
    raw_st_dir = 'E:\\2024\\Data\\ST\\raw_data'
    for st_file_name in os.listdir(raw_st_dir):
        if st_file_name.endswith('.csv'):
            st_file_path = os.path.join(raw_st_dir, st_file_name)
            df = pd.read_csv(st_file_path)
            raw_st = df['temp_data']

            denoised_st = st_filter(raw_st)[1:-1]

            p = df['P'][:len(denoised_st)]
            a = df['A'][:len(denoised_st)]
            d = df['D'][:len(denoised_st)]
            label = df['Label'][:len(denoised_st)]

            df2 = pd.DataFrame({
                'st_data': denoised_st,
                'P': p,
                'A': a,
                'D': d,
                'Label': label
            })
            st_save_dir = 'F:\\GitHub Project\\Data\\ST'
            df2.to_csv(os.path.join(st_save_dir, st_file_name), index=False)
