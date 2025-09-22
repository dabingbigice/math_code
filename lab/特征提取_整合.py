import os

import numpy as np
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kurtosis, skew, gmean, hmean
from 特征提取_时域 import extract_complete_time_domain_features
from 特征提取_频域 import extract_frequency_domain_features
from 特征提取_时频域 import extract_time_frequency_features


def proceed_features_raw(all_samples_features, file_path):
    # 加载数据
    mat_data = loadmat(file_path, squeeze_me=True)
    fs = 12000  # 采样频率
    for key in mat_data.keys():
        # 过滤掉以'__'开头和结尾的MATLAB自带元数据
        if key.find('_time') != -1:
            print(f"-{file_path}:__key: {key}: __{type(mat_data[key])}__, Shape: {np.shape(mat_data[key])}")
            signal_data = mat_data[key]  # 根据图片中的键名

            filtered_signal = signal_data  # 如果没有滤波，使用原始信号

            # 提取频域特征
            freq_features, freqs, fft_amp, freqs_psd, psd = extract_frequency_domain_features(filtered_signal, fs)
            # 创建特征数据框
            frequency_feature_df = pd.DataFrame.from_dict(freq_features, orient='index').T
            print("频域特征提取结果:")
            print(frequency_feature_df.round(6))

            # 提取时域特征
            time_feature_dict = extract_complete_time_domain_features(filtered_signal)
            # 添加文件名称
            time_feature_dict['key'] = key
            # 创建特征数据框并打印
            time_feature_df = pd.DataFrame.from_dict(time_feature_dict, orient='index').T
            print("时域特征全集提取结果:")
            print(time_feature_df.round(6))

            # 提取时、频域特征
            time_features, f, t_stft, Zxx = extract_time_frequency_features(filtered_signal, fs)
            # feature_df = pd.DataFrame.from_dict(tf_features, orient='index', columns=['Value'])
            # feature_df = pd.DataFrame([tf_features])
            time_frequency_feature_df = pd.DataFrame.from_dict(time_features, orient='index').T
            print("时频域特征提取结果:")
            print(time_frequency_feature_df.round(6))


            combined_features = pd.concat([time_feature_df, frequency_feature_df, time_frequency_feature_df], axis=1)
            combined_features = combined_features.set_index('key')
            # 将当前样本的特征添加到总DataFrame中
            all_samples_features = pd.concat([all_samples_features, combined_features], axis=0, ignore_index=False)

        all_samples_features.to_csv('all_samples_combined_features.csv', index=[key])


if __name__ == '__main__':
    all_samples_features = pd.DataFrame()
    proceed_features_raw(all_samples_features, '源域数据集/12kHz_FE_data/B/0007/B007_1.mat')
