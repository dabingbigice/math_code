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


def proceed_features_raw(file_path, all_samples_list):
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
            idx = key.find('_')
            # 切出来DA传动轴类型数据
            time_feature_dict['type'] = key[idx + 1:idx + 3]
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

            # 横向拼接三个DataFrame
            combined_features = pd.concat([time_feature_df, frequency_feature_df, time_frequency_feature_df], axis=1)
            # combined_features.set_index('key', inplace=True)
            # 将当前样本的特征添加到列表中
            all_samples_list.append(combined_features)


def get_all_file_data(root_dir):
    import os

    # 定义根目录（请替换为您电脑上的实际绝对路径，例如：C:\Users\YourName\源域数据集）

    # 创建一个空列表来存储所有文件的路径
    all_file_paths = []
    all_samples_list = []

    # 遍历根目录及其所有子文件夹
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            # 拼接完整路径，并添加到列表中
            full_path = os.path.join(foldername, filename)
            all_file_paths.append(full_path)

    # 打印所有路径
    for path in all_file_paths:
        print(path)
        proceed_features_raw(path, all_samples_list)

    if all_samples_list:

        all_samples_features = pd.concat(all_samples_list, ignore_index=False)
        # 一次性写入所有数据
        all_samples_features.set_index('key', inplace=True)
        all_samples_features.to_csv(f'{root_dir}_features.csv', index=True)
        print("所有样本的特征已保存")
    else:
        print("没有找到任何时间序列数据")


if __name__ == '__main__':
    # 12kHz_DE_data
    # all_samples_features = pd.DataFrame()
    # _12kHz_DE_data_B = "源域数据集/12kHz_DE_data/B"
    # get_all_file_data(_12kHz_DE_data_B)
    # _12kHz_DE_data_IR = "源域数据集/12kHz_DE_data/IR"
    # get_all_file_data(_12kHz_DE_data_IR)
    # _12kHz_DE_data_OR = "源域数据集/12kHz_DE_data/OR"
    # get_all_file_data(_12kHz_DE_data_OR)

    # _12kHz_FE_data_B = "源域数据集/12kHz_FE_data/B"
    # get_all_file_data(_12kHz_FE_data_B)
    # _12kHz_FE_data_IR = "源域数据集/12kHz_FE_data/IR"
    # get_all_file_data(_12kHz_FE_data_IR)
    # _12kHz_FE_data_OR = "源域数据集/12kHz_FE_data/OR"
    # get_all_file_data(_12kHz_FE_data_OR)

    # _48kHz_DE_data_B = "源域数据集/48kHz_DE_data/B"
    # get_all_file_data(_48kHz_DE_data_B)
    # _48kHz_DE_data_IR = "源域数据集/48kHz_DE_data/IR"
    # get_all_file_data(_48kHz_DE_data_IR)
    # _48kHz_DE_data_OR = "源域数据集/48kHz_DE_data/OR"
    # get_all_file_data(_48kHz_DE_data_OR)

    _48kHz_Normal_data = "源域数据集/48kHz_Normal_data"
    get_all_file_data(_48kHz_Normal_data)
