import numpy as np
from scipy import stats
from scipy.io import loadmat
import pandas as pd


def extract_complete_time_domain_features(signal_data, fs=12000):
    """
    提取完整的四大类时域特征

    参数:
    signal: 一维时域信号
    fs: 采样频率(Hz)，用于计算时间相关特征

    返回:
    features: 包含所有时域特征的字典
    """
    N = len(signal_data)
    t = np.arange(N) / fs  # 时间轴

    features = {}

    # ==================== 1. 有量纲幅值特征 ====================
    features['Peak'] = np.max(np.abs(signal_data))  # 峰值
    features['Mean'] = np.mean(signal_data)  # 均值
    features['RMS'] = np.sqrt(np.mean(signal_data ** 2))  # 均方根值
    features['Variance'] = np.var(signal_data)  # 方差
    features['Std'] = np.std(signal_data)  # 标准差
    features['Abs_Mean'] = np.mean(np.abs(signal_data))  # 绝对平均值

    # ==================== 2. 无量纲幅值特征 ====================
    # 基于峰值和RMS的特征
    features['Crest_Factor'] = features['Peak'] / features['RMS'] if features['RMS'] != 0 else 0  # 峰值因子
    features['Impulse_Factor'] = features['Peak'] / features['Abs_Mean'] if features['Abs_Mean'] != 0 else 0  # 脉冲因子

    # 基于幅值分布的特征
    features['Skewness'] = stats.skew(signal_data)  # 偏度
    features['Kurtosis'] = stats.kurtosis(signal_data, fisher=False)  # 峰度 (Fisher=False表示Pearson定义，正态分布=3)

    # 其他形状因子
    square_mean_root = (np.mean(np.sqrt(np.abs(signal_data)))) ** 2
    features['Clearance_Factor'] = features['Peak'] / square_mean_root if square_mean_root != 0 else 0  # 裕度因子
    features['Shape_Factor'] = features['RMS'] / features['Abs_Mean'] if features['Abs_Mean'] != 0 else 0  # 波形因子

    # ==================== 3. 脉冲与冲击相关特征 ====================
    # 寻找信号中的脉冲/冲击特征
    threshold = 3 * features['Std']  # 定义脉冲阈值
    pulses = signal_data[np.abs(signal_data) > threshold]

    features['Pulse_Count'] = len(pulses)  # 超过阈值的脉冲数量
    features['Pulse_Ratio'] = len(pulses) / N  # 脉冲占比
    if len(pulses) > 0:
        features['Max_Pulse_Amplitude'] = np.max(np.abs(pulses))  # 最大脉冲幅度
        features['Mean_Pulse_Amplitude'] = np.mean(np.abs(pulses))  # 平均脉冲幅度
    else:
        features['Max_Pulse_Amplitude'] = 0
        features['Mean_Pulse_Amplitude'] = 0

    # ==================== 4. 时间序列统计特征 ====================
    # 基于差分和导数的特征
    first_diff = np.diff(signal_data)
    second_diff = np.diff(signal_data, n=2)

    features['Mean_First_Diff'] = np.mean(first_diff)  # 一阶差分均值
    features['Std_First_Diff'] = np.std(first_diff)  # 一阶差分标准差
    features['Mean_Second_Diff'] = np.mean(second_diff)  # 二阶差分均值
    features['Std_Second_Diff'] = np.std(second_diff)  # 二阶差分标准差

    # 过零率
    zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
    features['Zero_Crossing_Rate'] = len(zero_crossings) / (N / fs)  # 每秒过零次数

    # 自相关特征
    autocorr = np.correlate(signal_data, signal_data, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # 取一半
    features['Autocorr_Max'] = np.max(autocorr)  # 自相关最大值
    features['Autocorr_First_Min'] = np.min(autocorr[1:100]) if len(autocorr) > 100 else np.min(
        autocorr[1:])  # 自相关第一个最小值

    # 能量和功率特征
    features['Energy'] = np.sum(signal_data ** 2)  # 信号总能量
    features['Power'] = features['Energy'] / N  # 平均功率

    return features


def proceed_time_features_raw(file_path):
    # 加载数据 - 根据您的图片信息调整路径和键名
    mat_data = loadmat(file_path, squeeze_me=True)
    for key in mat_data.keys():
        # 过滤掉以'__'开头和结尾的MATLAB自带元数据
        if key.find('_time') != -1:
            print(f"-{file_path}:__key: {key}: __{type(mat_data[key])}__, Shape: {np.shape(mat_data[key])}")
            signal_data = mat_data[key]  # 根据图片中的键名

            # 如果需要进行截取（如您之前的代码）
            # signal = signal[1023 * 41:1023 * 42]

            # 提取特征
            feature_dict = extract_complete_time_domain_features(signal_data)

            # 创建特征数据框并打印
            feature_df = pd.DataFrame.from_dict(feature_dict, orient='index').T
            print("时域特征全集提取结果:")
            print(feature_df.round(6))

            # 可选: 将特征保存到CSV文件
            output_path = file_path.replace('.mat', f'{key}_time_features.csv')
            feature_df.to_csv(output_path)
            print(f"\n特征已保存至: {output_path}")


if __name__ == '__main__':
    file_path = '源域数据集/12kHz_FE_data/B/0007/B007_1.mat'
    proceed_time_features_raw(file_path)