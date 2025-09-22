import numpy as np
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
import pywt  # 小波变换库

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def extract_time_frequency_features(signal_data, fs, wavelet='db4', level=5):
    """
    提取完整的时频域特征

    参数:
    signal_data: 输入信号
    fs: 采样频率
    wavelet: 小波基函数
    level: 小波分解层数

    返回:
    features: 包含所有时频域特征的字典
    """
    N = len(signal_data)
    t = np.arange(N) / fs

    # 初始化特征字典
    features = {}

    # ==================== 1. 短时傅里叶变换 (STFT) 特征 ====================
    # 计算STFT
    f, t_stft, Zxx = signal.stft(signal_data, fs, nperseg=256, noverlap=128)

    # STFT统计特征
    stft_magnitude = np.abs(Zxx)
    features['STFT_Mean'] = np.mean(stft_magnitude)
    features['STFT_Std'] = np.std(stft_magnitude)
    features['STFT_Max'] = np.max(stft_magnitude)
    features['STFT_Min'] = np.min(stft_magnitude)
    features['STFT_Energy'] = np.sum(stft_magnitude ** 2)

    # STFT频率特征
    features['STFT_Centroid'] = np.sum(f[:, np.newaxis] * stft_magnitude, axis=0).mean() / np.sum(stft_magnitude,
                                                                                                  axis=0).mean()
    features['STFT_Bandwidth'] = np.sqrt(
        np.sum((f[:, np.newaxis] - features['STFT_Centroid']) ** 2 * stft_magnitude, axis=0).mean() / np.sum(
            stft_magnitude, axis=0).mean())

    # ==================== 2. 小波变换特征 ====================
    # 小波包分解
    wp = pywt.WaveletPacket(data=signal_data, wavelet=wavelet, mode='symmetric', maxlevel=level)

    # 小波能量特征
    total_energy = 0
    node_names = [node.path for node in wp.get_level(level, 'natural')]

    for node_name in node_names:
        node = wp[node_name]
        node_energy = np.sum(node.data ** 2)
        total_energy += node_energy
        features[f'Wavelet_{node_name}_Energy'] = node_energy

    # 小波能量百分比
    for node_name in node_names:
        features[f'Wavelet_{node_name}_Energy_Ratio'] = features[f'Wavelet_{node_name}_Energy'] / total_energy

    # 小波熵特征
    energy_values = [features[f'Wavelet_{node_name}_Energy'] for node_name in node_names]
    features['Wavelet_Energy_Entropy'] = entropy(energy_values)

    # ==================== 3. 希尔伯特-黄变换 (HHT) 相关特征 ====================
    # 计算希尔伯特变换
    analytic_signal = signal.hilbert(signal_data)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)

    # HHT特征
    features['HHT_Envelope_Mean'] = np.mean(amplitude_envelope)
    features['HHT_Envelope_Std'] = np.std(amplitude_envelope)
    features['HHT_IF_Mean'] = np.mean(instantaneous_frequency)
    features['HHT_IF_Std'] = np.std(instantaneous_frequency)
    features['HHT_IF_Max'] = np.max(instantaneous_frequency)
    features['HHT_IF_Min'] = np.min(instantaneous_frequency)

    # ==================== 4. Wigner-Ville分布特征 ====================
    # 计算Wigner-Ville分布 (简化版)
    f_wvd, t_wvd, wvd = signal.spectrogram(signal_data, fs, window=('tukey', 0.25),
                                           nperseg=128, noverlap=120, mode='psd')
    features['WVD_Mean'] = np.mean(wvd)
    features['WVD_Std'] = np.std(wvd)
    features['WVD_Max'] = np.max(wvd)

    # ==================== 5. 时频矩特征 ====================
    # 基于STFT的时频矩
    t_mesh, f_mesh = np.meshgrid(t_stft, f)
    features['TF_Centroid_Time'] = np.sum(t_mesh * stft_magnitude) / np.sum(stft_magnitude)
    features['TF_Centroid_Freq'] = np.sum(f_mesh * stft_magnitude) / np.sum(stft_magnitude)
    features['TF_Spread_Time'] = np.sqrt(
        np.sum((t_mesh - features['TF_Centroid_Time']) ** 2 * stft_magnitude) / np.sum(stft_magnitude))
    features['TF_Spread_Freq'] = np.sqrt(
        np.sum((f_mesh - features['TF_Centroid_Freq']) ** 2 * stft_magnitude) / np.sum(stft_magnitude))

    # ==================== 6. 时频熵特征 ====================
    # 时频Rényi熵
    stft_power = stft_magnitude ** 2
    stft_power_norm = stft_power / np.sum(stft_power)
    alpha = 3  # Rényi熵的阶数
    features['TF_Renyi_Entropy'] = 1 / (1 - alpha) * np.log(np.sum(stft_power_norm ** alpha))

    # 时频香农熵
    features['TF_Shannon_Entropy'] = -np.sum(stft_power_norm * np.log(stft_power_norm + 1e-10))

    return features, f, t_stft, Zxx


def proceed_time_frequency_features_raw(file_path):
    # 加载数据
    mat_data = loadmat(file_path, squeeze_me=True)
    fs = 12000  # 采样频率

    fs = 12000  # 采样频率
    for key in mat_data.keys():
        # 过滤掉以'__'开头和结尾的MATLAB自带元数据
        if key.find('_time') != -1:
            print(f"-{file_path}:__key: {key}: __{type(mat_data[key])}__, Shape: {np.shape(mat_data[key])}")
            signal_data = mat_data[key]  # 根据图片中的键名

            # 使用原始数据（不进行滤波）
            raw_signal = signal_data

            # 提取时频域特征
            tf_features, f, t_stft, Zxx = extract_time_frequency_features(raw_signal, fs)

            # 创建特征数据框
            # feature_df = pd.DataFrame.from_dict(tf_features, orient='index', columns=['Value'])
            # feature_df = pd.DataFrame.from_dict(tf_features, orient='index', columns=['Value']).T
            feature_df = pd.DataFrame([tf_features])
            print("时频域特征提取结果:")
            print(feature_df.round(6))

            # 保存特征到CSV文件
            output_path = file_path.replace('.mat', f'{key}_time_frequency_features.csv')
            feature_df.to_csv(output_path)
            print(f"\n特征已保存至: {output_path}")


if __name__ == '__main__':
    file_path = '源域数据集/12kHz_FE_data/B/0007/B007_1.mat'
    proceed_time_frequency_features_raw(file_path)

# 可视化时频分析结果
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# fig.suptitle('原始数据的时频分析结果', fontsize=16)
#
# # 1. 原始时域信号
# axes[0, 0].plot(np.arange(len(raw_signal)) / fs, raw_signal, 'b-')
# axes[0, 0].set_title('原始时域信号')
# axes[0, 0].set_xlabel('时间 [s]')
# axes[0, 0].set_ylabel('幅度')
# axes[0, 0].grid(True)
#
# # 2. STFT频谱图
# stft_magnitude = np.abs(Zxx)
# im = axes[0, 1].pcolormesh(t_stft, f, 10 * np.log10(stft_magnitude + 1e-10),
#                            shading='gouraud', cmap='viridis')
# axes[0, 1].set_title('短时傅里叶变换 (STFT) 频谱图')
# axes[0, 1].set_xlabel('时间 [s]')
# axes[0, 1].set_ylabel('频率 [Hz]')
# axes[0, 1].set_ylim(0, 2000)  # 限制频率范围以便观察
# plt.colorbar(im, ax=axes[0, 1], label='功率谱密度 [dB]')
#
# # 3. 小波包分解能量分布
# wp = pywt.WaveletPacket(data=raw_signal, wavelet='db4', mode='symmetric', maxlevel=3)
# node_names = [node.path for node in wp.get_level(3, 'natural')]
# energy_values = [np.sum(wp[node_name].data ** 2) for node_name in node_names]
# energy_ratios = [energy / sum(energy_values) for energy in energy_values]
#
# axes[1, 0].bar(range(len(energy_ratios)), energy_ratios)
# axes[1, 0].set_title('小波包能量分布 (3层分解)')
# axes[1, 0].set_xlabel('小波包节点')
# axes[1, 0].set_ylabel('能量占比')
# axes[1, 0].set_xticks(range(len(node_names)))
# axes[1, 0].set_xticklabels(node_names, rotation=45)
#
# # 4. 希尔伯特包络谱
# analytic_signal = signal.hilbert(raw_signal)
# amplitude_envelope = np.abs(analytic_signal)
# envelope_spectrum = np.abs(np.fft.fft(amplitude_envelope))[:len(amplitude_envelope) // 2] * 2 / len(amplitude_envelope)
# freqs_env = np.fft.fftfreq(len(amplitude_envelope), 1 / fs)[:len(amplitude_envelope) // 2]
#
# axes[1, 1].plot(freqs_env, envelope_spectrum)
# axes[1, 1].set_title('希尔伯特包络谱')
# axes[1, 1].set_xlabel('频率 [Hz]')
# axes[1, 1].set_ylabel('幅度')
# axes[1, 1].grid(True)
# axes[1, 1].set_xlim(0, 1000)  # 包络谱通常关注低频成分
#
# plt.tight_layout()
# plt.savefig('time_frequency_analysis_results_raw.png', dpi=300, bbox_inches='tight')
# plt.show()
