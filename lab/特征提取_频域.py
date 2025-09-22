import numpy as np
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kurtosis, skew, gmean, hmean

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def extract_frequency_domain_features(signal_data, fs):
    """
    提取完整的频域特征

    参数:
    signal_data: 输入信号
    fs: 采样频率

    返回:
    features: 包含所有频域特征的字典
    freqs: 频率轴
    fft_amplitude: 幅度谱
    """
    N = len(signal_data)

    # 计算FFT
    fft_values = np.fft.fft(signal_data)
    fft_amplitude = np.abs(fft_values[:N // 2]) * 2 / N  # 单边幅度谱
    freqs = np.fft.fftfreq(N, 1 / fs)[:N // 2]  # 单边频率轴

    # 计算功率谱
    power_spectrum = fft_amplitude ** 2

    # 计算功率谱密度 (PSD)

    freqs_psd, psd = signal.welch(signal_data, fs, nperseg=1024)

    # 初始化特征字典
    features = {}

    # ==================== 1. 基本频谱特征 ====================
    features['Spectral_Mean'] = np.mean(fft_amplitude)
    features['Spectral_Std'] = np.std(fft_amplitude)
    features['Spectral_RMS'] = np.sqrt(np.mean(power_spectrum))
    features['Spectral_Skewness'] = skew(fft_amplitude)
    features['Spectral_Kurtosis'] = kurtosis(fft_amplitude)

    # ==================== 2. 频率中心特征 ====================
    features['Mean_Frequency'] = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
    features['Median_Frequency'] = freqs[np.where(np.cumsum(power_spectrum) >= 0.5 * np.sum(power_spectrum))[0][0]]
    features['Peak_Frequency'] = freqs[np.argmax(fft_amplitude)]
    features['Peak_Amplitude'] = np.max(fft_amplitude)

    # ==================== 3. 频带能量特征 ====================
    # 定义频带 (可根据实际情况调整)
    band_edges = [0, 100, 500, 1000, 2000, 3000, 6000]  # Hz

    for i in range(len(band_edges) - 1):
        low_freq = band_edges[i]
        high_freq = band_edges[i + 1]
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)

        if np.any(band_mask):
            band_power = np.sum(power_spectrum[band_mask])
            band_name = f'Band_{low_freq}_{high_freq}Hz'
            features[f'{band_name}_Power'] = band_power
            features[f'{band_name}_Power_Ratio'] = band_power / np.sum(power_spectrum)

    # ==================== 4. 频谱形状特征 ====================
    features['Spectral_Entropy'] = -np.sum((power_spectrum / np.sum(power_spectrum)) *
                                           np.log(power_spectrum / np.sum(power_spectrum) + 1e-10))

    features['Spectral_Flatness'] = gmean(fft_amplitude + 1e-10) / np.mean(fft_amplitude + 1e-10)
    features['Spectral_Crest'] = np.max(fft_amplitude) / np.mean(fft_amplitude)
    features['Spectral_Slope'] = np.polyfit(freqs, fft_amplitude, 1)[0]  # 频谱斜率

    # ==================== 5. 峰值相关特征 ====================
    # 找到所有峰值
    peaks, properties = signal.find_peaks(fft_amplitude, height=np.mean(fft_amplitude) * 0.1)

    if len(peaks) > 0:
        features['Peak_Count'] = len(peaks)
        features['Mean_Peak_Amplitude'] = np.mean(fft_amplitude[peaks])
        features['Peak_Amplitude_Std'] = np.std(fft_amplitude[peaks])

        # 前5个最大峰值
        top_peaks = peaks[np.argsort(fft_amplitude[peaks])[-5:]][::-1]
        for i, peak_idx in enumerate(top_peaks):
            features[f'Top_Peak_{i + 1}_Freq'] = freqs[peak_idx]
            features[f'Top_Peak_{i + 1}_Amp'] = fft_amplitude[peak_idx]
    else:
        features['Peak_Count'] = 0
        features['Mean_Peak_Amplitude'] = 0
        features['Peak_Amplitude_Std'] = 0

    # ==================== 6. PSD特征 ====================
    features['PSD_Mean'] = np.mean(psd)
    features['PSD_Std'] = np.std(psd)
    features['PSD_Peak'] = np.max(psd)
    features['PSD_Peak_Freq'] = freqs_psd[np.argmax(psd)]

    # ==================== 7. 包络谱特征 (对轴承故障诊断特别重要) ====================
    # 计算包络谱
    analytic_signal = signal.hilbert(signal_data)
    amplitude_envelope = np.abs(analytic_signal)
    envelope_spectrum = np.abs(np.fft.fft(amplitude_envelope))[:N // 2] * 2 / N

    features['Envelope_Spectral_Mean'] = np.mean(envelope_spectrum)
    features['Envelope_Spectral_RMS'] = np.sqrt(np.mean(envelope_spectrum ** 2))
    features['Envelope_Spectral_Kurtosis'] = kurtosis(envelope_spectrum)

    # 包络谱峰值
    env_peaks, _ = signal.find_peaks(envelope_spectrum, height=np.mean(envelope_spectrum) * 0.1)
    if len(env_peaks) > 0:
        features['Envelope_Peak_Count'] = len(env_peaks)
        features['Envelope_Peak_Freq'] = freqs[env_peaks[np.argmax(envelope_spectrum[env_peaks])]]
    else:
        features['Envelope_Peak_Count'] = 0
        features['Envelope_Peak_Freq'] = 0

    return features, freqs, fft_amplitude, freqs_psd, psd


def proceed_frequency_features_raw(file_path):
    # 加载数据
    mat_data = loadmat(file_path, squeeze_me=True)
    fs = 12000  # 采样频率
    for key in mat_data.keys():
        # 过滤掉以'__'开头和结尾的MATLAB自带元数据
        if key.find('_time') != -1:
            print(f"-{file_path}:__key: {key}: __{type(mat_data[key])}__, Shape: {np.shape(mat_data[key])}")
            signal_data = mat_data[key]  # 根据图片中的键名

            # 使用您之前滤波后的信号（图中的紫色曲线）
            # 如果没有滤波后的数据，可以使用原始信号
            filtered_signal = signal_data  # 如果没有滤波，使用原始信号

            # 提取频域特征
            freq_features, freqs, fft_amp, freqs_psd, psd = extract_frequency_domain_features(filtered_signal, fs)

            # 创建特征数据框
            feature_df = pd.DataFrame.from_dict(freq_features, orient='index').T
            print("频域特征提取结果:")
            print(feature_df.round(6))

            # 保存特征到CSV文件
            output_path = file_path.replace('.mat', f'{key}_frequency_features.csv')
            feature_df.to_csv(output_path)
            print(f"\n特征已保存至: {output_path}")

            # 可视化频谱分析结果
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('频域分析结果', fontsize=16)



if __name__ == '__main__':
    file_path = '源域数据集/12kHz_FE_data/B/0007/B007_1.mat'
    proceed_frequency_features_raw(file_path)
#
# # 1. 时域信号
# axes[0, 0].plot(np.arange(len(filtered_signal)) / fs, filtered_signal, 'purple')
# axes[0, 0].set_title('滤波后的时域信号')
# axes[0, 0].set_xlabel('时间 [s]')
# axes[0, 0].set_ylabel('幅度')
# axes[0, 0].grid(True)
#
# # 2. 幅度频谱
# axes[0, 1].plot(freqs, fft_amp, 'b-')
# axes[0, 1].set_title('幅度频谱')
# axes[0, 1].set_xlabel('频率 [Hz]')
# axes[0, 1].set_ylabel('幅度')
# axes[0, 1].grid(True)
# axes[0, 1].set_xlim(0, 2000)  # 限制频率范围以便观察
#
# # 3. 功率谱密度 (PSD)
# axes[1, 0].semilogy(freqs_psd, psd, 'r-')
# axes[1, 0].set_title('功率谱密度 (PSD)')
# axes[1, 0].set_xlabel('频率 [Hz]')
# axes[1, 0].set_ylabel('功率/频率 [dB/Hz]')
# axes[1, 0].grid(True)
# axes[1, 0].set_xlim(0, 2000)
#
# # 4. 频带能量分布
# band_powers = [freq_features.get(f'Band_{0}_{100}Hz_Power_Ratio', 0),
#                freq_features.get(f'Band_{100}_{500}Hz_Power_Ratio', 0),
#                freq_features.get(f'Band_{500}_{1000}Hz_Power_Ratio', 0),
#                freq_features.get(f'Band_{1000}_{2000}Hz_Power_Ratio', 0),
#                freq_features.get(f'Band_{2000}_{3000}Hz_Power_Ratio', 0),
#                freq_features.get(f'Band_{3000}_{6000}Hz_Power_Ratio', 0)]
#
# band_labels = ['0-100Hz', '100-500Hz', '500-1000Hz', '1000-2000Hz', '2000-3000Hz', '3000-6000Hz']
# axes[1, 1].pie([p for p in band_powers if p > 0],
#                labels=[band_labels[i] for i in range(len(band_powers)) if band_powers[i] > 0],
#                autopct='%1.1f%%')
# axes[1, 1].set_title('频带能量分布')
#
# plt.tight_layout()
# plt.savefig('frequency_analysis_results.png', dpi=300, bbox_inches='tight')
# plt.show()
