import os

import numpy as np
from matplotlib import rcParams
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import loadmat

rcParams["font.sans-serif"] = ["SimHei"]  # 指定中文字体
rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 评估不同参数组合的性能
def evaluate_sg_params(original, window_length, polyorder):
    """评估特定参数组合的Savitzky-Golay滤波器性能"""
    smoothed = signal.savgol_filter(original, window_length, polyorder)

    # 计算评估指标
    rmse = np.sqrt(np.mean((original - smoothed) ** 2))
    noise_original = original - np.mean(original)
    noise_filtered = smoothed - np.mean(smoothed)
    snr_improvement = 10 * np.log10(np.var(noise_original) / np.var(noise_filtered))
    correlation = np.corrcoef(original, smoothed)[0, 1]

    return rmse, snr_improvement, correlation


def proceed(de_signal, key,file_path):
    # 加载数据


    fs = 12000
    time_axis = np.arange(len(de_signal)) / fs

    # 定义多组参数进行对比
    parameter_sets = [
        {'window_length': 31, 'polyorder': 4, 'color': 'purple', 'label': 'WL=31, PO=4'}
    ]

    # 应用不同参数的滤波器
    plt.figure(figsize=(14, 10))

    # 绘制原始信号
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, de_signal, 'k-', alpha=0.3, label='原始信号', linewidth=1)

    # 应用并绘制不同参数的滤波结果
    for params in parameter_sets:
        smoothed = signal.savgol_filter(
            de_signal,
            window_length=params['window_length'],
            polyorder=params['polyorder']
        )
        plt.plot(time_axis, smoothed, color=params['color'],
                 label=params['label'], linewidth=1.5)

    plt.title(f'{file_path}_{key}')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    # # 修复保存路径问题
    idx = file_path.find('\\')
    output_dir = file_path[idx:]  # 获取目录路径
    output_dir = '源域数据集滤波图像' + output_dir
    output_dir = output_dir[:output_dir.find('.')]

    # 确保目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建目录

    plt.savefig(f"./{output_dir}/{key}.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("参数组合的性能评估:")
    print("WL=窗口长度, PO=多项式阶数")
    print("组合\t\tRMSE\t\tSNR改善(dB)\t相关系数")
    print("-" * 55)

    for params in parameter_sets:
        rmse, snr_imp, corr = evaluate_sg_params(
            de_signal,
            params['window_length'],
            params['polyorder']
        )
        print(f"WL={params['window_length']}, PO={params['polyorder']}\t{rmse:.6f}\t{snr_imp:.2f}\t\t{corr:.4f}")


if __name__ == '__main__':
    proceed(1, key='X118_DE_time')
