import os

import numpy as np
from scipy.io import loadmat
from savitzky import proceed
# 定义根目录（请替换为您电脑上的实际绝对路径，例如：C:\Users\YourName\源域数据集）
root_dir = "源域数据集"

# 创建一个空列表来存储所有文件的路径
all_file_paths = []

# 遍历根目录及其所有子文件夹
for foldername, subfolders, filenames in os.walk(root_dir):
    for filename in filenames:
        # 拼接完整路径，并添加到列表中
        full_path = os.path.join(foldername, filename)
        all_file_paths.append(full_path)

# 打印所有路径
for path in all_file_paths:
    print(path)

# 现在，all_file_paths 这个列表里就包含了所有文件的完整路径


for file_path in all_file_paths:
    # 1. 指定.mat文件的路径
    # file_path = '源域数据集/12kHz_DE_data/B/0007/B007_0.mat'  # 请将此处替换为您实际的文件路径

    # 2. 使用loadmat函数加载文件
    #    设置 squeeze_roots=True 可以自动解包单维度的数组，让数据更简洁
    mat_data = loadmat(file_path, squeeze_me=True)

    # 3. 探索加载的数据对象
    #    loadmat返回的是一个字典，键是MATLAB工作区中的变量名，值是对应的NumPy数组
    print("文件中包含的所有变量名：")
    for key in mat_data.keys():
        # 过滤掉以'__'开头和结尾的MATLAB自带元数据
        if key.find('_time') != -1:
            print(f"-{file_path}:__key: {key}: __{type(mat_data[key])}__, Shape: {np.shape(mat_data[key])}")
    #         获取到data
            de_signal = mat_data[key]  # 驱动端(Drive End)传感器数据
            proceed(de_signal[1300:2500],key,file_path)
    # 4. 提取您需要的特定变量（根据您的图片）
    #    提取时间序列数据
    # de_signal = mat_data['X118_DE_time']  # 驱动端(Drive End)传感器数据