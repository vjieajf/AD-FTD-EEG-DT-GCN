import os
import mne
import numpy as np
from scipy.signal import hilbert
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import scipy.stats as stats
from scipy.signal import butter, filtfilt, ricker, resample, cwt
import networkx as nx  # 用于计算图论指标
from networkx.algorithms.shortest_paths.weighted  import all_pairs_dijkstra
import pywt
from scipy.signal import morlet2  # Morlet 小波核
from scipy.fft import fft

# 设置中文字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 设置主目录路径
main_dir = 'E:/静息态脑电数据/111/ds004504/derivatives/'
# CSV文件路径
csv_file_path = '../normalized_channel_distances.csv'
# 创建字典来存储每个被试的 raw_data
raw_data_dict = {}

def ricker_wavelet(t, sigma=1):
    # 创建一个连续小波对象
    wavelet = pywt.ContinuousWavelet('mexh')
    # 生成小波数据
    psi, x = wavelet.wavefun(level=10)  # level 越大，分辨率越高
    # 调整小波的长度
    if len(psi) != len(t):
        # 如果小波长度与输入时间序列长度不匹配，进行插值
        x_target = np.linspace(x.min(), x.max(), len(t))
        psi = np.interp(x_target, x, psi)
    # 调整小波的宽度（sigma）
    psi = psi / sigma
    return psi

def compute_clustering_and_efficiency(plv_matrix):
    # 1. 清除自环
    np.fill_diagonal(plv_matrix, 0)

    # 2. 转换为距离矩阵
    distance_matrix = 1 - plv_matrix

    # 3. 构建加权图
    G = nx.Graph()
    num_nodes = distance_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = distance_matrix[i, j]
            if not np.isnan(weight) and not np.isinf(weight):  # 检查是否为有效数值
                weight = float(weight)  # 确保权重是浮点数
                if weight > 0:
                    G.add_edge(i, j, weight=weight)

    # 4. 计算加权聚类系数
    try:
        clustering_coefficient = nx.average_clustering(G, weight='weight')
    except:
        # 如果加权计算失败，尝试无权重计算
        clustering_coefficient = nx.average_clustering(G)

    # 5. 计算所有节点对的最短路径
    # 使用all_pairs_dijkstra并将其结果存储为字典
    shortest_paths = {}
    for source, (distances, paths) in all_pairs_dijkstra(G):
        shortest_paths[source] = distances

    # 6. 计算全局效率
    efficiency = 0.0
    num_pairs = 0

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if j in shortest_paths[i]:
                efficiency += 1.0 / shortest_paths[i][j]
                num_pairs += 1

    if num_pairs == 0:
        efficiency = 0.0
    else:
        efficiency /= num_pairs

    return clustering_coefficient, efficiency


def extract_phases(data):
    # data: 形状 (通道数, 样本点数)
    analytic_signal = hilbert(data, axis=1)  # 对时间轴进行Hilbert变换
    phases = np.angle(analytic_signal)  # 提取瞬时相位
    return phases


def compute_partial_correlation(signals, reference_channel_idx=0):
    """
    采用Poli等人(2014)提出的修正版部分相关法，控制参考通道(A1/A2平均信号)的间接干扰
    参考文献: Aniele Poli et al., BMC Neuroscience 15(Suppl 1):P99 (2014)
    公式: r_ij^partial = (r_ij - r_ik * r_jk) / sqrt((1-r_ik²)(1-r_jk²))
    """
    # 中心化数据
    centered = signals - signals.mean(axis=1, keepdims=True)
    # 计算标准差
    stds = np.std(centered, axis=1, ddof=0)
    # 计算协方差矩阵
    cov_matrix = centered @ centered.T / (signals.shape[1] - 1)
    # 计算相关系数矩阵
    corr_matrix = cov_matrix / np.outer(stds, stds)

    n_channels = signals.shape[0]
    partial_corr_matrix = np.zeros((n_channels, n_channels))

    # 对每个通道对计算部分相关系数
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                partial_corr_matrix[i, j] = 1.0
            else:
                # 获取参考通道的相关系数
                r_ik = corr_matrix[i, reference_channel_idx]
                r_jk = corr_matrix[j, reference_channel_idx]
                r_ij = corr_matrix[i, j]

                # 应用Poli等人的修正版部分相关公式
                # 添加数值稳定性处理
                term1 = 1 - r_ik ** 2
                term2 = 1 - r_jk ** 2

                # 确保分母项非负且不为零
                if term1 <= 0:
                    term1 = 1e-8  # 使用极小正值避免除零
                if term2 <= 0:
                    term2 = 1e-8  # 使用极小正值避免除零

                denominator = np.sqrt(term1 * term2)

                if denominator > 1e-8:  # 避免除零
                    partial_corr_matrix[i, j] = (r_ij - r_ik * r_jk) / denominator
                else:
                    partial_corr_matrix[i, j] = r_ij

    return partial_corr_matrix


def compute_partial_plv(phases, reference_channel_idx=0):
    """
    采用Poli等人(2014)提出的修正版部分PLV法，控制参考通道(A1/A2平均信号)的间接干扰
    参考文献: Aniele Poli et al., BMC Neuroscience 15(Suppl 1):P99 (2014)
    公式: r_ij^partial = (r_ij - r_ik * r_jk) / sqrt((1-r_ik²)(1-r_jk²))
    """
    n_channels = phases.shape[0]
    plv_matrix = np.zeros((n_channels, n_channels))

    # 计算原始PLV矩阵
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phases[i, :] - phases[j, :]
            plv_matrix[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[j, i] = plv_matrix[i, j]

    # 应用部分相关修正
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                # 获取参考通道的PLV值
                plv_ik = plv_matrix[i, reference_channel_idx]
                plv_jk = plv_matrix[j, reference_channel_idx]
                plv_ij = plv_matrix[i, j]

                # 应用类似Poli等人的修正公式
                # 添加数值稳定性处理
                term1 = 1 - plv_ik ** 2
                term2 = 1 - plv_jk ** 2

                # 确保分母项非负且不为零
                if term1 <= 0:
                    term1 = 1e-8  # 使用极小正值避免除零
                if term2 <= 0:
                    term2 = 1e-8  # 使用极小正值避免除零

                denominator = np.sqrt(term1 * term2)

                if denominator > 1e-8:
                    plv_matrix[i, j] = (plv_ij - plv_ik * plv_jk) / denominator
                else:
                    plv_matrix[i, j] = plv_ij

    return plv_matrix

def calculate_swn(adjacency_matrix, num_iterations=100):
    if adjacency_matrix is None or not isinstance(adjacency_matrix, np.ndarray):
        return None

    threshold = 0.5  # 根据需要调整阈值
    adjacency_matrix_thresholded = (adjacency_matrix > threshold).astype(int)

    G = nx.from_numpy_array(adjacency_matrix_thresholded)

    try:
        avg_clustering = nx.average_clustering(G)
        avg_path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        avg_clustering = np.nan
        avg_path_length = np.nan

    n = G.number_of_nodes()
    m = G.number_of_edges()
    clustering_coeffs = []
    path_lengths = []
    for _ in range(num_iterations):
        random_G = nx.gnm_random_graph(n, m)
        try:
            clustering_coeffs.append(nx.average_clustering(random_G))
            path_lengths.append(nx.average_shortest_path_length(random_G))
        except nx.NetworkXError:
            clustering_coeffs.append(np.nan)
            path_lengths.append(np.nan)

    random_clustering = np.nanmean(clustering_coeffs)
    random_path_length = np.nanmean(path_lengths)

    if avg_clustering == 0 or avg_path_length == 0 or random_clustering == 0 or random_path_length == 0:
        return np.nan

    swn_index = (avg_clustering / random_clustering) / (avg_path_length / random_path_length)
    return swn_index

# 计算Graph Index Complexity (GIC)
def calculate_gic(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    num_triangles = np.trace(np.linalg.matrix_power(adjacency_matrix, 3)) // 6
    return num_triangles / n

# 遍历sub-001到sub-065目录
for sub_id in range(1, 89):
    sub_dir = f'sub-{sub_id:03d}'
    file_path = os.path.join(main_dir, sub_dir, 'eeg', f'{sub_dir}_task-eyesclosed_eeg.set')

    if os.path.exists(file_path):
        # 读取EEG数据
        raw_data = mne.io.read_raw_eeglab(file_path, preload=True)

        # 查看所有注释
        print(f"Annotations for {sub_dir}:")
        print(raw_data.annotations)

        # 查找并删除所有 'boundary' 事件
        boundary_events = raw_data.annotations.description == 'boundary'

        if any(boundary_events):
            # 通过遍历 annotations 来删除 boundary 事件
            indices_to_delete = [idx for idx, desc in enumerate(raw_data.annotations.description) if desc == 'boundary']

            # 删除找到的 'boundary' 事件
            raw_data.annotations.delete(indices_to_delete)
            print(f"Deleted boundary events for {sub_dir}")
        else:
            print(f"No boundary events found for {sub_dir}")

        # 输出一些基本信息
        print(f"Loaded {sub_dir}: {raw_data.info['nchan']} channels, {raw_data.n_times} samples")

        # 将 raw_data 存入字典
        raw_data_dict[sub_dir] = raw_data
    else:
        print(f"File for {sub_dir} not found.")
"""
# 读取CSV文件
csv_file_path = '../normalized_channel_distances.csv'
channel_distances_df = pd.read_csv(csv_file_path)

# 读取CSV文件，指定第一列为索引
channel_distances_df = pd.read_csv(csv_file_path, index_col=0)

# 将DataFrame转换为Numpy数组
channel_distances = channel_distances_df.to_numpy()
"""
# 初始化存储标准差和中位数的数组
std_array = np.zeros(89)  # 存储 66 个被试的标准差特征
median_array = np.zeros(89)  # 存储 66 个被试的中位数特征
clustering_array = np.zeros(89)  # 存储 66 个被试的标准差特征
efficiency_array = np.zeros(89)  # 存储 66 个被试的中位数特征
GIC_array = np.zeros(89)
swn_array = np.zeros(89)
for sub_idx, sub_id in enumerate(range(1, 89)):
    sub_dir = f'sub-{sub_id:03d}'  # 创建参与者目录的名称，例如 'sub-001'
    sub_data = raw_data_dict.get(sub_dir, None)  # 获取参与者的 EEG 数据
    if sub_data is None:
        print(f"No data found for {sub_dir}")
        continue

    data = sub_data.get_data()
    print(f"正在读取{sub_dir}: {data.shape}")

    # 1. 降采样到 250 Hz
    original_sampling_rate = 500  # 原始采样率
    target_sampling_rate = 250    # 目标采样率
    new_samples = int(data.shape[1] * target_sampling_rate / original_sampling_rate)
    data = resample(data, new_samples, axis=1)  # 沿时间维进行降采样

    # 采样频率假设为 250 Hz
    Fs = 500
    # 设计带通滤波器的参数
    low = 1
    high = 40
    nyq = 0.5 * Fs
    low_cut = low / nyq
    high_cut = high / nyq

    # 这里我们手动选择一个阶数，但通常你可能需要根据你的需求来调整它
    order = 4  # 你可以尝试不同的值来找到最佳的滤波器性能

    # 设计滤波器
    b, a = butter(order, [low_cut, high_cut], btype='band')

    # 初始化滤波后的 EEG 矩阵
    filteredEEG = np.zeros_like(data)

    # 对每个通道应用滤波器
    for i in range(data.shape[0]):
        filteredEEG[i, :] = filtfilt(b, a, data[i, :])

    # 每个周期的样本点数
    n_samples_per_epoch = int(4000 * Fs / 1000)  # 2000 ms的样本点数

    # 分割EEG信号
    n_epochs = (filteredEEG.shape[1] // n_samples_per_epoch)
    epochs = np.zeros((19, n_epochs, n_samples_per_epoch))

    for i in range(n_epochs):
        epochs[:, i, :] = filteredEEG[0:19, i * n_samples_per_epoch:(i + 1) * n_samples_per_epoch]
    print(epochs.shape)

    EEG_data = np.zeros((epochs.shape[0], epochs.shape[1], 2048))

    for i in range(epochs.shape[1]):
        EEG_data[:, i, 24:24 + 2000] = epochs[:, i, :]

    print(EEG_data.shape)

    # 获取基本参数
    n_channels = EEG_data.shape[0]  # 通道数
    n_epochs = EEG_data.shape[1]  # 有效分段数
    n_samples = EEG_data.shape[2]  # 每段样本数

    pearson_results = []
    for epoch_idx in range(n_epochs):
        # 提取当前分段的信号
        segment = EEG_data[:, epoch_idx, :]
        # 计算修正版部分相关系数矩阵（控制参考通道干扰）
        # 采用Poli等人(2014)方法，假设第0个通道为参考通道
        corr_matrix = compute_partial_correlation(segment, reference_channel_idx=0)
        pearson_results.append(corr_matrix)
        
    # 计算平均皮尔逊相关系数矩阵
    avg_pearson_matrix = np.mean(np.array(pearson_results), axis=0)

    """
    # 打开文件并追加写入，如果文件不存在会自动创建
    with open("H:/数据/Daima/pythonProject/.venv/avg_pearson_matrix4.txt", "a", encoding="utf-8") as file:
        # 保存当前数组到文件
        np.savetxt(file, avg_pearson_matrix, fmt="%.6f", delimiter="\t")
        # 添加一个空行以分隔不同的数组（可选）
        file.write("\n")

    """
    plv_results = []
    #遍历分段
    for epoch_idx in range(n_epochs):
        # 提取当前分段的信号
        segment = epochs[:, epoch_idx, :]
        # 计算瞬时相位
        phases = extract_phases(segment)
        # 计算修正版部分PLV矩阵（控制参考通道干扰）
        # 采用Poli等人(2014)方法，假设第0个通道为参考通道
        plv_matrix = compute_partial_plv(phases, reference_channel_idx=0)
        plv_results.append(plv_matrix)

    # 计算平均 PLV 矩阵
    avg_plv_matrix = np.mean(np.array(plv_results), axis=0)
    # 将对角线上的值设置为 1
    np.fill_diagonal(avg_plv_matrix, 1)
    
    # 计算平均皮尔逊相关系数矩阵
    # 将对角线上的值设置为 1
    np.fill_diagonal(avg_pearson_matrix, 1)
    
    # 分别保存两个矩阵到不同的文件
    # 保存PLV矩阵到文件
    with open("H:/数据/Daima/pythonProject/.venv/plv_matrix.txt", "a", encoding="utf-8") as file:
        # 保存当前数组到文件
        np.savetxt(file, avg_plv_matrix, fmt="%.6f", delimiter="\t")
        # 添加一个空行以分隔不同的数组（可选）
        file.write("\n")
    
    # 保存皮尔逊相关系数矩阵到文件
    with open("H:/数据/Daima/pythonProject/.venv/pearson_matrix.txt", "a", encoding="utf-8") as file:
        # 保存当前数组到文件
        np.savetxt(file, avg_pearson_matrix, fmt="%.6f", delimiter="\t")
        # 添加一个空行以分隔不同的数组（可选）

        file.write("\n")
