import os
import mne
import numpy as np
from scipy.signal import hilbert
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.signal import butter, filtfilt, ricker, resample
import networkx as nx  # 用于计算图论指标
from networkx.algorithms.shortest_paths.weighted  import all_pairs_dijkstra
import pywt
from statsmodels.tsa.stattools  import grangercausalitytests
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.impute  import SimpleImputer
from scipy.optimize  import minimize, Bounds
from scipy.stats  import rankdata, norm, t, stats
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft

# 初始化标准化器
scaler = StandardScaler()

# 设置中文字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 设置主目录路径
main_dir = 'E:/静息态脑电数据/111/ds004504/derivatives/'
# CSV文件路径
csv_file_path = '../normalized_channel_distances.csv'
# 创建字典来存储每个被试的 raw_data
raw_data_dict = {}

# 定义小波函数
def ricker_wavelet(t, sigma=1):
    return ricker(len(t), sigma)

def compute_clustering_and_efficiency(plv_matrix):
    # 1. 归一化处理
    # 检查是否存在不合理值（小于0或大于1）
    if np.min(plv_matrix) < 0 or np.max(plv_matrix) > 1:
        # 计算归一化后的矩阵
        min_val = np.min(plv_matrix)
        max_val = np.max(plv_matrix)
        plv_matrix = (plv_matrix - min_val) / (max_val - min_val)

    # 2. 处理 NaN 值
    # 检查是否存在 NaN 值
    if np.isnan(plv_matrix).any():
        # 使用均值插补填充 NaN
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        plv_matrix = imputer.fit_transform(plv_matrix)

        # 3. 清除自环
    np.fill_diagonal(plv_matrix, 0)

    # 4. 转换为矩阵距离
    distance_matrix = 1 - plv_matrix

    # 5. 构建加权图
    G = nx.Graph()
    num_nodes = distance_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = distance_matrix[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    # 6. 计算加权聚类系数
    clustering_coefficient = nx.average_clustering(G, weight='weight')

    # 7. 计算所有节点对的最短路径
    shortest_paths = dict(nx.all_pairs_dijkstra(G))

    # 8. 计算全局效率
    efficiency = 0.0
    num_pairs = 0

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if j in shortest_paths[i][0]:  # shortest_paths[i][0] 是距离字典
                distance = shortest_paths[i][0][j]
                if np.isfinite(distance):
                    efficiency += 1.0 / distance
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
"""
def compute_pearson_correlation(signals):
    # 皮尔逊相关系数计算
    # 中心化数据
    centered = signals - signals.mean(axis=1,  keepdims=True)
    # 计算标准差
    stds = np.std(centered,  axis=1, ddof=0)
    # 计算协方差矩阵
    cov_matrix = centered @ centered.T / (signals.shape[1]  - 1)
    # 计算相关系数矩阵
    corr_matrix = cov_matrix / np.outer(stds,  stds)
    np.fill_diagonal(corr_matrix,  1.0)  # 确保对角线精确为1
    return corr_matrix
"""

def gaussian_copula(u, v, theta):
    """高斯Copula函数"""
    return norm.cdf((norm.ppf(u) * theta + norm.ppf(v)) / np.sqrt(1 + theta ** 2))

def student_t_copula(u, v, theta, df):
    """学生T Copula函数"""
    return t.cdf((t.ppf(u, df) * theta + t.ppf(v, df)) / np.sqrt(1 + theta ** 2), df)

def gumbel_copula(u, v, theta):
    """Gumbel Copula函数"""
    return np.exp(-(((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta)))

def frank_copula(u, v, theta, epsilon=1e-8):
    """Frank Copula函数（带数值稳定性增强）"""
    theta = np.clip(theta, -100, 100)  # 防止theta过大导致数值溢出
    denominator = np.exp(-theta) - 1 + epsilon  # 添加极小值避免除零
    numerator = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
    return -np.log(np.clip(1 + numerator / denominator, epsilon, None)) / theta

def clayton_copula(u, v, theta):
    """Clayton Copula函数"""
    return (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta)

def copula_log_likelihood(theta, u, v, copula_func, df=None, epsilon=1e-8):
    """通用对数似然函数"""
    if copula_func == student_t_copula:
        if df is None:
            raise ValueError("df must be provided for student_t_copula")
        copula_vals = copula_func(u, v, theta, df)
    else:
        copula_vals = copula_func(u, v, theta)
    return -np.sum(np.log(np.clip(copula_vals, epsilon, None) + epsilon))  # 防止log(0)


def copula_log_likelihood(theta, u, v, copula_func, df=None):
    """Copula 对数似然函数"""
    if df is not None:
        copula_val = copula_func(u, v, theta, df)
    else:
        copula_val = copula_func(u, v, theta)
    return -np.nansum(np.log(copula_val))


def compute_copula_matrix(epoch_data, copula_func, initial_theta=5.0, bounds=None, df=None):
    """通用 Copula 矩阵计算"""
    # 数据标准化
    scaler = StandardScaler()

    # 提取实部和虚部
    real_part = np.real(epoch_data)
    imag_part = np.imag(epoch_data)

    # 合并实部和虚部为实数矩阵
    combined_data = np.concatenate((real_part, imag_part), axis=0)

    # 标准化处理
    standardized_data = scaler.fit_transform(combined_data.T).T

    # 分离实部和虚部
    standardized_real = standardized_data[:real_part.shape[0], :]
    standardized_imag = standardized_data[real_part.shape[0]:, :]

    # 重新组合为复数矩阵
    standardized_complex = standardized_real + 1j * standardized_imag

    n_channels = standardized_complex.shape[0]
    copula_mat = np.full((n_channels, n_channels), np.nan)

    for target in range(n_channels):
        for source in range(n_channels):
            if target == source:
                continue
            try:
                # 经验 CDF 计算（避免 0 和 1 边界）
                u = (rankdata(standardized_complex[target].real) + 0.5) / (len(standardized_complex[target].real) + 1)
                v = (rankdata(standardized_complex[source].real) + 0.5) / (len(standardized_complex[source].real) + 1)

                # 参数优化（带边界约束）
                if bounds is not None:
                    result = minimize(
                        copula_log_likelihood,
                        x0=initial_theta,
                        bounds=bounds,
                        args=(u, v, copula_func, df),
                        method='L-BFGS-B'
                    )
                else:
                    result = minimize(
                        copula_log_likelihood,
                        x0=initial_theta,
                        args=(u, v, copula_func, df),
                        method='L-BFGS-B'
                    )
                optimal_theta = result.x[0]

                # 计算 Copula 值
                if copula_func == student_t_copula:
                    if df is None:
                        raise ValueError("df must be provided for student_t_copula")
                    copula_vals = copula_func(u, v, optimal_theta, df)
                else:
                    copula_vals = copula_func(u, v, optimal_theta)
                copula_mat[source, target] = np.nanmean(copula_vals)
            except Exception as e:
                print(f"Error at ({source}, {target}): {str(e)}")
                continue

    # 对结果进行后处理，例如应用阈值
    copula_mat = np.clip(copula_mat, 0, 0.5)  # 将相似度限制在 0 到 0.5 之间

    return copula_mat
def calculate_swn(adjacency_matrix, num_iterations=100):
    # 检查输入是否有效
    if adjacency_matrix is None or not isinstance(adjacency_matrix, np.ndarray):
        return None

    # 归一化处理
    if np.min(adjacency_matrix) < 0 or np.max(adjacency_matrix) > 1:
        min_val = np.min(adjacency_matrix)
        max_val = np.max(adjacency_matrix)
        if max_val == min_val:
            return None  # 避免除以零
        adjacency_matrix = (adjacency_matrix - min_val) / (max_val - min_val)

    # 调整阈值以确保图连通
    threshold = 0.5  # 初始阈值
    while True:
        adjacency_matrix_thresholded = (adjacency_matrix > threshold).astype(int)
        G = nx.from_numpy_array(adjacency_matrix_thresholded)
        if nx.is_connected(G) or threshold <= 0.1:  # 阈值下限
            break
        threshold -= 0.1  # 降低阈值

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
        # 生成随机连通图
        random_G = nx.gnm_random_graph(n, m)
        if not nx.is_connected(random_G):
            continue  # 跳过不连通的图
        try:
            clustering_coeffs.append(nx.average_clustering(random_G))
            path_lengths.append(nx.average_shortest_path_length(random_G))
        except nx.NetworkXError:
            continue  # 跳过计算失败的情况

    # 检查是否有有效的随机图数据
    if not clustering_coeffs or not path_lengths:
        return np.nan

    random_clustering = np.nanmean(clustering_coeffs)
    random_path_length = np.nanmean(path_lengths)

    # 避免除以零
    if avg_clustering == 0 or avg_path_length == 0 or random_clustering == 0 or random_path_length == 0:
        return np.nan

    swn_index = (avg_clustering / random_clustering) / (avg_path_length / random_path_length)
    return swn_index
def calculate_gic(adjacency_matrix):
    # 1. 清除自环
    np.fill_diagonal(adjacency_matrix, 0)

    # 2. 转换为二进制邻接矩阵
    # 根据数据分布选择合适的阈值
    threshold = np.percentile(adjacency_matrix, 70)  # 选择70百分位作为阈值
    binary_adjacency = (adjacency_matrix > threshold).astype(int)

    # 3. 创建无向图
    G = nx.Graph(binary_adjacency)

    # 4. 计算所有三角形数量
    num_triangles = sum(nx.triangles(G).values()) // 3

    # 5. 计算 GIC
    n = G.number_of_nodes()
    if n == 0:
        return 0
    gic = num_triangles / n

    return gic

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
copula_results = []
# 遍历 sub-001 到 sub-065，提取并打印每个参与者的数据形状
for sub_idx, sub_id in enumerate(range(1, 89)):
    sub_dir = f'sub-{sub_id:03d}'  # 创建参与者目录的名称，例如 'sub-001'
    sub_data = raw_data_dict.get(sub_dir, None)  # 获取参与者的 EEG 数据
    data = sub_data.get_data()
    print(data.shape)

    # 1. 降采样到250 Hz
    original_sampling_rate = 500  # 原始采样率
    target_sampling_rate = 250  # 目标采样率
    new_samples = int(data.shape[1] * target_sampling_rate / original_sampling_rate)
    data = resample(data, new_samples, axis=1)  # 沿时间维进行降采样

    # 采样频率假设为250 Hz
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

    # 初始化滤波后的EEG矩阵
    filteredEEG = np.zeros_like(data)

    # 对每个通道应用滤波器
    for i in range(data.shape[0]):
        filteredEEG[i, :] = filtfilt(b, a, data[i, :])

    print(filteredEEG.shape)

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

    # 遍历所有epoch
    for epoch_idx in range(n_epochs):
        epoch_data = EEG_data[:, epoch_idx, :]
        gc_matrix = compute_copula_matrix(epoch_data, gumbel_copula, initial_theta=1.0, bounds=[(1, 10)])
        copula_results.append(gc_matrix)
        
    # 计算平均并替换 NaN 为 0
    avg_copula_matrix = np.nan_to_num(np.mean(np.abs(np.array(copula_results)), axis=0))
    """
    # 打开文件并追加写入，如果文件不存在会自动创建
    with open("H:/数据/Daima/pythonProject/.venv/avg_copula_matrix_FFT.txt", "a", encoding="utf-8") as file:
        # 保存当前数组到文件
        np.savetxt(file, avg_copula_matrix, fmt="%.6f", delimiter="\t")
        # 添加一个空行以分隔不同的数组（可选）
        file.write("\n")
    """
    # 忽略 NaN 计算标准差和中位数
    std_features = np.nanstd(avg_copula_matrix)
    median_features = np.nanmedian(avg_copula_matrix)

    # 计算聚类系数和全局效率
    clustering_coefficient, efficiency = compute_clustering_and_efficiency(avg_copula_matrix)
    gic = calculate_gic(avg_copula_matrix)
    swn = calculate_swn(avg_copula_matrix)

    # 保存到数组
    std_array[sub_idx] = std_features
    median_array[sub_idx] = median_features
    clustering_array[sub_idx] = clustering_coefficient
    efficiency_array[sub_idx] = efficiency
    GIC_array[sub_idx] = gic
    swn_array[sub_idx] = swn

# 保存 std_array 到 std_array.txt
with open('../copula std_array.txt', 'w') as std_file:
    for value in std_array:
        std_file.write(f"{value}\n")

print("std_array 已保存到 copula std_array.txt 文件中")

# 保存 median_array 到 median_array.txt
with open('../copula median_array.txt', 'w') as median_file:
    for value in median_array:
        median_file.write(f"{value}\n")

print("median_array 已保存到 copula median_array.txt 文件中")

# 保存聚类系数数组到 clustering_array.txt
with open('../copula clustering_array.txt', 'w') as clustering_file:
    for value in clustering_array:
        clustering_file.write(f"{value}\n")

print("clustering_array 已保存到 copula clustering_array.txt 文件中")

# 保存全局效率数组到 efficiency_array.txt
with open('../copula efficiency_array.txt', 'w') as efficiency_file:
    for value in efficiency_array:
        efficiency_file.write(f"{value}\n")

print("efficiency_array 已保存到 copula efficiency_array.txt 文件中")

# 保存全局效率数组到 efficiency_array.txt
with open('../copula GIC_array.txt', 'w') as GIC_file:
    for value in GIC_array:
        GIC_file.write(f"{value}\n")

print("GIC_array已保存到 copula GIC_array.txt 文件中")

# 保存全局效率数组到 efficiency_array.txt
with open('../copula swn_array.txt', 'w') as swn_file:
    for value in swn_array:
        swn_file.write(f"{value}\n")

print("swn_array已保存到 copula swn_array.txt 文件中")

"""
# 计算高斯Copula矩阵
gaussian_copula_mat = compute_copula_matrix(epoch_data, gaussian_copula, initial_theta=0.5, bounds=[(-1, 1)])

# 计算学生T Copula矩阵
student_t_copula_mat = compute_copula_matrix(epoch_data, student_t_copula, initial_theta=0.5, bounds=[(-1, 1)], df=5)

# 计算Gumbel Copula矩阵
gumbel_copula_mat = compute_copula_matrix(epoch_data, gumbel_copula, initial_theta=1.0, bounds=[(1, 10)])

# 计算Frank Copula矩阵
frank_copula_mat = compute_copula_matrix(epoch_data, frank_copula, initial_theta=5.0, bounds=[(1e-5, 100)])

# 计算Clayton Copula矩阵
clayton_copula_mat = compute_copula_matrix(epoch_data, clayton_copula, initial_theta=0.5, bounds=[(0, 10)])

"""
