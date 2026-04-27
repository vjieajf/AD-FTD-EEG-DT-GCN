import os
import mne
import numpy as np
from scipy.signal import hilbert
import pandas as pd
import matplotlib
import scipy.stats as stats
from scipy.signal import butter, filtfilt, resample
import networkx as nx  # 用于计算图论指标
from networkx.algorithms.shortest_paths.weighted  import all_pairs_dijkstra
import pywt
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.impute  import SimpleImputer
import warnings
# 忽略FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

# 设置中文字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 设置主目录路径
main_dir = 'E:/静息态脑电数据/111/ds004504/derivatives/'
# 五个标准频段（用于多频段格兰杰矩阵输出）
frequency_bands = [
    ("delta", 1, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("gamma", 30, 45),
]
# CSV文件路径
csv_file_path = '../normalized_channel_distances.csv'
# 创建字典来存储每个被试的 raw_data
raw_data_dict = {}

# 定义小波函数
def ricker_wavelet(t, sigma=1):
    points = len(t)
    a = float(sigma)
    x = np.arange(points) - (points - 1.0) / 2.0
    xsq = (x / a) ** 2
    return (2.0 / (np.sqrt(3 * a) * np.pi ** 0.25)) * (1 - xsq) * np.exp(-xsq / 2.0)

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
    shortest_paths = dict(all_pairs_dijkstra(G))

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

def _build_lag_features(signal, maxlag):
    """构造滞后特征，返回形状 (T-maxlag, maxlag)。"""
    n_samples = signal.shape[0]
    if n_samples <= maxlag:
        return None
    rows = n_samples - maxlag
    lagged = np.empty((rows, maxlag), dtype=float)
    for lag in range(1, maxlag + 1):
        lagged[:, lag - 1] = signal[maxlag - lag:n_samples - lag]
    return lagged


def _compute_granger_fc_strength(target_signal, source_signal, maxlag=2):
    """计算 source->target 的 Granger FC 强度（非检验统计量）。"""
    y = np.asarray(target_signal, dtype=float)
    x = np.asarray(source_signal, dtype=float)

    y_lags = _build_lag_features(y, maxlag)
    x_lags = _build_lag_features(x, maxlag)
    if y_lags is None or x_lags is None:
        return np.nan

    y_t = y[maxlag:]
    n_obs = y_t.shape[0]
    if n_obs <= (2 * maxlag + 1):
        return np.nan

    ones = np.ones((n_obs, 1), dtype=float)
    x_restricted = np.hstack((ones, y_lags))
    x_unrestricted = np.hstack((ones, y_lags, x_lags))

    try:
        beta_r, _, _, _ = np.linalg.lstsq(x_restricted, y_t, rcond=None)
        beta_u, _, _, _ = np.linalg.lstsq(x_unrestricted, y_t, rcond=None)
        resid_r = y_t - x_restricted @ beta_r
        resid_u = y_t - x_unrestricted @ beta_u
        rss_r = np.sum(resid_r ** 2)
        rss_u = np.sum(resid_u ** 2)

        if rss_u <= 0 or rss_r <= 0:
            return np.nan

        # 使用对数残差比作为方向性连接强度，去除 F/AIC 检验语义
        fc_strength = np.log(rss_r / rss_u)
        return fc_strength if np.isfinite(fc_strength) else np.nan
    except Exception:
        return np.nan


def compute_granger_fc_matrix(epoch_data, maxlag=2):
    """固定滞后阶数计算格兰杰FC矩阵（仅连接强度，不做统计检验）。"""
    n_channels = epoch_data.shape[0]
    fc_mat = np.full((n_channels, n_channels), np.nan)

    for target in range(n_channels):
        target_signal = epoch_data[target]
        for source in range(n_channels):
            if target == source:
                continue
            fc_mat[source, target] = _compute_granger_fc_strength(
                target_signal=target_signal,
                source_signal=epoch_data[source],
                maxlag=maxlag,
            )
    return fc_mat

def calculate_swn(adjacency_matrix, num_iterations=100):
    if adjacency_matrix is None or not isinstance(adjacency_matrix, np.ndarray):
        return None
        # 归一化处理
        # 检查是否存在不合理值（小于0或大于1）
    if np.min(adjacency_matrix) < 0 or np.max(adjacency_matrix) > 1:
        # 计算归一化后的矩阵
        min_val = np.min(adjacency_matrix)
        max_val = np.max(adjacency_matrix)
        adjacency_matrix = (adjacency_matrix - min_val) / (max_val - min_val)

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


def calculate_gic(adjacency_matrix):
    # 1. 清除自环
    np.fill_diagonal(adjacency_matrix, 0)

    # 2. 转换为二进制邻接矩阵
    threshold = 0.5  # 根据数据分布选择合适的阈值
    binary_adjacency = (adjacency_matrix > threshold).astype(int)

    # 3. 创建无向图
    G = nx.Graph(binary_adjacency)

    # 4. 计算所有三角形数量
    num_triangles = sum(nx.triangles(G).values()) // 3

    # 5. 计算 GIC
    n = G.number_of_nodes()
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
all_subject_granger_matrices = []  # 存储所有被试的平均格兰杰因果矩阵
# 新增：固定形状保存多频段矩阵，目标形状 (88, 19, 19*5)
all_subject_granger_multiband = np.zeros((88, 19, 19 * len(frequency_bands)), dtype=float)

# 遍历 sub-001 到 sub-065，提取并打印每个参与者的数据形状
for sub_idx, sub_id in enumerate(range(1, 89)):
    sub_dir = f'sub-{sub_id:03d}'  # 创建参与者目录的名称，例如 'sub-001'
    sub_data = raw_data_dict.get(sub_dir, None)  # 获取参与者的 EEG 数据
    
    if sub_data is None:
        print(f"警告: {sub_dir} 数据未找到，跳过")
        continue
        
    data = sub_data.get_data()
    # 打印现在的sub_idx
    print(f"目前是 sub-{sub_id:03d} ({sub_idx + 1}/{len(range(1, 89))})")
    print(data.shape)
    
    # 显示进度
    print(f"处理进度: {sub_idx + 1}/88 ({((sub_idx + 1) / 88 * 100):.1f}%)")

    # 1. 降采样到250 Hz
    original_sampling_rate = 500  # 原始采样率
    target_sampling_rate = 250  # 目标采样率
    new_samples = int(data.shape[1] * target_sampling_rate / original_sampling_rate)
    data = resample(data, new_samples, axis=1)  # 沿时间维进行降采样

    # 采样频率（降采样后）
    Fs = target_sampling_rate
    # 每个周期的样本点数（保留原有5秒分段策略）
    n_samples_per_epoch = int(5000 * Fs / 1000)
    order = 4

    # 存储当前被试五个频段的 19x19 平均格兰杰FC矩阵
    subject_band_matrices = []

    for band_name, low, high in frequency_bands:
        nyq = 0.5 * Fs
        low_cut = low / nyq
        high_cut = high / nyq
        b, a = butter(order, [low_cut, high_cut], btype='band')

        # 对当前频段进行滤波
        filteredEEG = np.zeros_like(data)
        for i in range(data.shape[0]):
            filteredEEG[i, :] = filtfilt(b, a, data[i, :])

        # 分割EEG信号
        n_epochs = filteredEEG.shape[1] // n_samples_per_epoch
        if n_epochs == 0:
            print(f"警告: {sub_dir} 在频段 {band_name} 分段后无有效epoch，填充零矩阵")
            subject_band_matrices.append(np.zeros((19, 19)))
            continue

        epochs = np.zeros((19, n_epochs, n_samples_per_epoch))
        for i in range(n_epochs):
            epochs[:, i, :] = filteredEEG[0:19, i * n_samples_per_epoch:(i + 1) * n_samples_per_epoch]

        # 固定滞后阶数=2，逐 epoch 计算格兰杰FC并在线平均为单个 19x19 矩阵
        running_sum = np.zeros((19, 19), dtype=float)
        valid_count = np.zeros((19, 19), dtype=float)
        for epoch_idx in range(n_epochs):
            epoch_data = epochs[:, epoch_idx, :]
            gc_matrix = compute_granger_fc_matrix(epoch_data, maxlag=2)
            finite_mask = np.isfinite(gc_matrix)
            running_sum[finite_mask] += gc_matrix[finite_mask]
            valid_count[finite_mask] += 1.0

        avg_granger_matrix = np.divide(
            running_sum,
            valid_count,
            out=np.zeros_like(running_sum),
            where=valid_count > 0,
        )
        subject_band_matrices.append(avg_granger_matrix)

    # 五个频段按最后一维拼接：19x19x5 -> 19x95
    subject_multiband_3d = np.stack(subject_band_matrices, axis=-1)
    subject_multiband_2d = subject_multiband_3d.reshape(19, 19 * len(frequency_bands))

    # 保存到固定容器，确保最终形状严格为 (88, 19, 95)
    all_subject_granger_multiband[sub_idx] = subject_multiband_2d
    all_subject_granger_matrices.append(subject_multiband_2d)

# 将所有被试的多频段格兰杰矩阵保存到同一个文件中（每个被试19x95）
filename = "H:/数据/Daima/pythonProject/保存数据/avg_granger_matrix_multiband.txt"
with open(filename, "w", encoding="utf-8") as file:
    for i, matrix in enumerate(all_subject_granger_matrices):
        # 保存每个被试的数组到文件
        np.savetxt(file, matrix, fmt="%.6f", delimiter="\t")
        # 添加一个空行以分隔不同的数组
        file.write("\n")

# 直接保存结构化npy，形状为(88, 19, 95)
npy_filename = "H:/数据/Daima/pythonProject/保存数据/avg_granger_matrix_multiband.npy"
np.save(npy_filename, all_subject_granger_multiband)
print(f"多频段矩阵npy已保存: {npy_filename}, shape={all_subject_granger_multiband.shape}")

# 添加等待输入以防止脚本立即退出
input('按回车键退出...')
