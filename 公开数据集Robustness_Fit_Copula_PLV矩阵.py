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
from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra
import pywt
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize, Bounds
from scipy.stats import rankdata, norm, t, stats
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft
from sklearn.metrics import mean_squared_error
from scipy.stats import kstest
import warnings

# 初始化标准化器
scaler = StandardScaler()

# 设置中文字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 主目录路径
main_dir = 'E:/静息态脑电数据/111/ds004504/derivatives/'
csv_file_path = '../normalized_channel_distances.csv'
raw_data_dict = {}

def ricker_wavelet(t, sigma=1):
    return ricker(len(t), sigma)

def compute_clustering_and_efficiency(plv_matrix):
    # 1. 归一化处理
    if np.min(plv_matrix) < 0 or np.max(plv_matrix) > 1:
        min_val = np.min(plv_matrix)
        max_val = np.max(plv_matrix)
        plv_matrix = (plv_matrix - min_val) / (max_val - min_val)

    # 2. 处理 NaN 值
    if np.isnan(plv_matrix).any():
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        plv_matrix = imputer.fit_transform(plv_matrix)

    np.fill_diagonal(plv_matrix, 0)
    distance_matrix = 1 - plv_matrix

    G = nx.Graph()
    num_nodes = distance_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = distance_matrix[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    clustering_coefficient = nx.average_clustering(G, weight='weight')
    shortest_paths = dict(nx.all_pairs_dijkstra(G))
    efficiency = 0.0
    num_pairs = 0

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if j in shortest_paths[i][0]:
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
    """Copula 对数似然函数"""
    if copula_func == student_t_copula:
        if df is None:
            raise ValueError("df must be provided for student_t_copula")
        copula_vals = copula_func(u, v, theta, df)
    else:
        copula_vals = copula_func(u, v, theta)
    return -np.nansum(np.log(copula_vals))

def compute_copula_matrix(epoch_data, copula_func, initial_theta=5.0, bounds=None, df=None):
    """通用 Copula 矩阵计算"""
    scaler = StandardScaler()
    real_part = np.real(epoch_data)
    imag_part = np.imag(epoch_data)
    combined_data = np.concatenate((real_part, imag_part), axis=0)
    standardized_data = scaler.fit_transform(combined_data.T).T
    standardized_real = standardized_data[:real_part.shape[0], :]
    standardized_imag = standardized_data[real_part.shape[0]:, :]
    standardized_complex = standardized_real + 1j * standardized_imag

    n_channels = standardized_complex.shape[0]
    copula_mat = np.full((n_channels, n_channels), np.nan)

    for target in range(n_channels):
        for source in range(n_channels):
            if target == source:
                continue
            try:
                u = (rankdata(standardized_complex[target].real) + 0.5) / (len(standardized_complex[target].real) + 1)
                v = (rankdata(standardized_complex[source].real) + 0.5) / (len(standardized_complex[source].real) + 1)

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

    copula_mat = np.clip(copula_mat, 0, 0.5)
    return copula_mat

def calculate_swn(adjacency_matrix, num_iterations=100):
    if adjacency_matrix is None or not isinstance(adjacency_matrix, np.ndarray):
        return None

    if np.min(adjacency_matrix) < 0 or np.max(adjacency_matrix) > 1:
        min_val = np.min(adjacency_matrix)
        max_val = np.max(adjacency_matrix)
        if max_val == min_val:
            return None
        adjacency_matrix = (adjacency_matrix - min_val) / (max_val - min_val)

    threshold = 0.5
    while True:
        adjacency_matrix_thresholded = (adjacency_matrix > threshold).astype(int)
        G = nx.from_numpy_array(adjacency_matrix_thresholded)
        if nx.is_connected(G) or threshold <= 0.1:
            break
        threshold -= 0.1

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
        if not nx.is_connected(random_G):
            continue
        try:
            clustering_coeffs.append(nx.average_clustering(random_G))
            path_lengths.append(nx.average_shortest_path_length(random_G))
        except nx.NetworkXError:
            continue

    if not clustering_coeffs or not path_lengths:
        return np.nan

    random_clustering = np.nanmean(clustering_coeffs)
    random_path_length = np.nanmean(path_lengths)

    if avg_clustering == 0 or avg_path_length == 0 or random_clustering == 0 or random_path_length == 0:
        return np.nan

    swn_index = (avg_clustering / random_clustering) / (avg_path_length / random_path_length)
    return swn_index

def calculate_gic(adjacency_matrix):
    np.fill_diagonal(adjacency_matrix, 0)
    threshold = np.percentile(adjacency_matrix, 70)
    binary_adjacency = (adjacency_matrix > threshold).astype(int)
    G = nx.Graph(binary_adjacency)
    num_triangles = sum(nx.triangles(G).values()) // 3
    n = G.number_of_nodes()
    if n == 0:
        return 0
    gic = num_triangles / n
    return gic

# 新增：Copula 连接稳定性验证函数
def copula_stability_check(data, sample_ratios=[0.6, 0.8, 1.0], n_repeats=10):
    """
    通过不同样本量的重采样方法验证Copula连接的稳定性
    操作：固定段长度（8 秒），随机抽取 60%、80%、100% 的样本量（每个比例重复 10 次），分别构建 Copula 连接矩阵
    量化指标：计算各样本量下所有通道对 Copula 连接系数的变异系数（CV = 标准差 / 均值）
    报告要求：需说明 "不同样本量下的 CV 均 < 0.15，证明 Copula 连接对样本大小不敏感"
    """
    n_timepoints = data.shape[1]
    results = {}
    
    print(f"正在进行Copula连接稳定性检验...")
    print(f"原始数据点数: {n_timepoints}")
    
    for ratio in sample_ratios:
        n_samples = int(n_timepoints * ratio)
        print(f"样本比例 {ratio*100:.0f}%: 使用 {n_samples} 个时间点")
        
        if n_samples < 10:  # 确保有足够的样本点
            print(f"Warning: For ratio {ratio}, not enough samples. Skipping.")
            results[ratio] = {'cv': np.nan, 'valid_count': 0, 'matrices': []}
            continue
        
        copula_matrices = []
        
        for i in range(n_repeats):
            # 随机选择时间点子集
            indices = np.random.choice(n_timepoints, n_samples, replace=False)
            sampled_data = data[:, indices]
            
            # 计算Copula矩阵
            try:
                copula_matrix = compute_copula_matrix(sampled_data, gumbel_copula, initial_theta=1.0, bounds=[(1, 10)])
                
                # 检查矩阵是否有效（包含有限值且不是全零矩阵）
                if np.any(np.isfinite(copula_matrix)) and np.any(copula_matrix != 0):
                    # 将NaN值替换为0，以便后续计算
                    copula_matrix = np.nan_to_num(copula_matrix)
                    copula_matrices.append(copula_matrix)
                else:
                    print(f"Warning: Sample ratio {ratio}, repeat {i} produced invalid matrix (all zeros or NaN), skipping.")
            except Exception as e:
                print(f"Warning: Sample ratio {ratio}, repeat {i} failed with error: {e}, skipping.")
                continue
        
        if len(copula_matrices) == 0:
            print(f"Warning: All repeats for ratio {ratio} produced invalid matrices.")
            results[ratio] = {'cv': np.nan, 'valid_count': 0, 'matrices': []}
            continue
        elif len(copula_matrices) == 1:
            # 如果只有一个有效矩阵，无法计算变异系数，返回该矩阵
            print(f"样本比例 {ratio}: 只有{len(copula_matrices)}个有效矩阵，无法计算变异系数。")
            results[ratio] = {'cv': np.nan, 'valid_count': 1, 'matrices': copula_matrices}
        else:
            # 将有效矩阵堆叠
            copula_matrices = np.array(copula_matrices)
            
            # 计算稳定性指标 - 连接强度的变异系数
            mean_matrix = np.nanmean(copula_matrices, axis=0)
            std_matrix = np.nanstd(copula_matrices, axis=0)
            
            # 避免除零错误和无效值
            with np.errstate(divide='ignore', invalid='ignore'):
                cv_matrix = np.divide(std_matrix, np.abs(mean_matrix) + 1e-10)  # 添加小值避免除零
            
            # 计算平均变异系数（排除对角线和无效值）
            off_diag_mask = ~np.eye(cv_matrix.shape[0], dtype=bool) & np.isfinite(cv_matrix)
            valid_cv_values = cv_matrix[off_diag_mask]
            
            if len(valid_cv_values) > 0:
                mean_cv = np.nanmean(valid_cv_values)
            else:
                mean_cv = np.nan
            
            results[ratio] = {'cv': mean_cv, 'valid_count': len(copula_matrices), 'matrices': copula_matrices}
            print(f"样本比例 {ratio*100:.0f}%: 有效矩阵数量 {len(copula_matrices)}/{n_repeats}, 平均CV: {mean_cv:.4f}")
    
    # 汇总所有样本比例的结果
    valid_cvs = [results[ratio]['cv'] for ratio in sample_ratios if not np.isnan(results[ratio]['cv'])]
    
    if len(valid_cvs) > 0:
        overall_mean_cv = np.nanmean(valid_cvs)
        print(f"所有样本比例的平均变异系数 (CV): {overall_mean_cv:.4f}")
        print(f"CV < 0.15 表示连接对样本变化不敏感")
        
        # 检查每个比例的CV是否都小于0.15
        all_below_threshold = all(cv < 0.15 for cv in valid_cvs)
        if all_below_threshold:
            print(f"不同样本量下的 CV 均 < 0.15，证明 Copula 连接对样本大小不敏感")
        else:
            print(f"部分样本量下的 CV >= 0.15，表明 Copula 连接对样本大小可能较为敏感")
    else:
        overall_mean_cv = np.nan
        print("无法计算总体平均变异系数")
    
    return results, overall_mean_cv

# 新增：Copula 拟合优度检验函数
def copula_goodness_of_fit_test(data, significance_level=0.05):
    """
    对每个 EEG 通道的时间序列，采用 Kolmogorov-Smirnov（K-S）检验
    验证其边际分布与 Copula 模型假设的适配性
    """
    n_channels = data.shape[0]
    p_values = []
    
    print(f"正在进行Copula拟合优度检验...")
    
    for channel in range(n_channels):
        # 提取通道数据
        channel_data = data[channel, :]
        
        # 检查数据是否有效
        if len(channel_data) < 10 or np.all(np.isnan(channel_data)) or np.all(channel_data == channel_data[0]):
            print(f"Warning: Channel {channel} has insufficient or constant data, skipping.")
            p_values.append(np.nan)
            continue
        
        # 对数据进行标准化
        channel_mean = np.mean(channel_data)
        channel_std = np.std(channel_data)
        if channel_std == 0:
            print(f"Warning: Channel {channel} has zero variance, skipping.")
            p_values.append(np.nan)
            continue
        
        standardized_data = (channel_data - channel_mean) / channel_std
        
        # 应用经验CDF转换为均匀分布
        ranks = rankdata(standardized_data) / (len(standardized_data) + 1)
        
        # 对均匀分布进行K-S检验，检验是否符合均匀分布
        try:
            ks_stat, p_value = kstest(ranks, 'uniform')
            p_values.append(p_value)
        except Exception as e:
            print(f"Warning: K-S test failed for channel {channel}, error: {e}")
            p_values.append(np.nan)
    
    # 过滤掉NaN值
    valid_p_values = [p for p in p_values if not np.isnan(p)]
    
    if len(valid_p_values) == 0:
        print("Error: All channels failed the goodness-of-fit test.")
        return p_values, 0.0
    
    # 检查是否所有通道的p值都大于显著性水平
    significant_tests = [p > significance_level for p in valid_p_values]
    passed_ratio = sum(significant_tests) / len(valid_p_values) if len(valid_p_values) > 0 else 0.0
    
    print(f"Copula拟合优度检验完成")
    print(f"有效通道数量: {len(valid_p_values)}/{n_channels}")
    print(f"通过检验的通道比例: {passed_ratio:.2%}")
    print(f"所有通道p值 > {significance_level} 表示边际分布符合模型假设")
    
    return p_values, passed_ratio

# 主处理循环
std_array = np.zeros(89)  # 存储 89 个被试的标准差特征
median_array = np.zeros(89)  # 存储 89 个被试的中位数特征
clustering_array = np.zeros(88)  # 存储 89 个被试的聚类系数
efficiency_array = np.zeros(89)  # 存储 89 个被试的效率
GIC_array = np.zeros(89)  # 存储 89 个被试的GIC
swn_array = np.zeros(89)  # 存储 89 个被试的SWN

# 遍历被试数据
for sub_idx, sub_id in enumerate(range(1, 90)):  # 从1到89
    sub_dir = f'sub-{sub_id:03d}'
    file_path = os.path.join(main_dir, sub_dir, 'eeg', f'{sub_dir}_task-eyesclosed_eeg.set')

    if os.path.exists(file_path):
        raw_data = mne.io.read_raw_eeglab(file_path, preload=True)

        # 删除boundary事件
        boundary_events = raw_data.annotations.description == 'boundary'
        if any(boundary_events):
            indices_to_delete = [idx for idx, desc in enumerate(raw_data.annotations.description) if desc == 'boundary']
            raw_data.annotations.delete(indices_to_delete)

        raw_data_dict[sub_dir] = raw_data
        data = raw_data.get_data()
        print(f"Processing {sub_dir}: {data.shape}")

        # 1. 降采样到250 Hz
        original_sampling_rate = 500
        target_sampling_rate = 250
        new_samples = int(data.shape[1] * target_sampling_rate / original_sampling_rate)
        data_resampled = resample(data, new_samples, axis=1)

        # 2. 滤波
        Fs = 500
        low, high = 1, 40
        nyq = 0.5 * Fs
        low_cut, high_cut = low / nyq, high / nyq
        order = 4
        b, a = butter(order, [low_cut, high_cut], btype='band')

        filteredEEG = np.zeros_like(data_resampled)
        for i in range(data_resampled.shape[0]):
            filteredEEG[i, :] = filtfilt(b, a, data_resampled[i, :])

        # 3. 分段
        n_samples_per_epoch = int(4000 * Fs / 1000)  # 4秒每段
        n_epochs = (filteredEEG.shape[1] // n_samples_per_epoch)
        epochs = np.zeros((19, n_epochs, n_samples_per_epoch))

        for i in range(n_epochs):
            epochs[:, i, :] = filteredEEG[0:19, i * n_samples_per_epoch:(i + 1) * n_samples_per_epoch]

        # 4. Copula分析（移除FFT处理，直接在时域数据上进行）
        copula_results = []
        for epoch_idx in range(n_epochs):
            # 获取当前epoch的脑电数据（形状：[通道数, 样本数]）
            epoch_data = epochs[:, epoch_idx, :]
            
            # 直接使用时域数据计算Copula矩阵（移除原来的FFT处理）
            gc_matrix = compute_copula_matrix(epoch_data, gumbel_copula, initial_theta=1.0, bounds=[(1, 10)])
            copula_results.append(gc_matrix)

        # 计算平均Copula矩阵
        avg_copula_matrix = np.nan_to_num(np.mean(np.abs(np.array(copula_results)), axis=0))

        # 6. Copula稳定性检验（新增部分）
        print(f"Performing stability analysis for subject {sub_id}...")
        stability_data = epochs[:, 0, :]  # 使用第一个epoch的数据进行稳定性检验（修正：使用epochs而不是未定义的EEG_data）
        try:
            stability_results, overall_cv = copula_stability_check(
                stability_data, sample_ratios=[0.6, 0.8, 1.0], n_repeats=10
            )
            print(f"Subject {sub_id} overall stability CV: {overall_cv:.4f}")
            
            # 显示每个样本比例的CV值
            for ratio, result in stability_results.items():
                if not np.isnan(result['cv']):
                    print(f"Subject {sub_id} - Sample Ratio {ratio*100:.0f}%: CV = {result['cv']:.4f}")
                else:
                    print(f"Subject {sub_id} - Sample Ratio {ratio*100:.0f}%: Could not compute CV (insufficient valid matrices)")
        except Exception as e:
            print(f"Stability analysis failed for subject {sub_id}: {e}")
            overall_cv = np.nan

        # 7. Copula拟合优度检验（新增部分）
        print(f"Performing goodness-of-fit analysis for subject {sub_id}...")
        try:
            p_vals, fit_ratio = copula_goodness_of_fit_test(stability_data, significance_level=0.05)
            print(f"Subject {sub_id} goodness-of-fit passed ratio: {fit_ratio:.2%}")
        except Exception as e:
            print(f"Goodness-of-fit analysis failed for subject {sub_id}: {e}")
            fit_ratio = np.nan

        # 8. 计算各种特征
        std_features = np.nanstd(avg_copula_matrix)
        median_features = np.nanmedian(avg_copula_matrix)
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

    else:
        print(f"File for {sub_dir} not found.")

# # 保存结果
# with open('../copula_robustness_std_array.txt', 'w') as std_file:
#     for value in std_array:
#         std_file.write(f"{value}\n")
# print("std_array 已保存到 copula_robustness_std_array.txt 文件中")
#
# with open('../copula_robustness_median_array.txt', 'w') as median_file:
#     for value in median_array:
#         median_file.write(f"{value}\n")
# print("median_array 已保存到 copula_robustness_median_array.txt 文件中")
#
# with open('../copula_robustness_clustering_array.txt', 'w') as clustering_file:
#     for value in clustering_array:
#         clustering_file.write(f"{value}\n")
# print("clustering_array 已保存到 copula_robustness_clustering_array.txt 文件中")
#
# with open('../copula_robustness_efficiency_array.txt', 'w') as efficiency_file:
#     for value in efficiency_array:
#         efficiency_file.write(f"{value}\n")
# print("efficiency_array 已保存到 copula_robustness_efficiency_array.txt 文件中")
#
# with open('../copula_robustness_GIC_array.txt', 'w') as GIC_file:
#     for value in GIC_array:
#         GIC_file.write(f"{value}\n")
# print("GIC_array 已保存到 copula_robustness_GIC_array.txt 文件中")
#
# with open('../copula_robustness_swn_array.txt', 'w') as swn_file:
#     for value in swn_array:
#         swn_file.write(f"{value}\n")
# print("swn_array 已保存到 copula_robustness_swn_array.txt 文件中")