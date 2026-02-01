import os
import mne
import numpy as np
from scipy.signal import hilbert
import pandas as pd
import matplotlib
import scipy.stats as stats
from scipy.signal import butter, filtfilt, ricker, resample
import networkx as nx  # 用于计算图论指标
from networkx.algorithms.shortest_paths.weighted  import all_pairs_dijkstra
import pywt
from statsmodels.tsa.stattools  import grangercausalitytests
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

def compute_aic_for_granger(data, maxlag):
    """
    为格兰杰因果分析计算AIC值
    """
    try:
        # 使用格兰杰因果检验
        result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        # 从结果中提取F统计量和自由度信息来计算AIC
        # AIC = 2k - 2ln(L)，对于回归模型可以近似为 n*ln(RSS/n) + 2k
        rss = result[maxlag][0]['ssr_resid']  # 残差平方和
        n_obs = data.shape[0]  # 观测数量
        k_params = maxlag  # 参数数量（滞后阶数）
        
        # 避免RSS为0或负数的情况
        if rss <= 0:
            rss = 1e-10
        
        # 计算AIC值
        aic = n_obs * np.log(rss / n_obs) + 2 * k_params
        return aic
    except:
        return np.inf

def select_optimal_lag_aic(epoch_data, max_lag=4):
    """
    使用AIC准则选择最优滞后阶数 - 改进版本，更平衡性能和准确性
    """
    n_channels = epoch_data.shape[0]
    optimal_lags = np.full((n_channels, n_channels), 2, dtype=int)  # 默认为2
    
    # 为了平衡性能和准确性，我们只计算部分有代表性的通道对
    # 选择通道的子集，而不是仅仅前几个
    selected_channels = min(n_channels, 8)  # 选择前8个通道进行计算，增加代表性
    
    # 计算选定通道之间的最优滞后阶数
    lag_candidates = []
    
    for target in range(selected_channels):
        for source in range(selected_channels):
            if target == source:
                continue
            
            try:
                # 准备两变量数据
                data = np.column_stack([epoch_data[source], epoch_data[target]])  # source -> target
                
                aic_values = []
                for lag in range(1, max_lag + 1):
                    try:
                        aic = compute_aic_for_granger(data, lag)
                        aic_values.append(aic)
                    except:
                        aic_values.append(np.inf)
                
                if aic_values and not all(np.isinf(v) for v in aic_values):
                    # 找到AIC值最小的滞后阶数
                    optimal_lag = np.argmin(aic_values) + 1
                    lag_candidates.append(optimal_lag)
                else:
                    lag_candidates.append(2)  # 默认值
                    
            except Exception as e:
                lag_candidates.append(2)  # 默认值
    
    # 如果有有效的滞后阶数候选值，使用众数或中位数作为全局最优滞后阶数
    if lag_candidates:
        # 使用众数或中位数作为代表值
        from scipy import stats
        try:
            # 计算众数，如果存在多个众数则取第一个
            mode_result = stats.mode(lag_candidates, keepdims=True)
            global_optimal_lag = mode_result.mode[0]
        except:
            # 如果众数计算失败，使用中位数
            global_optimal_lag = int(np.median(lag_candidates))
        
        # 将全局最优滞后阶数应用到所有通道对
        for target in range(n_channels):
            for source in range(n_channels):
                if target != source:
                    optimal_lags[target, source] = global_optimal_lag
    else:
        # 如果没有有效的滞后阶数，保持默认值
        pass
    
    return optimal_lags

def compute_granger_fvalue_matrix_with_optimal_lag(epoch_data, optimal_lags=None):
    """使用最优滞后阶数计算格兰杰F值矩阵（含异常处理）"""
    n_channels = epoch_data.shape[0]
    fvalue_mat = np.full((n_channels, n_channels), np.nan)  # 初始化为NaN

    for target in range(n_channels):
        for source in range(n_channels):
            if target == source: continue
            try:
                # 如果提供了optimal_lags，则使用对应的最佳滞后阶数
                if optimal_lags is not None:
                    maxlag = optimal_lags[target, source]
                else:
                    maxlag = 2  # 默认值
                data = np.column_stack([epoch_data[target], epoch_data[source]])
                test_res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                fvalue_mat[source, target] = test_res[maxlag][0]['ssr_ftest'][0]
            except:
                pass  # 保持错误静默
    return fvalue_mat

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
# 初始化存储标准差和中位数的数组
std_array = np.zeros(89)  # 存储 66 个被试的标准差特征
median_array = np.zeros(89)  # 存储 66 个被试的中位数特征
clustering_array = np.zeros(89)  # 存储 66 个被试的标准差特征
efficiency_array = np.zeros(89)  # 存储 66 个被试的中位数特征
GIC_array = np.zeros(89)
swn_array = np.zeros(89)
granger_results = []
all_subject_granger_matrices = []  # 存储所有被试的平均格兰杰因果矩阵

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
    n_samples_per_epoch = int(5000 * Fs / 1000)  # 2000 ms的样本点数

    # 分割EEG信号
    n_epochs = (filteredEEG.shape[1] // n_samples_per_epoch)
    epochs = np.zeros((19, n_epochs, n_samples_per_epoch))

    for i in range(n_epochs):
        epochs[:, i, :] = filteredEEG[0:19, i * n_samples_per_epoch:(i + 1) * n_samples_per_epoch]
    print(epochs.shape)

    # 使用AIC准则选择最优滞后阶数
    # 计算第一个epoch的最优滞后阶数作为代表
    sample_epoch_data = epochs[:, 0, :]  # 使用第一个epoch的数据来选择最优滞后
    print("开始计算AIC准则选择最优滞后阶数...")
    optimal_lags = select_optimal_lag_aic(sample_epoch_data, max_lag=4)
    print(f"被试 {sub_id} 的最优滞后阶数矩阵形状: {optimal_lags.shape}")
    print(f"被试 {sub_id} 的全局最优滞后阶数: {optimal_lags[0, 1] if optimal_lags.size > 0 else 'N/A'}")  # 显示一个示例值

    # 计算使用最优滞后阶数的格兰杰因果矩阵
    granger_matrices_with_aic = []
    for epoch_idx in range(n_epochs):
        # 获取当前epoch的脑电数据（形状：[通道数, 样本数]）
        epoch_data = epochs[:, epoch_idx, :]
        # 使用AIC选择的最优滞后阶数计算格兰杰因果矩阵
        gc_matrix = compute_granger_fvalue_matrix_with_optimal_lag(epoch_data, optimal_lags)
        granger_matrices_with_aic.append(gc_matrix)

    # 计算使用AIC选择滞后阶数的平均格兰杰因果矩阵
    avg_granger_matrix_aic = np.nan_to_num(np.mean(granger_matrices_with_aic, axis=0))
    
    # 将当前被试的平均格兰杰因果矩阵添加到总列表中
    all_subject_granger_matrices.append(avg_granger_matrix_aic)
    
    # 使用AIC选择的最优滞后阶数计算的结果进行后续分析
    avg_granger_matrix = avg_granger_matrix_aic

    # 忽略 NaN 计算标准差和中位数
    std_features = np.nanstd(avg_granger_matrix)
    median_features = np.nanmedian(avg_granger_matrix)

    # 计算聚类系数和全局效率
    clustering_coefficient, efficiency = compute_clustering_and_efficiency(avg_granger_matrix)
    gic = calculate_gic(avg_granger_matrix)
    swn = calculate_swn(avg_granger_matrix)

    # 保存到数组
    std_array[sub_idx] = std_features
    median_array[sub_idx] = median_features
    clustering_array[sub_idx] = clustering_coefficient
    efficiency_array[sub_idx] = efficiency
    GIC_array[sub_idx] = gic
    swn_array[sub_idx] = swn

# 将所有被试的格兰杰因果矩阵保存到同一个文件中
filename = "H:/数据/Daima\pythonProject/保存数据/avg_granger_matrix.txt"
with open(filename, "w", encoding="utf-8") as file:
    for i, matrix in enumerate(all_subject_granger_matrices):
        # 保存每个被试的数组到文件
        np.savetxt(file, matrix, fmt="%.6f", delimiter="\t")
        # 添加一个空行以分隔不同的数组
        file.write("\n")

# 保存最终结果到文件
np.savetxt('H:/数据/Daima/pythonProject/.venv/保存数据/granger_std_features.txt', std_array, fmt='%.6f')
np.savetxt('H:/数据/Daima/pythonProject/.venv/保存数据/granger_median_features.txt', median_array, fmt='%.6f')
np.savetxt('H:/数据/Daima/pythonProject/.venv/保存数据/granger_clustering_coeff.txt', clustering_array, fmt='%.6f')
np.savetxt('H:/数据/Daima/pythonProject/.venv/保存数据/granger_efficiency.txt', efficiency_array, fmt='%.6f')
np.savetxt('H:/数据/Daima/pythonProject/.venv/保存数据/granger_gic.txt', GIC_array, fmt='%.6f')
np.savetxt('H:/数据/Daima/pythonProject/.venv/保存数据/granger_swn.txt', swn_array, fmt='%.6f')

# 添加等待输入以防止脚本立即退出
input('按回车键退出...')
