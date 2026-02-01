import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def read_matrices_from_file(file_path):
    matrices = []
    current_matrix = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:  # 如果是空行，表示一个矩阵结束
                if current_matrix:
                    matrices.append(current_matrix)
                    current_matrix = []
            else:
                # 将当前行解析为浮点数列表
                row = list(map(float, line.split()))  # 修改为 float
                current_matrix.append(row)

        # 添加最后一个矩阵（如果文件末尾没有空行）
        if current_matrix:
            matrices.append(current_matrix)

    return matrices
def standardize_matrix_list(matrix_list):
    """标准化矩阵列表长度至88个元素"""
    num = len(matrix_list)
    if num < 88:
        # 创建19x19的全零矩阵
        zero_matrix = [[0.0 for _ in range(19)] for _ in range(19)]
        # 填充全零矩阵直到88个
        matrix_list.extend([zero_matrix] * (88 - num))
    elif num > 88:
        # 截断至前88个矩阵
        matrix_list = matrix_list[:88]
    return matrix_list

# 通道数据
channel_data = {
    "Fp1": [-0.0294367, 0.0839171, -0.006990],
    "Fp2": [0.0298723, 0.0848959, -0.007080],
    "F3": [-0.0502438, 0.0531112, 0.042192],
    "F4": [0.0518362, 0.0543048, 0.040814],
    "C3": [-0.0653581, -0.0116317, 0.064358],
    "C4": [0.0671179, -0.0109003, 0.06358],
    "P3": [-0.0530073, -0.0787878, 0.05594],
    "P4": [0.0556667, -0.0785602, 0.056561],
    "O1": [-0.0294134, -0.112449, 0.008839],
    "O2": [0.0298426, -0.112156, 0.0088],
    "F7": [-0.0702629, 0.0424743, -0.011420],
    "F8": [0.0730431, 0.0444217, -0.012000],
    "T3": [-0.0841611, -0.0160187, -0.009346],
    "T4": [0.0850799, -0.0150203, -0.009490],
    "T5": [-0.0724343, -0.0734527, -0.002487],
    "T6": [0.0730557, -0.0730683, -0.002540],
    "Fz": [0.0003122, 0.0585120, 0.066462],
    "Cz": [0.0004009, -0.0091670, 0.100244],
    "Pz": [0.0003247, -0.0811150, 0.082615]
}

file_path = 'H:/数据/Daima/pythonProject/.venv/avg_copula_matrix.txt'
file_path2 = 'H:/数据/Daima/pythonProject/保存数据/avg_granger_matrix.txt'
file_path3 = 'H:/数据/Daima/pythonProject/.venv/plv_matrix.txt'
file_path4 = 'H:/数据/Daima/pythonProject/.venv/pearson_matrix.txt'

matrices = read_matrices_from_file(file_path)
matrices2 = read_matrices_from_file(file_path2)
matrices3 = read_matrices_from_file(file_path3)
matrices4 = read_matrices_from_file(file_path4)

# 读取并标准化矩阵列表
matrices = standardize_matrix_list(matrices)
matrices2 = standardize_matrix_list(matrices2)
matrices3 = standardize_matrix_list(matrices3)
matrices4 = standardize_matrix_list(matrices4)
# 把matrices变成matrices4和matrices5两个相加混合之后平均
matrices5 = [0.5 * (np.array(m4) + np.array(m3)) for m4, m3 in zip(matrices4, matrices3)]

# 假设 matrices 是原始数据（确保其形状为 (88, 19, 19)）
 # 类型为 numpy.ndarray
copula = np.array(matrices)
gelanjie = np.array(matrices2)
plv = np.array(matrices3)
pearson = np.array(matrices4)
pearson_plv = np.array(matrices5)
# 将数据分为三部分
# 前36个
first_part = pearson[:36]
# 中间36个后29个
middle_part = pearson[36:65]
# 最后24个
last_part = pearson[65:]
# 前36个
first_part1 = copula[:36]
# 中间36个后29个
middle_part1 = copula[36:65]
# 最后24个
last_part1 = copula[65:]
# 前36个
first_part2 = gelanjie[:36]
# 中间36个后29个
middle_part2 = gelanjie[36:65]
# 最后24个
last_part2 = gelanjie[65:]
# 前36个
first_part3 = plv[:36]
# 中间36个后29个
middle_part3 = plv[36:65]
# 最后24个
last_part3 = plv[65:]
# 前36个
first_part4 = pearson_plv[:36]
# 中间36个后29个
middle_part4 = pearson_plv[36:65]
# 最后24个
last_part4 = pearson_plv[65:]

# 计算各部分的平均
avg_first = np.mean(first_part, axis=0)
avg_middle = np.mean(middle_part, axis=0)
avg_last = np.mean(last_part, axis=0)
avg_first1 = np.mean(first_part1, axis=0)
avg_middle1 = np.mean(middle_part1, axis=0)
avg_last1 = np.mean(last_part1, axis=0)
avg_first2 = np.mean(first_part2, axis=0)
avg_middle2 = np.mean(middle_part2, axis=0)
avg_last2 = np.mean(last_part2, axis=0)
avg_first3 = np.mean(first_part3, axis=0)
avg_middle3 = np.mean(middle_part3, axis=0)
avg_last3 = np.mean(last_part3, axis=0)
avg_first4 = np.mean(first_part4, axis=0)
avg_middle4 = np.mean(middle_part4, axis=0)
avg_last4 = np.mean(last_part4, axis=0)

# 将结果组合成一个形状为 (3, 19, 19) 的数组
avg_matrices_pearson_plv = np.stack([avg_first, avg_middle, avg_last])
avg_matrices_copula = np.stack([avg_first1, avg_middle1, avg_last1])
avg_matrices_granger = np.stack([avg_first2, avg_middle2, avg_last2])
avg_matrices_plv = np.stack([avg_first3, avg_middle3, avg_last3])
avg_matrices_pearson = np.stack([avg_first4, avg_middle4, avg_last4])

# 选择并归一化矩阵，去除极值影响
original_matrix = avg_matrices_pearson_plv[2]

# 去除极值影响的归一化：排除异常大或小的值
non_zero_mask = original_matrix != 0
if np.any(non_zero_mask):
    non_zero_values = original_matrix[non_zero_mask]
    # 计算百分位数来排除极值
    p5 = np.percentile(non_zero_values, 5)  # 5th percentile
    p95 = np.percentile(non_zero_values, 95)  # 95th percentile
    
    # 过滤掉极值
    filtered_values = non_zero_values[(non_zero_values >= p5) & (non_zero_values <= p95)]
    
    if len(filtered_values) > 0:
        min_val = np.min(filtered_values)
        max_val = np.max(filtered_values)
        
        # 避免除零错误
        if max_val != min_val:
            # 使用过滤后的范围进行归一化，但应用于所有非零值
            temp_normalized = np.zeros_like(original_matrix, dtype=float)
            temp_normalized[non_zero_mask] = (original_matrix[non_zero_mask] - min_val) / (max_val - min_val)
            # 将超出[0,1]范围的值限制在范围内
            norm_matrix = np.clip(temp_normalized, 0, 1)
        else:
            # 如果过滤后的值都相同，则设置为0.5（中间值）
            norm_matrix = np.zeros_like(original_matrix, dtype=float)
            norm_matrix[non_zero_mask] = 0.5
    else:
        # 如果过滤后没有值，则使用全部非零值
        min_val = np.min(non_zero_values)
        max_val = np.max(non_zero_values)
        
        if max_val != min_val:
            norm_matrix = np.zeros_like(original_matrix, dtype=float)
            norm_matrix[non_zero_mask] = (non_zero_values - min_val) / (max_val - min_val)
        else:
            norm_matrix = np.zeros_like(original_matrix, dtype=float)
            norm_matrix[non_zero_mask] = 0.5
else:
    # 如果矩阵全是0，则保持为0
    norm_matrix = original_matrix
"""
# 创建符合神经科学期刊标准的颜色映射
cmap = LinearSegmentedColormap.from_list('div_cmap',
                                       ['#2166AC', '#FFFFFF', '#B2182B'],
                                       N=256)

plt.figure(figsize=(14, 12))

# 绘制主热力图
heatmap = plt.imshow(norm_matrix,
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    aspect='equal',
                    interpolation='none')  # 禁用插值保持单元格边界清晰

# 添加单元格网格线
for i in range(20):  # 绘制20条分隔线（19单元格+1终点）
    # 水平线（行分隔）
    plt.hlines(i-0.5, -0.5, 18.5, colors='gray', linewidth=0.8, linestyles='-')
    # 垂直线（列分隔）
    plt.vlines(i-0.5, -0.5, 18.5, colors='gray', linewidth=0.8, linestyles='-')

# 设置坐标轴范围保证边框完整
plt.xlim(-0.5, 18.5)
plt.ylim(18.5, -0.5)  # 保持矩阵上三角方向

# 专业级颜色条配置
cbar = plt.colorbar(heatmap,
                   shrink=0.6,
                   ticks=np.linspace(0, 1, 5))
cbar.ax.set_ylabel('Normalized Connectivity',
                 rotation=270,
                 labelpad=25,
                 fontsize=12)

# 电极标签设置
electrodes = list(channel_data.keys())
plt.xticks(ticks=np.arange(19), labels=electrodes,
          rotation=60, ha='right', fontsize=10)
plt.yticks(ticks=np.arange(19), labels=electrodes,
          fontsize=10, va='center')

# 突出显示中央电极（Cz）
cz_index = electrodes.index('Cz')
plt.gca().add_patch(plt.Rectangle((cz_index-0.5, cz_index-0.5), 1, 1,
                                fill=False, edgecolor='gold', linewidth=2))

plt.title('Channel-wise Bordered Connectivity Matrix',
         fontsize=14, pad=20)

plt.tight_layout()
plt.show()
"""
# 可解释性连接图
# 提取坐标和标签
x = []
y = []
labels = []
for channel, coords in channel_data.items():
    x.append(coords[0])
    y.append(coords[1])
    labels.append(channel)

# 定义颜色映射
colors = [
    (0.0, 'none'),
    (0.5, 'blue'),
    (0.7, 'green'),
    (0.8, 'orange'),
    (0.9, 'red'),
    (1.0, 'red')
]
cmap = LinearSegmentedColormap.from_list('connection_strength', colors)

# 创建一个 4x3 的子图网格
fig, axs = plt.subplots(4, 3, figsize=(18, 20))
axs = axs.flatten()  # 将 2D 子图数组展平为 1D

# 绘制每个连接矩阵
matrix_sets = [avg_matrices_granger, avg_matrices_copula, avg_matrices_plv, avg_matrices_pearson]
print(np.array(matrix_sets).shape)
titles = ['Granger Causality', 'Copula Analysis', 'PLV', 'Pearson']
states = ['AD', 'Healthy', 'FTD']

for i, matrix_set in enumerate(matrix_sets):
    # 绘制每个子图
    for j in range(3):
        ax = axs[i * 3 + j]
        current_matrix = matrix_set[j]
        
        # 只绘制最强的前20%连接，避免过度拥挤
        non_zero_values = current_matrix[current_matrix != 0]
        if len(non_zero_values) > 0:
            # 计算阈值 - 只绘制强度最高的20%连接
            threshold = np.percentile(np.abs(non_zero_values), 80)  # 取绝对值的80%分位数
            
            # 统计实际绘制的连接数
            connections_drawn = 0
            
            # 只绘制上三角矩阵以避免重复连线
            for k in range(19):
                for l in range(k+1, 19):  # 只绘制上三角部分
                    connection_strength = current_matrix[k, l]
                    
                    # 只绘制强度高于阈值的连接
                    if abs(connection_strength) >= threshold:
                        # 根据值的大小设定颜色和线宽
                        if abs(connection_strength) < np.percentile(np.abs(non_zero_values), 85):
                            color = 'blue'
                            linewidth = 1.0
                        elif abs(connection_strength) < np.percentile(np.abs(non_zero_values), 95):
                            color = 'green'
                            linewidth = 1.5
                        else:
                            color = 'red'
                            linewidth = 2.0
                        
                        ax.plot([x[k], x[l]], [y[k], y[l]], color=color, linewidth=linewidth)
                        connections_drawn += 1

            # 打印每个子图的连接统计信息
            print(f"{titles[i]} - {states[j]}: 绘制了 {connections_drawn} 条连接 (阈值: {threshold:.4f})")
        else:
            # 如果没有非零值，什么都不画
            print(f"{titles[i]} - {states[j]}: 无非零连接")
            pass

        # 不添加电极标签
        pass

        ax.set_title(f"{titles[i]} - {states[j]}")
        ax.axis('off')  # 隐藏坐标轴

# 添加一个整体的颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])  # dummy array for the data
fig.subplots_adjust(right=0.9)  # 调整子图布局，为颜色条留出空间
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(sm, cax=cbar_ax, label='Connection Strength')

# 保存为PDF格式到指定路径
output_path = r'C:\Users\Administrator\Desktop\小论文图\脑电图功能连接分析.pdf'
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"已保存综合图表PDF到: {output_path}")

plt.show()

# 输出归一化后的数据，方便复制
print("=== 归一化后绘图数据输出 ===")
print("\n通道坐标数据：")
for channel, coords in channel_data.items():
    print(f"{channel}: {coords}")

print("\n平均矩阵形状：")
print(f"Granger Causality 矩阵形状: {avg_matrices_granger.shape}")
print(f"Copula Analysis 矩阵形状: {avg_matrices_copula.shape}")
print(f"PLV 矩阵形状: {avg_matrices_plv.shape}")
print(f"Pearson 矩阵形状: {avg_matrices_pearson.shape}")

# 创建输出文件
# output_file_path = r'H:\数据\Daima\pythonProject\保存数据\原始矩阵数据.txt'
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     f.write("=== 原始矩阵数据输出 ===\n\n")
#
#     # 直接输出原始矩阵数据（未归一化）并保存到文件
#     f.write("Granger Causality 原始矩阵数据：\n")
#     for i, matrix in enumerate(avg_matrices_granger):
#         f.write(f"\nGranger State {i} ({['AD', 'Healthy', 'FTD'][i]}):\n")
#         np.savetxt(f, matrix, fmt='%.6f', delimiter=' ')
#
#     f.write("\n\nCopula Analysis 原始矩阵数据：\n")
#     for i, matrix in enumerate(avg_matrices_copula):
#         f.write(f"\nCopula State {i} ({['AD', 'Healthy', 'FTD'][i]}):\n")
#         np.savetxt(f, matrix, fmt='%.6f', delimiter=' ')
#
#     f.write("\n\nPLV 原始矩阵数据：\n")
#     for i, matrix in enumerate(avg_matrices_plv):
#         f.write(f"\nPLV State {i} ({['AD', 'Healthy', 'FTD'][i]}):\n")
#         np.savetxt(f, matrix, fmt='%.6f', delimiter=' ')
#
#     f.write("\n\nPearson 原始矩阵数据：\n")
#     for i, matrix in enumerate(avg_matrices_pearson):
#         f.write(f"\nPearson State {i} ({['AD', 'Healthy', 'FTD'][i]}):\n")
#         np.savetxt(f, matrix, fmt='%.6f', delimiter=' ')
#
# print(f"\n数据已同步保存到: {output_file_path}")

# 显示每个矩阵的百分位数统计信息
def print_percentile_stats(name, matrix_set):
    print(f"\n{name} 百分位数统计:")
    for i, matrix in enumerate(matrix_set):
        non_zero_values = matrix[matrix != 0]
        if len(non_zero_values) > 0:
            p50 = np.percentile(non_zero_values, 50)
            p70 = np.percentile(non_zero_values, 70)
            p85 = np.percentile(non_zero_values, 85)
            p95 = np.percentile(non_zero_values, 95)
            print(f"  {name} State {i} ({['AD', 'Healthy', 'FTD'][i]}) - P50: {p50:.4f}, P70: {p70:.4f}, P85: {p85:.4f}, P95: {p95:.4f}")
        else:
            print(f"  {name} State {i} ({['AD', 'Healthy', 'FTD'][i]}) - 无非零值")

print_percentile_stats("Granger Causality", avg_matrices_granger)
print_percentile_stats("Copula Analysis", avg_matrices_copula)
print_percentile_stats("PLV", avg_matrices_plv)
print_percentile_stats("Pearson", avg_matrices_pearson)

# 直接输出原始矩阵数据（未归一化）
print("\nGranger Causality 原始矩阵数据：")
for i, matrix in enumerate(avg_matrices_granger):
    print(f"\nGranger State {i} ({['AD', 'Healthy', 'FTD'][i]}):\n", matrix)

print("\nCopula Analysis 原始矩阵数据：")
for i, matrix in enumerate(avg_matrices_copula):
    print(f"\nCopula State {i} ({['AD', 'Healthy', 'FTD'][i]}):\n", matrix)

print("\nPLV 原始矩阵数据：")
for i, matrix in enumerate(avg_matrices_plv):
    print(f"\nPLV State {i} ({['AD', 'Healthy', 'FTD'][i]}):\n", matrix)

print("\nPearson 原始矩阵数据：")
for i, matrix in enumerate(avg_matrices_pearson):
    print(f"\nPearson State {i} ({['AD', 'Healthy', 'FTD'][i]}):\n", matrix)


"""
# 定义颜色映射，仅涵盖 0.9 到 1.0 的范围
colors = [
    (0.0, 'gray'),  # 0.9
    (0.25, 'blue'),  # 0.925
    (0.5, 'green'),  # 0.95
    (0.75, 'orange'),  # 0.975
    (1.0, 'red')  # 1.0
]
cmap = LinearSegmentedColormap.from_list('connection_strength', colors)
# 绘制每个连接矩阵
for i in range(avg_matrices.shape[0]):
    plt.figure(figsize=(12, 10))
    plt.scatter(x, y)

    # 获取当前矩阵并归一化
    current_matrix = avg_matrices[i]
    min_val = np.min(current_matrix)
    max_val = np.max(current_matrix)
    normalized_matrix = (current_matrix - min_val) / (max_val - min_val)

    # 绘制连接线，根据归一化后的连接强度使用不同颜色和线宽
    for j in range(19):
        for k in range(19):
            if j != k:  # 不绘制自连接
                connection_strength = normalized_matrix[j, k]
                if connection_strength <= 0.9:
                    continue  # 跳过连接强度小于 0.9 的连接
                elif connection_strength <= 0.925:
                    color = 'gray'
                    linewidth = 0.5
                elif connection_strength <= 0.95:
                    color = 'blue'
                    linewidth = 1.0
                elif connection_strength <= 0.975:
                    color = 'green'
                    linewidth = 1.5
                elif connection_strength <= 0.99:
                    color = 'orange'
                    linewidth = 2.0
                else:
                    color = 'red'
                    linewidth = 2.5
                plt.plot([x[j], x[k]], [y[j], y[k]], color=color, linewidth=linewidth)

    # 添加标签
    for idx, label in enumerate(labels):
        plt.text(x[idx], y[idx], label)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(titles[i])  # 设置对应的标题

    # 添加颜色条图注
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.9, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.05)
    cbar.set_label('Connection Strength')
    cbar.set_ticks([0.9, 0.925, 0.95, 0.975, 0.99, 1.0])
    cbar.set_ticklabels(['0.9', '0.925', '0.95', '0.975', '0.99', '1.0'])
    # 隐藏X轴和Y轴
    plt.axis('off')
    plt.show()
"""