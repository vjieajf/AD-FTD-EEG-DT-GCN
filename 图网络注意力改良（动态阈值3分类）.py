import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, classification_report

# 尝试导入图神经网络相关库，如果失败则给出提示
try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, GATConv
    from torch_geometric.nn import MessagePassing

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("警告：未安装 PyTorch Geometric 库，将无法运行图神经网络相关代码")
    print("请使用以下命令安装：pip install torch-geometric")
    TORCH_GEOMETRIC_AVAILABLE = False

import re
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os


def read_matrices_from_file(file_path):
    matrices = []
    with open(file_path, 'r', encoding='utf-8') as file:
        matrix = []
        for line in file:
            line = line.strip()
            if not line:  # 跳过空行
                if matrix:
                    matrices.append(matrix)
                    matrix = []
                continue
            # 使用正则表达式匹配复数格式
            complex_pattern = re.compile(r'\(([\d.-]+)\+([\d.-]+)j\)')
            row = []
            for item in line.split():
                match = complex_pattern.match(item)
                if match:
                    real = float(match.group(1))
                    imag = float(match.group(2))
                    row.append(complex(real, imag))
                else:
                    try:
                        row.append(float(item))
                    except ValueError:
                        # 如果不是复数也不是浮点数，保持原样
                        row.append(item)
            matrix.append(row)
        if matrix:
            matrices.append(matrix)
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


def prepare_data(matrices, threshold_percent):
    dataset = []
    for adj_matrix in matrices:
        # 步骤1：归一化（以Min-Max为例）
        adj_min = adj_matrix.min()
        adj_max = adj_matrix.max()
        adj_normalized = (adj_matrix - adj_min) / (adj_max - adj_min + 1e-8)  # 防止除零

        # 步骤2：应用阈值（动态阈值，保留50%的值）
        threshold = np.percentile(adj_normalized, threshold_percent)  # 计算第50百分位数作为阈值
        binary_adj = (adj_normalized > threshold).astype(float)
        edge_index = torch.tensor(np.stack(np.where(binary_adj > 0)), dtype=torch.long)
        degrees = torch.tensor(binary_adj.sum(axis=1), dtype=torch.float).unsqueeze(1)
        x = torch.cat([torch.eye(19), degrees], dim=1)  # 原始特征+节点度

        dataset.append(Data(x=x, edge_index=edge_index))
    return dataset


def dynamic_threshold_preprocessing(original_matrices, threshold_percent):
    """为每个 epoch 应用动态阈值处理原始矩阵"""
    processed_data = []
    for adj_matrix in original_matrices:
        # 步骤1：归一化（Min-Max）
        adj_min = adj_matrix.min()
        adj_max = adj_matrix.max()
        adj_normalized = (adj_matrix - adj_min) / (adj_max - adj_min + 1e-8)  # 防止除零

        # 步骤2：应用阈值（动态）
        threshold = np.percentile(adj_normalized, threshold_percent)
        binary_adj = (adj_normalized > threshold).astype(float)

        # 创建图数据结构
        edge_index = torch.tensor(np.stack(np.where(binary_adj > 0)), dtype=torch.long)
        degrees = torch.tensor(binary_adj.sum(axis=1), dtype=torch.float).unsqueeze(1)
        x = torch.cat([torch.eye(19), degrees], dim=1)  # 原始特征+节点度

        processed_data.append(Data(x=x, edge_index=edge_index))
    return processed_data


def dynamic_threshold_preprocessing_with_intra_normalization(original_matrices, threshold_percent):
    """为每个 epoch 应用动态阈值处理原始矩阵，使用内部标准化（每个矩阵单独标准化）    """
    processed_data = []
    for adj_matrix in original_matrices:
        # 步骤1：内部（Intra）标准化 - 每个矩阵单独标准化
        adj_min = adj_matrix.min()
        adj_max = adj_matrix.max()
        adj_normalized = (adj_matrix - adj_min) / (adj_max - adj_min + 1e-8)  # 防止除零

        # 步骤2：应用阈值（动态）
        threshold = np.percentile(adj_normalized, threshold_percent)
        binary_adj = (adj_normalized > threshold).astype(float)

        # 创建图数据结构
        edge_index = torch.tensor(np.stack(np.where(binary_adj > 0)), dtype=torch.long)
        degrees = torch.tensor(binary_adj.sum(axis=1), dtype=torch.float).unsqueeze(1)
        x = torch.cat([torch.eye(19), degrees], dim=1)  # 原始特征+节点度

        processed_data.append(Data(x=x, edge_index=edge_index))
    return processed_data


def fixed_threshold_preprocessing_with_global_normalization(original_matrices, threshold_value, global_min=None, global_max=None):
    """固定阈值处理，使用全局标准化（基于整个训练集的全局最小最大值）    """
    # 如果没有提供全局最小最大值，计算它们
    if global_min is None or global_max is None:
        all_values = np.concatenate([matrix.flatten() for matrix in original_matrices])
        global_min = all_values.min()
        global_max = all_values.max()
    
    processed_data = []
    for adj_matrix in original_matrices:
        # 步骤1：全局标准化 - 使用整个训练集的全局最小最大值
        adj_normalized = (adj_matrix - global_min) / (global_max - global_min + 1e-8)  # 防止除零

        # 步骤2：应用固定阈值
        binary_adj = (adj_normalized > threshold_value).astype(float)

        # 创建图数据结构
        edge_index = torch.tensor(np.stack(np.where(binary_adj > 0)), dtype=torch.long)
        degrees = torch.tensor(binary_adj.sum(axis=1), dtype=torch.float).unsqueeze(1)
        x = torch.cat([torch.eye(19), degrees], dim=1)  # 原始特征+节点度

        processed_data.append(Data(x=x, edge_index=edge_index))
    return processed_data, global_min, global_max


class GNNClassifier(nn.Module):
    def __init__(self, num_classes=3):  # 修改为三分类
        super().__init__()
        # 保持GCN层名称不变
        self.conv1 = GCNConv(20, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = GCNConv(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = GCNConv(32, 16)
        self.bn3 = nn.BatchNorm1d(16)

        # 替换为标准的GAT层
        self.gat = GATConv(
            in_channels=16,  # 输入特征维度
            out_channels=16,  # 输出特征维度（保持与之前相同）
            heads=8,  # 多头注意力机制（标准GAT配置）
            concat=False,  # 不拼接多头输出（保持维度不变）
            dropout=0.6,  # 注意力dropout
            add_self_loops=True  # 添加自环
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(16, num_classes)  # 修改为二分类输出

    def get_node_embeddings(self, data):
        """获取节点嵌入用于计算对比损失"""
        x, edge_index = data.x, data.edge_index

        # GCN层保持不变
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        # 添加GAT层（标准GAT实现）
        x = self.gat(x, edge_index)

        return x

    def forward(self, data):
        # 获取节点嵌入（包含GAT层）
        x = self.get_node_embeddings(data)

        # 全局池化（不再需要单独的注意力加权）
        global_emb = global_add_pool(x, data.batch)

        # 分类（保持输出不变）
        out = self.fc(global_emb)
        return out  # 二分类直接输出logits


def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    for data, target in loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    # 计算总体准确率
    accuracy = correct / total if total > 0 else 0

    return accuracy


def cross_validation(all_raw_matrices, labels, n_splits=5, epochs_per_fold=100):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=18)
    fold_best_val_acc = []  # 每折最佳验证准确率
    fold_best_thresholds = []  # 每折最佳阈值
    fold_se = []  # 每折最佳阈值对应的所有类别灵敏度（Se）展平后的一维数组
    fold_sp = []  # 每折最佳阈值对应的所有类别特异性（Sp）展平后的一维数组
    fold_p = []  # 每折最佳阈值对应的所有类别精确度（P）展平后的一维数组

    # 计算全局类别数（从训练/验证标签中获取）
    num_classes = len(torch.unique(labels))  # 根据实际标签确定类别数

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_raw_matrices)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # 划分训练集和验证集
        train_matrices = [all_raw_matrices[i] for i in train_idx]
        val_matrices = [all_raw_matrices[i] for i in val_idx]
        train_labels_fold = labels[train_idx]
        val_labels_fold = labels[val_idx]

        # 初始化模型和优化器（显式指定设备）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GNNClassifier(num_classes=num_classes).to(device)  # 传入实际类别数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        # 自适应阈值训练（传递类别数）
        model, best_threshold, best_val_acc, best_se, best_sp, best_p = adaptive_threshold_training(
            model, optimizer,
            train_matrices, train_labels_fold,
            val_matrices, val_labels_fold,
            epochs=epochs_per_fold,
            device=device,
            num_classes=num_classes  # 显式传递类别数
        )

        # 存储结果（将每折的多类别指标展平为一维数组）
        fold_best_val_acc.append(best_val_acc)
        fold_best_thresholds.append(best_threshold)
        fold_se.append(best_se.flatten())  # 展平为(n_classes,)
        fold_sp.append(best_sp.flatten())  # 展平为(n_classes,)
        fold_p.append(best_p.flatten())  # 展平为(n_classes,)

        # 打印单折结果（仅保留关键指标）
        print(f"Fold {fold + 1} best threshold: {best_threshold}%")
        print(f"Fold {fold + 1} Best Validation Accuracy: {best_val_acc:.4f}")

    # 计算跨折整体指标（所有类别+所有折的均值±标准差）
    def compute_overall_metric(fold_metrics):
        """计算跨折整体指标的均值和标准差（展平所有类别后）"""
        all_metrics = np.concatenate(fold_metrics)  # 展平为一维数组（总样本数×类别数）
        mean = np.mean(all_metrics)
        std = np.std(all_metrics)
        return f"{mean:.4f} ± {std:.4f}"

    # 输出最终结果
    print("\nK-Fold Results:")
    print(f"  Avg Best Validation Accuracy: {np.mean(fold_best_val_acc):.4f} ± {np.std(fold_best_val_acc):.4f}")
    print(f"  Best Thresholds: {fold_best_thresholds}")
    print(f"  Sensitivity (Se): {compute_overall_metric(fold_se)}")
    print(f"  Specificity (Sp): {compute_overall_metric(fold_sp)}")
    print(f"  Precision (P): {compute_overall_metric(fold_p)}")

    return fold_best_val_acc, fold_best_thresholds, fold_se, fold_sp, fold_p


def adaptive_threshold_training(model, optimizer, train_matrices, train_labels,
                                val_matrices, val_labels, epochs=100, device='cpu', num_classes=3):
    current_threshold = 0.5  # 恢复原始初始阈值0.5
    best_val_acc = 0.0
    best_model_state = None
    best_threshold = current_threshold
    best_se = np.zeros(num_classes)
    best_sp = np.zeros(num_classes)
    best_p = np.zeros(num_classes)
    
    # 存储最佳模型的预测结果
    best_all_targets = []
    best_all_preds = []

    # 存储所有训练数据集的图特征（动态更新）
    train_feature_cache = {}

    # 预先处理验证集数据，使用折内标准化的动态阈值
    val_dataset_fixed = dynamic_threshold_preprocessing_with_intra_normalization(val_matrices, current_threshold * 100)
    for i, data in enumerate(val_dataset_fixed):
        data.y = torch.tensor([val_labels[i]], dtype=torch.long)
        data.original_adj = torch.tensor(val_matrices[i], dtype=torch.float32)

    # 自适应阈值调整参数（恢复原始参数）
    learning_rate = 0.1  # 恢复原始学习率
    momentum = 0.5  # 恢复原始动量
    prev_total_loss = None
    
    # 损失历史记录用于自适应阶段判断
    loss_history = []
    current_stage = 0  # 初始训练阶段

    # 定义损失权重调整策略（基于损失变化的自适应方法）
    def get_loss_weights(loss_history, current_stage):
        """根据损失变化自适应调整训练阶段"""
        # 需要至少2个历史点才能计算变化
        if len(loss_history) < 2:
            stage = 0
        else:
            # 计算最近几次损失的平均变化率
            window_size = min(5, len(loss_history))  # 使用最近5次或全部历史
            recent_losses = loss_history[-window_size:]
            
            # 计算损失变化率
            if len(recent_losses) >= 2:
                changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
                avg_change = sum(changes) / len(changes)
                relative_change = avg_change / (recent_losses[0] + 1e-8)  # 避免除零
                
                # 根据损失变化率判断是否进入下一阶段
                if current_stage == 0 and relative_change < 0.01:  # 第一阶段到第二阶段
                    stage = 1
                elif current_stage == 1 and relative_change < 0.01:  # 第二阶段到第三阶段
                    stage = 2
                else:
                    stage = current_stage
            else:
                stage = current_stage
        
        # 根据阶段返回相应的损失权重
        if stage == 0:  # 初始阶段：侧重于重构和对比损失
            return 0.4, 0.4, 0.2, stage  # λ_recon, λ_cont, λ_cls, stage
        elif stage == 1:  # 中间阶段：增强对比损失
            return 0.3, 0.5, 0.2, stage
        else:  # 最终阶段：聚焦分类损失
            return 0.2, 0.3, 0.5, stage

    for epoch in range(epochs):
        # 获取当前阶段的损失权重
        lambda_recon, lambda_cont, lambda_cls, current_stage = get_loss_weights(loss_history, current_stage)

        # 1. 只使用当前阈值创建训练集图数据（验证集数据已预先固定）
        if current_threshold not in train_feature_cache:
            # 使用折内标准化的动态阈值处理
            train_feature_cache[current_threshold] = dynamic_threshold_preprocessing_with_intra_normalization(
                train_matrices, current_threshold * 100)

        train_dataset = train_feature_cache[current_threshold]

        # 为训练集图添加标签
        for i, data in enumerate(train_dataset):
            data.y = torch.tensor([train_labels[i]], dtype=torch.long)
            # 保存原始邻接矩阵用于计算重构损失
            data.original_adj = torch.tensor(train_matrices[i], dtype=torch.float32)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset_fixed, batch_size=8)  # 使用固定的验证集

        # 2. 训练步骤
        model.train()
        total_loss = 0
        cls_loss_total = 0
        recon_loss_total = 0
        cont_loss_total = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # 模型前向传播
            output = model(data)

            # ======================== 计算分类损失 ========================
            # 确保output和target具有相同的维度
            cls_loss = F.cross_entropy(output, data.y)  # 修改为交叉熵损失

            # ======================== 计算重构损失 ========================
            # 这里计算重构损失：原始邻接矩阵与当前阈值邻接矩阵的差异
            original_adj = data.original_adj.to(device)

            # 计算更准确的重构损失
            # 提取特征矩阵中的邻接表示部分（前19列）
            reconstructed_adj = data.x[:, :19].to(device)

            # 使用均方误差作为重构损失
            recon_loss = F.mse_loss(reconstructed_adj, original_adj)

            # ======================== 计算对比损失 ========================
            # 使用节点嵌入计算对比损失
            node_embeddings = model.get_node_embeddings(data)
            normalized_emb = F.normalize(node_embeddings, p=2, dim=1)

            contrast_loss = torch.tensor(0.0, device=device)

            # 仅当有边存在时计算对比损失
            if data.edge_index.size(1) > 0:
                # 获取相邻节点的索引对
                adj_pairs = data.edge_index.t()

                # 计算正样本对（相邻节点）的相似度
                if adj_pairs.size(0) > 0:  # 确保有有效的边
                    pos_sim = torch.cosine_similarity(
                        normalized_emb[adj_pairs[:, 0]],
                        normalized_emb[adj_pairs[:, 1]]
                    )
                    # 正样本标签 (1表示相邻节点对)
                    pos_labels = torch.ones_like(pos_sim)
                else:
                    pos_sim = torch.tensor([], device=device)
                    pos_labels = torch.tensor([], device=device)

                # 创建负样本对 - 随机选择不相邻的节点对
                num_nodes = normalized_emb.size(0)
                num_neg_samples = min(adj_pairs.size(0), num_nodes * (num_nodes - 1) // 2)

                if num_neg_samples > 0:
                    # 随机选择第一组节点
                    neg_indices1 = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
                    # 随机选择第二组节点（确保不同节点）
                    neg_indices2 = torch.randint(0, num_nodes, (num_neg_samples,), device=device)

                    # 确保不是自环
                    non_self_mask = neg_indices1 != neg_indices2
                    neg_indices1 = neg_indices1[non_self_mask]
                    neg_indices2 = neg_indices2[non_self_mask]

                    if len(neg_indices1) > 0:
                        # 计算负样本相似度
                        neg_sim = torch.cosine_similarity(
                            normalized_emb[neg_indices1],
                            normalized_emb[neg_indices2]
                        )
                        # 负样本标签 (0表示不相邻节点对)
                        neg_labels = torch.zeros_like(neg_sim)

                # 合并正负样本和标签
                all_sim = torch.cat([pos_sim, neg_sim]) if len(pos_sim) > 0 and len(neg_sim) > 0 else pos_sim if len(
                    pos_sim) > 0 else neg_sim
                all_labels = torch.cat([pos_labels, neg_labels]) if len(pos_sim) > 0 and len(
                    neg_sim) > 0 else pos_labels if len(pos_sim) > 0 else neg_labels

                if len(all_sim) > 0:
                    contrast_loss = F.binary_cross_entropy_with_logits(
                        all_sim, all_labels
                    )

            # ======================== 智能损失缩放 ========================
            # 如果任何损失分量超过100，则除以它的数量级
            def scale_loss(loss):
                if loss.item() > 100:
                    # 计算数量级（10的位数次方）
                    order_of_magnitude = 10 ** math.floor(math.log10(loss.item()))
                    # 除以数量级
                    return loss / order_of_magnitude
                return loss

            # 应用智能缩放
            recon_loss = scale_loss(recon_loss)
            contrast_loss = scale_loss(contrast_loss)
            cls_loss = scale_loss(cls_loss)
            # 组合加权损失
            loss = (
                    lambda_cls * cls_loss +
                    lambda_recon * recon_loss +
                    lambda_cont * contrast_loss
            )

            loss.backward()
            optimizer.step()

            # 累计损失用于分析
            total_loss += loss.item()
            cls_loss_total += cls_loss.item()
            recon_loss_total += recon_loss.item()
            cont_loss_total += contrast_loss.item()

        # 3. 验证评估（使用固定验证集）
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                preds = output.argmax(dim=1)  # 修改为argmax获取预测类别
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(data.y.cpu().numpy())

        # 计算准确率
        val_acc = accuracy_score(all_targets, all_preds)

        # 计算多分类指标
        conf_mat = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
        class_metrics = {
            'se': [],
            'sp': [],
            'p': []
        }

        for c in range(num_classes):
            tp = conf_mat[c, c]
            fp = conf_mat[:, c].sum() - tp
            fn = conf_mat[c, :].sum() - tp
            tn = conf_mat.sum() - tp - fp - fn

            se = tp / (tp + fn) if tp + fn > 0 else 0
            sp = tn / (tn + fp) if tn + fp > 0 else 0
            p_val = tp / (tp + fp) if tp + fp > 0 else 0

            class_metrics['se'].append(se)
            class_metrics['sp'].append(sp)
            class_metrics['p'].append(p_val)

        # 4. 更新最佳模型（仅当验证准确率提高时）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_threshold = current_threshold
            best_se = np.array(class_metrics['se'])
            best_sp = np.array(class_metrics['sp'])
            best_p = np.array(class_metrics['p'])
            best_model_state = model.state_dict().copy()
            # 保存最佳模型的预测结果
            best_all_targets = all_targets.copy()
            best_all_preds = all_preds.copy()

        # 记录本轮总损失到历史
        loss_history.append(total_loss)
        
        # 5. 智能阈值调整策略（关键修改：只基于训练损失调整阈值，不考虑验证集）
        # 基于总损失变化调整阈值，仅使用训练集信息
        if prev_total_loss is None:
            loss_change = 0.01  # 设置一个小的初始变化值，避免零值
        else:
            loss_change = prev_total_loss - total_loss  # 损失下降为正

        prev_total_loss = total_loss

        # 计算阈值调整量
        # 使用更大的调整幅度和自适应灵敏度
        sensitivity_factor = max(lambda_recon, lambda_cont, lambda_cls) * 0.4
        threshold_delta = loss_change * learning_rate * sensitivity_factor * 10

        # 应用动量更新
        new_threshold = current_threshold + threshold_delta
        current_threshold = momentum * current_threshold + (1 - momentum) * new_threshold

        # 限制阈值在0.3-0.7之间
        current_threshold = max(0.3, min(current_threshold, 0.7))

        # 每20个epoch或最后一个epoch打印一次训练进度
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            # 计算Precision和Recall
            if len(all_targets) > 0:
                # 计算宏平均Precision和Recall
                from sklearn.metrics import precision_score, recall_score
                precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
                print(f"Epoch {epoch + 1}/{epochs}: Threshold={current_threshold:.3f} | "
                      f"Loss={total_loss:.4f} (Cls:{cls_loss_total:.4f}, Recon:{recon_loss_total:.4f}, Cont:{cont_loss_total:.4f}) | "
                      f"ΔThreshold={threshold_delta:.4f} | "
                      f"Weights: λr={lambda_recon:.1f}, λc={lambda_cont:.1f}, λcls={lambda_cls:.1f} | "
                      f"Stage: {current_stage} | "
                      f"Val Acc={val_acc:.4f} (Best {best_val_acc:.4f}) | "
                      f"Precision={precision:.4f}, Recall={recall:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}: Threshold={current_threshold:.3f} | "
                      f"Loss={total_loss:.4f} (Cls:{cls_loss_total:.4f}, Recon:{recon_loss_total:.4f}, Cont:{cont_loss_total:.4f}) | "
                      f"ΔThreshold={threshold_delta:.4f} | "
                      f"Weights: λr={lambda_recon:.1f}, λc={lambda_cont:.1f}, λcls={lambda_cls:.1f} | "
                      f"Stage: {current_stage} | "
                      f"Val Acc={val_acc:.4f} (Best {best_val_acc:.4f})")

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_threshold, best_val_acc, best_se, best_sp, best_p, best_all_targets, best_all_preds


def create_binary_dataset(original_matrices, original_labels, class_a, class_b):
    """创建指定两个类别的二分类数据集"""
    binary_matrices = []
    binary_labels = []

    for i, label in enumerate(original_labels):
        if label == class_a or label == class_b:
            binary_matrices.append(original_matrices[i])
            # 将class_a设为0，class_b设为1
            binary_labels.append(0 if label == class_a else 1)

    return np.array(binary_matrices), torch.tensor(binary_labels, dtype=torch.long)


def visualize_confusion_matrix(true_labels, pred_labels, class_names, title="Confusion Matrix", save_path=None):
    """
    可视化混淆矩阵
    
    Args:
        true_labels: 真实标签
        pred_labels: 预测标签
        class_names: 类别名称列表
        title: 图表标题
        save_path: 保存路径（可选）
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 确保标签是整数类型，如果是浮点数则转换为整数
    true_labels = np.round(np.array(true_labels)).astype(int)
    pred_labels = np.round(np.array(pred_labels)).astype(int)
    
    # 获取所有的唯一标签值
    unique_labels = np.unique(np.concatenate([true_labels, pred_labels]))
    
    # 确保标签范围是连续的（对于三分类通常是0, 1, 2）
    # 如果标签不连续，创建完整的标签序列
    min_label = min(unique_labels)
    max_label = max(unique_labels)
    full_labels = list(range(min_label, max_label + 1))
    
    # 计算混淆矩阵，使用完整的标签范围以确保矩阵结构正确
    cm = confusion_matrix(true_labels, pred_labels, labels=full_labels)
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    
    # 为标签创建适当的类别名称
    # 如果类别名称列表足够长，使用对应的名称；否则使用标签值
    tick_labels = []
    for label in full_labels:
        if label < len(class_names):
            tick_labels.append(class_names[label])
        else:
            tick_labels.append(str(label))  # 如果超出类别名称范围，使用标签值
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=tick_labels, 
                yticklabels=tick_labels)
    
    # 设置标题和轴标签
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 如果指定了保存路径，则保存图片
    if save_path:
        # 确保目录存在
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    
    # 显示图形
    plt.show()


def fixed_threshold_training(model, optimizer, train_matrices, train_labels,
                           val_matrices, val_labels, epochs=100, device='cpu', num_classes=3):  # 修改为num_classes=3
    """固定阈值训练函数，用于消融实验对比，使用全局标准化"""
    # 计算全局最小最大值用于全局标准化
    all_values = np.concatenate([matrix.flatten() for matrix in train_matrices])
    global_min = all_values.min()
    global_max = all_values.max()
    fixed_threshold = 0.5  # 使用一个固定的阈值
    
    # 使用全局标准化预处理训练集和验证集数据
    train_dataset, _, _ = fixed_threshold_preprocessing_with_global_normalization(train_matrices, fixed_threshold, global_min, global_max)
    val_dataset, _, _ = fixed_threshold_preprocessing_with_global_normalization(val_matrices, fixed_threshold, global_min, global_max)
    
    for i, data in enumerate(train_dataset):
        data.y = torch.tensor([train_labels[i]], dtype=torch.long)
        data.original_adj = torch.tensor(train_matrices[i], dtype=torch.float32)
    
    for i, data in enumerate(val_dataset):
        data.y = torch.tensor([val_labels[i]], dtype=torch.long)
        data.original_adj = torch.tensor(val_matrices[i], dtype=torch.float32)
    
    best_val_acc = 0.0
    best_model_state = None
    best_se = np.zeros(num_classes)
    best_sp = np.zeros(num_classes)
    best_p = np.zeros(num_classes)
    
    # 存储最佳模型的预测结果
    best_all_targets = []
    best_all_preds = []

    # 固定阈值方法只使用基础分类损失进行训练
    for epoch in range(epochs):
        # 训练步骤
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        model.train()
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            # 只使用基本的分类损失
            loss = F.cross_entropy(output, data.y)
            
            loss.backward()
            optimizer.step()

        # 验证步骤
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(data.y.cpu().numpy())
        
        val_acc = accuracy_score(all_targets, all_preds)
        
        # 计算多分类指标
        conf_mat = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
        class_metrics = {
            'se': [],
            'sp': [],
            'p': []
        }
        
        for c in range(num_classes):
            tp = conf_mat[c, c]
            fp = conf_mat[:, c].sum() - tp
            fn = conf_mat[c, :].sum() - tp
            tn = conf_mat.sum() - tp - fp - fn
            
            se = tp / (tp + fn) if tp + fn > 0 else 0
            sp = tn / (tn + fp) if tn + fp > 0 else 0
            p_val = tp / (tp + fp) if tp + fp > 0 else 0
            
            class_metrics['se'].append(se)
            class_metrics['sp'].append(sp)
            class_metrics['p'].append(p_val)
        
        # 更新最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_se = np.array(class_metrics['se'])
            best_sp = np.array(class_metrics['sp'])
            best_p = np.array(class_metrics['p'])
            best_model_state = model.state_dict().copy()
            # 保存最佳模型的预测结果
            best_all_targets = all_targets.copy()
            best_all_preds = all_preds.copy()
        
        if (epoch + 1) % 50 == 0:
            print(f"Fixed Threshold (Global Norm) - Epoch {epoch + 1}/{epochs}: Val Acc={val_acc:.4f} (Best {best_val_acc:.4f})")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, 30.0, best_val_acc, best_se, best_sp, best_p, best_all_targets, best_all_preds


def cross_validation_ablation(all_raw_matrices, labels, n_splits=5, epochs_per_fold=100):
    """消融实验的交叉验证函数，比较动态阈值和固定阈值"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=18)
    
    # 用于存储动态阈值方法的结果
    dynamic_results = {
        'acc': [], 'thresholds': [], 'se': [], 'sp': [], 'p': [], 'all_true': [], 'all_pred': []
    }
    
    # 用于存储固定阈值方法的结果
    fixed_results = {
        'acc': [], 'se': [], 'sp': [], 'p': [], 'all_true': [], 'all_pred': []
    }
    
    num_classes = len(torch.unique(labels))  # 根据实际标签确定类别数

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_raw_matrices)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # 划分训练集和验证集
        train_matrices = [all_raw_matrices[i] for i in train_idx]
        val_matrices = [all_raw_matrices[i] for i in val_idx]
        train_labels_fold = labels[train_idx]
        val_labels_fold = labels[val_idx]

        # 初始化模型和优化器（动态阈值方法）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dynamic = GNNClassifier(num_classes=num_classes).to(device)
        optimizer_dynamic = torch.optim.Adam(model_dynamic.parameters(), lr=0.01, weight_decay=1e-4)
        
        # 动态阈值训练
        _, best_threshold, best_val_acc_dynamic, best_se, best_sp, best_p, all_targets_dynamic, all_preds_dynamic = adaptive_threshold_training(
            model_dynamic, optimizer_dynamic,
            train_matrices, train_labels_fold,
            val_matrices, val_labels_fold,
            epochs=epochs_per_fold,
            device=device,
            num_classes=num_classes
        )
        
        # 存储动态阈值方法结果
        dynamic_results['acc'].append(best_val_acc_dynamic)
        dynamic_results['thresholds'].append(best_threshold)
        dynamic_results['se'].append(best_se.flatten())
        dynamic_results['sp'].append(best_sp.flatten())
        dynamic_results['p'].append(best_p.flatten())
        dynamic_results['all_true'].extend(all_targets_dynamic)
        dynamic_results['all_pred'].extend(all_preds_dynamic)

        # 初始化模型和优化器（固定阈值方法）
        model_fixed = GNNClassifier(num_classes=num_classes).to(device)
        optimizer_fixed = torch.optim.Adam(model_fixed.parameters(), lr=0.01, weight_decay=1e-4)
        
        # 固定阈值训练
        _, _, best_val_acc_fixed, best_se_fixed, best_sp_fixed, best_p_fixed, all_targets_fixed, all_preds_fixed = fixed_threshold_training(
            model_fixed, optimizer_fixed,
            train_matrices, train_labels_fold,
            val_matrices, val_labels_fold,
            epochs=epochs_per_fold,
            device=device,
            num_classes=num_classes
        )
        
        # 存储固定阈值方法结果
        fixed_results['acc'].append(best_val_acc_fixed)
        fixed_results['se'].append(best_se_fixed.flatten())
        fixed_results['sp'].append(best_sp_fixed.flatten())
        fixed_results['p'].append(best_p_fixed.flatten())
        fixed_results['all_true'].extend(all_targets_fixed)
        fixed_results['all_pred'].extend(all_preds_fixed)

        print(f"Fold {fold + 1} Dynamic Threshold: {best_threshold}%")
        print(f"Fold {fold + 1} Dynamic Acc: {best_val_acc_dynamic:.4f}, Fixed Acc: {best_val_acc_fixed:.4f}")

    # 输出消融实验结果
    print("\n=== 消融实验结果 ===")
    print(f"动态阈值方法 - 平均准确率: {np.mean(dynamic_results['acc']):.4f} ± {np.std(dynamic_results['acc']):.4f}")
    print(f"固定阈值方法 - 平均准确率: {np.mean(fixed_results['acc']):.4f} ± {np.std(fixed_results['acc']):.4f}")
    print(f"动态阈值方法的优势: {np.mean(dynamic_results['acc']) - np.mean(fixed_results['acc']):.4f}")
    
    # 计算整体指标（包括所有类别）
    def compute_overall_metric(fold_metrics):
        all_metrics = np.concatenate(fold_metrics)
        mean = np.mean(all_metrics)
        std = np.std(all_metrics)
        return f"{mean:.4f} ± {std:.4f}"
    
    print(f"动态阈값 - 灵敏度 (Se): {compute_overall_metric(dynamic_results['se'])}")
    print(f"动态阈값 - 特异性 (Sp): {compute_overall_metric(dynamic_results['sp'])}")
    print(f"动态阈값 - 精确度 (P): {compute_overall_metric(dynamic_results['p'])}")
    
    print(f"固定阈값 - 灵敏度 (Se): {compute_overall_metric(fixed_results['se'])}")
    print(f"固定阈값 - 特异性 (Sp): {compute_overall_metric(fixed_results['sp'])}")
    print(f"固定阈값 - 精确度 (P): {compute_overall_metric(fixed_results['p'])}")
    
    return dynamic_results, fixed_results


# 创建原始三分类标签
labels = torch.cat([
    torch.zeros(36),  # 类别0 —— AD
    torch.ones(29),  # 类别1 —— HC
    torch.full((23,), 2)  # 类别2 —— FTD
]).long()

# 检查是否安装了必要的库
if TORCH_GEOMETRIC_AVAILABLE:
    # 选择要使用的数据集
    selected_dataset = copula

    # 运行消融实验（对三分类进行对比）
    print("\n=== 消融实验对比 ===")
    
    # 运行动态阈值和固定阈值的对比实验
    dynamic_results, fixed_results = cross_validation_ablation(
        selected_dataset, labels, n_splits=5, epochs_per_fold=500
    )
    
    # 收集结果
    task_result = {
        "task": "0vs1vs2",
        "dynamic_acc": np.mean(dynamic_results['acc']),
        "dynamic_std": np.std(dynamic_results['acc']),
        "fixed_acc": np.mean(fixed_results['acc']),
        "fixed_std": np.std(fixed_results['acc'])
    }
    all_results = [task_result]  # 将单个三分类任务放入列表，以保持与原代码兼容
    
    # 消融实验结果汇总
    print("\n=== 消融实验结果汇总 ===")
    print(f"{'任务':<10} {'动态阈值准确率':<18} {'固定阈值准确率':<18} {'性能提升':<12}")
    print("-" * 70)
    
    for result in all_results:
        improvement = result['dynamic_acc'] - result['fixed_acc']
        print(f"{result['task']:<10} {result['dynamic_acc']:.4f}±{result['dynamic_std']:.4f}      "
              f"{result['fixed_acc']:.4f}±{result['fixed_std']:.4f}      {improvement:.4f}")
    
    # 总体性能对比
    avg_dynamic_acc = np.mean([r['dynamic_acc'] for r in all_results])
    avg_fixed_acc = np.mean([r['fixed_acc'] for r in all_results])
    overall_improvement = avg_dynamic_acc - avg_fixed_acc
    
    print("-" * 70)
    print(f"{'总体平均':<10} {avg_dynamic_acc:.4f}±{np.mean([r['dynamic_std'] for r in all_results]):.4f}      "
          f"{avg_fixed_acc:.4f}±{np.mean([r['fixed_std'] for r in all_results]):.4f}      {overall_improvement:.4f}")
    
    print(f"\n总体性能提升: {overall_improvement:.4f} ({overall_improvement/avg_fixed_acc*100:.2f}% improvement)")

    # 生成混淆矩阵
    print("\n正在生成混淆矩阵可视化...\n")

    # 定义类别名称
    class_names = ["AD", "HC", "FTD"]

    # 生成动态阈值混淆矩阵
    dynamic_save_path = "C:/Users/Administrator/Desktop/小论文图/三分类_动态阈值_混淆矩阵.pdf"
    visualize_confusion_matrix(
        dynamic_results['all_true'],
        dynamic_results['all_pred'],
        class_names,
        title="三分类 - 动态阈值方法混淆矩阵",
        save_path=dynamic_save_path
    )

    # 生成固定阈值混淆矩阵
    fixed_save_path = "C:/Users/Administrator/Desktop/小论文图/三分类_固定阈值_混淆矩阵.pdf"
    visualize_confusion_matrix(
        fixed_results['all_true'],
        fixed_results['all_pred'],
        class_names,
        title="三分类 - 固定阈值方法混淆矩阵",
        save_path=fixed_save_path
    )

    # 添加等待输入以防止窗口关闭
    input("\n按回车键退出...")
