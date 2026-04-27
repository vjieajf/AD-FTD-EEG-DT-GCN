import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv, GCNConv, global_add_pool

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    Data = DataLoader = GATConv = GCNConv = global_add_pool = None
    TORCH_GEOMETRIC_AVAILABLE = False


# -------------------------
# Config
# -------------------------
OUTPUT_DIR = r"G:\Daima\pythonProject\保存数据"
DATASET_FILES = {
    "copula": r"G:\Daima\pythonProject\保存数据\avg_copula_matrix_multiband.npy",
    "granger": r"G:\Daima\pythonProject\保存数据\avg_granger_matrix_multiband.npy",
    "pearson": r"G:\Daima\pythonProject\保存数据\avg_pearson_matrix_multiband.npy",
    "plv": r"G:\Daima\pythonProject\保存数据\avg_plv_matrix_multiband.npy",
}

N_CHANNELS = 19
N_BANDS = 5
N_CLASSES = 2
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

# 标签顺序与原项目保持一致：AD(36), HC(29), FTD(23)
LABELS = np.array([0] * 36 + [1] * 29 + [2] * 23, dtype=np.int64)
CLASS_NAME_MAP = {0: "AD", 1: "HC", 2: "FTD"}

# 仅跑用户指定的 4 个数据集-频段组合
SELECTED_CASES = [
    ("copula", "beta"),
    ("granger", "alpha"),
    ("pearson", "beta"),
    ("plv", "beta"),
]

# 三分类标签的两两二分类：AD-HC, AD-FTD, HC-FTD
PAIRWISE_CLASS_PAIRS = [(0, 1), (0, 2), (1, 2)]

SEED = 18
N_SPLITS = 5
EPOCHS = 300
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD_PERCENT = 70  # 用于根据融合邻接矩阵构图


# -------------------------
# Utils
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_multiband_fc(path):
    data = np.load(path)
    if data.ndim != 3:
        raise ValueError(f"期望三维数组 (N, 19, 95)，实际维度: {data.shape}")

    if data.shape[1] != N_CHANNELS or data.shape[2] != N_CHANNELS * N_BANDS:
        raise ValueError(
            f"期望形状 (N, {N_CHANNELS}, {N_CHANNELS * N_BANDS})，实际: {data.shape}"
        )

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return data


def build_graph_dataset(raw_data, labels, threshold_percent=70):
    dataset = []
    for i in range(len(raw_data)):
        dataset.append(sample_to_graph(raw_data[i], labels[i], threshold_percent=threshold_percent))
    return dataset


def sample_to_graph(sample_19x19, label, threshold_percent=70):
    # 单频段邻接矩阵: (19, 19)
    adj = np.asarray(sample_19x19, dtype=np.float32)

    fused_adj = np.abs(adj)
    np.fill_diagonal(fused_adj, 0.0)

    # 使用分位数阈值生成边
    threshold = np.percentile(fused_adj, threshold_percent)
    binary_adj = (fused_adj > threshold).astype(np.int64)
    np.fill_diagonal(binary_adj, 1)  # 保证每个节点至少有自环

    # 节点特征: one-hot(19) + 出强度(1) + 入强度(1)
    out_strength = adj.sum(axis=1, keepdims=True)  # (19, 1)
    in_strength = adj.sum(axis=0, keepdims=True).T  # (19, 1)
    node_features = np.concatenate(
        [np.eye(N_CHANNELS, dtype=np.float32), out_strength, in_strength],
        axis=1,
    ).astype(np.float32)

    edge_index = np.stack(np.where(binary_adj > 0))

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor([label], dtype=torch.long),
    )


def dynamic_threshold_preprocessing_with_intra_normalization(original_matrices, threshold_percent):
    """对每个矩阵单独标准化后，按当前阈值比例构图。"""
    processed_data = []
    for adj_matrix in original_matrices:
        adj_matrix = np.asarray(adj_matrix, dtype=np.float32)
        adj_min = adj_matrix.min()
        adj_max = adj_matrix.max()
        adj_normalized = (adj_matrix - adj_min) / (adj_max - adj_min + 1e-8)

        threshold = np.percentile(adj_normalized, threshold_percent)
        binary_adj = (adj_normalized > threshold).astype(float)
        np.fill_diagonal(binary_adj, 1.0)

        edge_index = torch.tensor(np.stack(np.where(binary_adj > 0)), dtype=torch.long)
        out_strength = adj_matrix.sum(axis=1, keepdims=True)
        in_strength = adj_matrix.sum(axis=0, keepdims=True).T
        node_features = np.concatenate(
            [np.eye(N_CHANNELS, dtype=np.float32), out_strength, in_strength],
            axis=1,
        ).astype(np.float32)

        processed_data.append(
            Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=edge_index,
            )
        )
    return processed_data


def get_loss_weights(loss_history, current_stage):
    """根据损失变化自适应调整训练阶段。"""
    if len(loss_history) < 2:
        stage = 0
    else:
        window_size = min(5, len(loss_history))
        recent_losses = loss_history[-window_size:]

        if len(recent_losses) >= 2:
            changes = [abs(recent_losses[i] - recent_losses[i - 1]) for i in range(1, len(recent_losses))]
            avg_change = sum(changes) / len(changes)
            relative_change = avg_change / (recent_losses[0] + 1e-8)

            if current_stage == 0 and relative_change < 0.01:
                stage = 1
            elif current_stage == 1 and relative_change < 0.01:
                stage = 2
            else:
                stage = current_stage
        else:
            stage = current_stage

    if stage == 0:
        return 0.4, 0.4, 0.2, stage
    elif stage == 1:
        return 0.3, 0.5, 0.2, stage
    else:
        return 0.2, 0.3, 0.5, stage


def split_multiband_data(raw_data):
    """将 (N, 19, 95) 切分为 5 个频段的 (N, 19, 19)。"""
    n_samples = raw_data.shape[0]
    cube = raw_data.reshape(n_samples, N_CHANNELS, N_CHANNELS, N_BANDS)
    band_data = {}
    for band_idx, band_name in enumerate(BAND_NAMES):
        band_data[band_name] = cube[:, :, :, band_idx]
    return band_data


def build_binary_subset(raw_matrices, labels, class_a, class_b):
    """按给定类别对原始样本做筛选，并将标签重映射为 0/1。"""
    mask = np.isin(labels, [class_a, class_b])
    binary_matrices = raw_matrices[mask]
    selected_labels = labels[mask]
    binary_labels = np.where(selected_labels == class_a, 0, 1).astype(np.int64)
    return binary_matrices, binary_labels


# -------------------------
# Model
# -------------------------
class MultiBandGNN(nn.Module):
    def __init__(self, in_dim, num_classes=3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.gat = GATConv(32, 32, heads=4, concat=False, dropout=0.2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gat(x, edge_index))
        x = global_add_pool(x, batch)
        return self.fc(x)


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def evaluate(model, loader, device, class_labels=None):
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            pred = logits.argmax(dim=1)
            all_true.extend(batch.y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())

    acc = accuracy_score(all_true, all_pred)
    f1_macro = f1_score(all_true, all_pred, average="macro", zero_division=0)
    if class_labels is None:
        class_labels = sorted(set(all_true) | set(all_pred))
    cm = confusion_matrix(all_true, all_pred, labels=class_labels)
    return acc, f1_macro, cm, all_true, all_pred


def run_kfold(raw_data, labels, band_name="unknown", num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_acc = []
    fold_f1 = []
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_data, labels), start=1):
        print(f"\n[{band_name}] Fold {fold}/{N_SPLITS}")

        train_matrices = raw_data[train_idx]
        val_matrices = raw_data[val_idx]

        model = MultiBandGNN(in_dim=N_CHANNELS + 2, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        _, best_threshold, best_acc, best_f1, best_cm, _, _ = adaptive_threshold_training(
            model,
            optimizer,
            train_matrices,
            labels[train_idx],
            val_matrices,
            labels[val_idx],
            epochs=EPOCHS,
            device=device,
            num_classes=num_classes,
        )

        fold_acc.append(best_acc)
        fold_f1.append(best_f1)
        total_cm += best_cm

        print(f"[{band_name}] Fold {fold} Best Threshold: {best_threshold:.2f}%")
        print(f"[{band_name}] Fold {fold} Best Acc: {best_acc:.4f}, Best F1-macro: {best_f1:.4f}")

    return {
        "fold_acc": fold_acc,
        "fold_f1_macro": fold_f1,
        "mean_acc": float(np.mean(fold_acc)),
        "std_acc": float(np.std(fold_acc)),
        "mean_f1_macro": float(np.mean(fold_f1)),
        "std_f1_macro": float(np.std(fold_f1)),
        "confusion_matrix": total_cm,
    }


def adaptive_threshold_training(model, optimizer, train_matrices, train_labels,
                                val_matrices, val_labels, epochs=100, device=None, num_classes=3):
    """将改良脚本中的自适应阈值更新机制迁移到二分类/多分类任务。"""
    if device is None:
        device = torch.device('cpu')

    current_threshold = 0.5
    best_val_acc = 0.0
    best_model_state = None
    best_threshold = current_threshold * 100
    best_se = np.zeros(num_classes)
    best_sp = np.zeros(num_classes)
    best_p = np.zeros(num_classes)
    best_f1 = 0.0
    best_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    best_all_targets = []
    best_all_preds = []

    train_feature_cache = {}
    val_dataset_fixed = dynamic_threshold_preprocessing_with_intra_normalization(val_matrices, current_threshold * 100)
    for i, data in enumerate(val_dataset_fixed):
        data.y = torch.tensor([val_labels[i]], dtype=torch.long)

    learning_rate = 0.1
    momentum = 0.5
    prev_total_loss = None
    loss_history = []
    current_stage = 0

    for epoch in range(epochs):
        lambda_recon, lambda_cont, lambda_cls, current_stage = get_loss_weights(loss_history, current_stage)

        if current_threshold not in train_feature_cache:
            train_feature_cache[current_threshold] = dynamic_threshold_preprocessing_with_intra_normalization(
                train_matrices, current_threshold * 100
            )

        train_dataset = train_feature_cache[current_threshold]
        for i, data in enumerate(train_dataset):
            data.y = torch.tensor([train_labels[i]], dtype=torch.long)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset_fixed, batch_size=BATCH_SIZE, shuffle=False)

        model.train()
        total_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            cls_loss = F.cross_entropy(output, data.y)
            loss = cls_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

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
        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        conf_mat = confusion_matrix(all_targets, all_preds, labels=range(num_classes))

        class_metrics = {'se': [], 'sp': [], 'p': []}
        for c in range(num_classes):
            tp = conf_mat[c, c]
            fp = conf_mat[:, c].sum() - tp
            fn = conf_mat[c, :].sum() - tp
            tn = conf_mat.sum() - tp - fp - fn
            class_metrics['se'].append(tp / (tp + fn) if tp + fn > 0 else 0)
            class_metrics['sp'].append(tn / (tn + fp) if tn + fp > 0 else 0)
            class_metrics['p'].append(tp / (tp + fp) if tp + fp > 0 else 0)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_threshold = current_threshold * 100
            best_se = np.array(class_metrics['se'])
            best_sp = np.array(class_metrics['sp'])
            best_p = np.array(class_metrics['p'])
            best_f1 = val_f1
            best_cm = conf_mat
            best_model_state = model.state_dict().copy()
            best_all_targets = all_targets.copy()
            best_all_preds = all_preds.copy()

        loss_history.append(total_loss)

        if prev_total_loss is None:
            loss_change = 0.01
        else:
            loss_change = prev_total_loss - total_loss
        prev_total_loss = total_loss

        sensitivity_factor = max(lambda_recon, lambda_cont, lambda_cls) * 0.4
        threshold_delta = loss_change * learning_rate * sensitivity_factor * 10

        new_threshold = current_threshold + threshold_delta
        current_threshold = momentum * current_threshold + (1 - momentum) * new_threshold
        current_threshold = max(0.3, min(current_threshold, 0.7))

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1}/{epochs}: Threshold={current_threshold:.3f} | "
                f"Loss={total_loss:.4f} | ΔThreshold={threshold_delta:.4f} | "
                f"Stage={current_stage} | Val Acc={val_acc:.4f} (Best {best_val_acc:.4f})"
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_threshold, best_val_acc, best_f1, best_cm, best_all_targets, best_all_preds


def save_results(all_results, cm_stack, case_order):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cm_path = os.path.join(OUTPUT_DIR, "selected4_pairwise_2class_confusion_matrix.npy")
    np.save(cm_path, cm_stack)

    json_path = os.path.join(OUTPUT_DIR, "selected4_pairwise_2class_results.json")
    payload = {
        "selected_cases": [f"{d}/{b}" for d, b in SELECTED_CASES],
        "pairwise_pairs": [
            f"{CLASS_NAME_MAP[a]}({a}) vs {CLASS_NAME_MAP[b]}({b})" for a, b in PAIRWISE_CLASS_PAIRS
        ],
        "case_order": case_order,
        "results": all_results,
        "config": {
            "data_paths": DATASET_FILES,
            "n_splits": N_SPLITS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "threshold_percent": THRESHOLD_PERCENT,
            "seed": SEED,
            "label_order": "AD(36), HC(29), FTD(23)",
            "binary_label_mapping": "pair中前者->0, 后者->1",
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"\n结果已保存:\n- {cm_path}\n- {json_path}")


def main():
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("未安装 torch-geometric，请先安装后再运行。")

    set_seed(SEED)
    all_case_results = {}
    cm_list = []
    case_order = []

    print("开始指定 4 个数据集-频段组合的两两二分类训练...")

    for dataset_name, band_name in SELECTED_CASES:
        dataset_path = DATASET_FILES[dataset_name]
        raw_data = load_multiband_fc(dataset_path)

        if len(raw_data) != len(LABELS):
            raise ValueError(
                f"样本数与标签数不一致：dataset={dataset_name}, data={len(raw_data)}, labels={len(LABELS)}。"
                "请确认标签顺序与数据顺序一致。"
            )

        band_data_dict = split_multiband_data(raw_data)
        selected_band_data = band_data_dict[band_name]

        case_name = f"{dataset_name}/{band_name}"
        print(f"\n===== Case: {case_name} =====")
        print(f"原始数据形状: {raw_data.shape} (N, 19, 95)")

        all_case_results[case_name] = {}

        for class_a, class_b in PAIRWISE_CLASS_PAIRS:
            pair_key = f"{class_a}_vs_{class_b}"
            pair_name = f"{CLASS_NAME_MAP[class_a]} vs {CLASS_NAME_MAP[class_b]}"
            binary_data, binary_labels = build_binary_subset(selected_band_data, LABELS, class_a, class_b)

            print(f"\n--- Pair: {case_name} | {pair_name} ---")
            band_results = run_kfold(
                binary_data,
                binary_labels,
                band_name=f"{case_name} | {pair_name}",
                num_classes=N_CLASSES,
            )

            print(f"[{case_name} | {pair_name}] Accuracy: {band_results['mean_acc']:.4f} ± {band_results['std_acc']:.4f}")
            print(f"[{case_name} | {pair_name}] F1-macro: {band_results['mean_f1_macro']:.4f} ± {band_results['std_f1_macro']:.4f}")
            print(f"[{case_name} | {pair_name}] Confusion Matrix:")
            print(band_results["confusion_matrix"])

            all_case_results[case_name][pair_key] = {
                "pair_name": pair_name,
                "original_labels": [int(class_a), int(class_b)],
                "binary_mapping": {"0": int(class_a), "1": int(class_b)},
                "n_samples": int(len(binary_labels)),
                "fold_acc": band_results["fold_acc"],
                "fold_f1_macro": band_results["fold_f1_macro"],
                "mean_acc": band_results["mean_acc"],
                "std_acc": band_results["std_acc"],
                "mean_f1_macro": band_results["mean_f1_macro"],
                "std_f1_macro": band_results["std_f1_macro"],
                "confusion_matrix": band_results["confusion_matrix"].tolist(),
            }
            cm_list.append(band_results["confusion_matrix"])
            case_order.append(f"{case_name}/{pair_key}")

    cm_stack = np.stack(cm_list, axis=0)  # (12, 2, 2)

    print("\n=== 指定 4 组合 × 3 对类别的二分类结果汇总 ===")
    for dataset_name, band_name in SELECTED_CASES:
        case_name = f"{dataset_name}/{band_name}"
        for class_a, class_b in PAIRWISE_CLASS_PAIRS:
            pair_key = f"{class_a}_vs_{class_b}"
            pair_tag = f"{CLASS_NAME_MAP[class_a]}-{CLASS_NAME_MAP[class_b]}"
            r = all_case_results[case_name][pair_key]
            print(
                f"{dataset_name:<8} | {band_name:<6} | {pair_tag:<11} | "
                f"Acc={r['mean_acc']:.4f}±{r['std_acc']:.4f} | "
                f"F1={r['mean_f1_macro']:.4f}±{r['std_f1_macro']:.4f}"
            )

    save_results(all_case_results, cm_stack, case_order)


if __name__ == "__main__":
    main()
