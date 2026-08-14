"""
流域 预训练 - 跨站点Transformer (CrossStationTransformer)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import pickle
import math
import copy
import random
import warnings
import logging
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)


# 配置
class Config:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = str(REPO_ROOT / "data" / "data_cleaned" / "yangzte")

    INDICATORS = ['CODMn', 'DO', 'NH4N', 'pH']
    INPUT_DIM = 4

    SEQ_LEN = 8
    PRED_LEN = 8

    # 模型参数（保持与原版相同以公平对比）
    D_MODEL = 32
    N_HEADS = 4
    N_LAYERS = 4
    DROPOUT = 0.01#0.076

    # 跨站点Transformer特有参数
    SPATIAL_D_MODEL = 64      # 空间维度的d_model（可以更大）
    SPATIAL_N_HEADS = 4       # 空间注意力头数
    SPATIAL_LAYERS = 2        # 空间交互层数

    # 训练参数
    EPOCHS = 300
    BATCH_SIZE = 32
    BASE_LR = 5e-4
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 10
    MAX_GRAD_NORM = 1.0

    # 课程掩码
    CURRICULUM_START_MASK = 0.2
    CURRICULUM_END_MASK = 0.5
    CURRICULUM_WARMUP = 50

    # 数据增强
    AUGMENT_MULTIPLIER = 5
    MIXUP_ALPHA = 0.4
    MIXUP_PROB = 0.0
    AUGMENT_JITTER_SIGMA = 0.03
    AUGMENT_SCALE_SIGMA = 0.1

    PATIENCE = 40
    SNAPSHOT_INTERVAL = 50

    DISTILL_WEIGHT = 0.0
    DISTILL_START_EPOCH = 999

    TRAIN_SPLIT = 0.8
    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    @staticmethod
    def _get_save_path():
        """按文件名建立文件夹，每次运行在该文件夹下新建时间子文件夹"""
        import sys
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"./results/{script_name}/{timestamp}"
    MODEL_SAVE_PATH = _get_save_path.__func__()


config = Config()


# 日志设置
def setup_logger(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_file = os.path.join(save_path, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logger = logging.getLogger('Training')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# StandardScaler
class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# 数据增强
class TimeSeriesAugmentation:
    @staticmethod
    def jittering(x, sigma=0.03):
        noise = torch.randn_like(x) * sigma
        return x + noise

    @staticmethod
    def scaling(x, sigma=0.1):
        factor = torch.randn(x.shape[0], 1, 1, 1).to(x.device) * sigma + 1.0
        return x * factor

    @staticmethod
    def jitter_and_scale(x, jitter_sigma=0.02, scale_sigma=0.05):
        x = x + torch.randn_like(x) * jitter_sigma
        factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * scale_sigma
        return x * factor

    @staticmethod
    def augment_batch(x, augment_prob=0.7):
        if torch.rand(1).item() > augment_prob:
            return x
        aug_type = torch.randint(0, 3, (1,)).item()
        if aug_type == 0:
            return TimeSeriesAugmentation.jittering(x, config.AUGMENT_JITTER_SIGMA)
        elif aug_type == 1:
            return TimeSeriesAugmentation.scaling(x, config.AUGMENT_SCALE_SIGMA)
        else:
            return TimeSeriesAugmentation.jitter_and_scale(x)

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


# 课程掩码
class CurriculumMasking:
    def __init__(self, start_ratio=0.2, end_ratio=0.5, warmup_epochs=50):
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_epochs = warmup_epochs

    def get_mask_ratio(self, epoch):
        if epoch < self.warmup_epochs:
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * (epoch / self.warmup_epochs)
        else:
            ratio = self.end_ratio
        return ratio

def random_masking(data, mask_ratio):
    mask = torch.rand(data.shape, device=data.device) >= mask_ratio
    masked_data = data.clone()
    masked_data[~mask] = 0
    return masked_data

def parameter_masking(data, mask_ratio):
    masked_data = data.clone()
    batch_size, seq_len, num_feat, num_stations = data.shape
    for i in range(batch_size):
        for station_idx in range(num_stations):
            for param_idx in range(num_feat):
                if torch.rand(1).item() < mask_ratio:
                    masked_data[i, :, param_idx, station_idx] = 0
    return masked_data

def station_masking(data, mask_ratio):
    masked_data = data.clone()
    batch_size, seq_len, num_feat, num_stations = data.shape
    for i in range(batch_size):
        for station_idx in range(num_stations):
            if torch.rand(1).item() < mask_ratio:
                masked_data[i, :, :, station_idx] = 0
    return masked_data

def temporal_masking(data, mask_ratio):
    masked_data = data.clone()
    batch_size, seq_len, num_feat, num_stations = data.shape
    for i in range(batch_size):
        for time_step in range(seq_len):
            if torch.rand(1).item() < mask_ratio:
                masked_data[i, time_step, :, :] = 0
    return masked_data


# 核心创新：跨站点Transformer架构
class TemporalEncoder(nn.Module):
    """时间编码器"""
    def __init__(self, num_feat, seq_len, d_model, dropout=0.1):
        super(TemporalEncoder, self).__init__()
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.d_model = d_model

        # 输入投影：[T, F] → [T, d_model]
        self.input_proj = nn.Linear(num_feat, d_model)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # 时间注意力层
        self.temporal_attn = nn.MultiheadAttention(
            d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, F] - 已经在外部归一化
        B, T, F = x.shape

        # 投影到 d_model
        x = self.input_proj(x)  # [B, T, d_model]

        # 加位置编码
        x = x + self.pos_encoding

        # 时间注意力
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # 池化：[B, T, d_model] → [B, d_model]
        x = x.mean(dim=1)

        return x


class CrossStationAttention(nn.Module):
    """跨站点注意力层"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossStationAttention, self).__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, N, d_model]
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights


class SpatioTemporalDecoder(nn.Module):
    """时空解码器 - 输出归一化空间的重构"""
    def __init__(self, d_model, num_feat, seq_len, pred_len, dropout=0.1):
        super(SpatioTemporalDecoder, self).__init__()
        self.pred_len = pred_len
        self.num_feat = num_feat

        # 从站点表示展开到时间序列
        self.expand = nn.Linear(d_model, pred_len * d_model)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_feat)
        )

    def forward(self, x):
        # x: [B, d_model]
        B = x.shape[0]

        # 展开到时间序列
        x = self.expand(x)  # [B, pred_len * d_model]
        x = x.reshape(B, self.pred_len, -1)  # [B, pred_len, d_model]

        # 解码到特征 - 输出归一化空间的重构
        x = self.decoder(x)  # [B, pred_len, num_feat]

        return x


class CrossStationTransformer(nn.Module):
    """跨站点Transformer主模型"""
    def __init__(self, num_stations, num_feat, seq_len, pred_len,
                 d_model=32, spatial_d_model=64,
                 n_heads=4, spatial_n_heads=4, spatial_layers=2,
                 dropout=0.1):
        super(CrossStationTransformer, self).__init__()
        self.num_stations = num_stations
        self.d_model = d_model
        self.spatial_d_model = spatial_d_model

        # 共享的时间编码器
        self.temporal_encoder = TemporalEncoder(num_feat, seq_len, d_model, dropout)

        # 投影到空间维度
        self.temporal_to_spatial = nn.Linear(d_model, spatial_d_model)

        # 跨站点交互层
        self.cross_station_layers = nn.ModuleList([
            CrossStationAttention(spatial_d_model, spatial_n_heads, dropout)
            for _ in range(spatial_layers)
        ])

        # 投影回时间维度
        self.spatial_to_temporal = nn.Linear(spatial_d_model, d_model)

        # 每个站点独立的解码器
        self.decoders = nn.ModuleList([
            SpatioTemporalDecoder(d_model, num_feat, seq_len, pred_len, dropout)
            for _ in range(num_stations)
        ])

    def forward(self, x):
        # x: [B, T, F, N] - 已经在外部归一化
        B, T, F, N = x.shape

        # 1: 时间编码
        station_embeddings = []
        for i in range(N):
            station_data = x[..., i]  # [B, T, F]
            station_emb = self.temporal_encoder(station_data)
            station_embeddings.append(station_emb)

        # [B, N, d_model]
        station_tokens = torch.stack(station_embeddings, dim=1)

        # 2: 投影到空间维度
        station_tokens = self.temporal_to_spatial(station_tokens)  # [B, N, spatial_d_model]

        # 3: 跨站点交互
        attn_weights_list = []
        for cross_layer in self.cross_station_layers:
            station_tokens, attn_weights = cross_layer(station_tokens)
            attn_weights_list.append(attn_weights)

        # ===== Step 4: 投影回时间维度 =====
        station_tokens = self.spatial_to_temporal(station_tokens)  # [B, N, d_model]

        # ===== Step 5: 为每个站点解码 =====
        predictions = []
        for i in range(N):
            station_repr = station_tokens[:, i]  # [B, d_model]
            pred = self.decoders[i](station_repr)
            predictions.append(pred)

        # [B, T_out, F, N] - 输出归一化空间的重构
        output = torch.stack(predictions, dim=-1)

        return output



# 数据集 - 修改为掩码重构任务
class PreTrainDataset(Dataset):
    def __init__(self, df_list, seq_len, num_feat, mode="train", train_split=0.8):
        self.df_list = df_list
        self.seq_len = seq_len
        self.num_feat = num_feat
        self.num_stations = len(df_list)
        self.mode = mode
        self.train_split = train_split

        self.samples = []
        min_len = float('inf')
        for df in df_list:
            split_idx = int(len(df) * train_split)
            if mode == "train":
                data_len = split_idx
            else:
                data_len = len(df) - split_idx

            # 修改：只需要 seq_len 长度
            if data_len > seq_len:
                min_len = min(min_len, data_len)

        # 修改：样本数量计算
        num_samples = min_len - seq_len
        for i in range(num_samples):
             self.samples.append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx = self.samples[idx]

        x_tensor = torch.zeros(self.seq_len, self.num_feat, self.num_stations)

        for station_idx, df in enumerate(self.df_list):
            split_idx = int(len(df) * self.train_split)

            if self.mode == "train":
                data = df[:split_idx]
            else:
                data = df[split_idx:]

            # 修改：只读取 seq_len 长度
            if start_idx + self.seq_len <= len(data):
                x = data[start_idx:start_idx + self.seq_len]
                x_tensor[:, :, station_idx] = torch.FloatTensor(x)

        # 修改：掩码重构任务 - y 就是 x 本身
        y_tensor = x_tensor.clone()

        return x_tensor, y_tensor

def augment_dataset(dataset, config):
    augmented_data = []
    multiplier = config.AUGMENT_MULTIPLIER

    print(f"正在进行{multiplier}倍数据增强...")
    for i in tqdm(range(len(dataset)), desc="数据增强"):
        x, y = dataset[i]

        augmented_data.append((x, y))

        for j in range(multiplier - 1):
            aug_type = j % 3

            if aug_type == 0:
                x_aug = x + torch.randn_like(x) * config.AUGMENT_JITTER_SIGMA
            elif aug_type == 1:
                scale_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * config.AUGMENT_SCALE_SIGMA
                x_aug = x * scale_factor
            else:
                x_aug = x + torch.randn_like(x) * config.AUGMENT_JITTER_SIGMA * 0.5
                scale_factor = 1.0 + (torch.rand(1).item() - 0.5) * config.AUGMENT_SCALE_SIGMA
                x_aug = x_aug * scale_factor

            # 修改：y_aug 与 x_aug 保持一致
            y_aug = x_aug.clone()
            augmented_data.append((x_aug, y_aug))

    return augmented_data

class AugmentedDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]


# 辅助工具
def get_dataset_files(data_dir):
    dataset_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            dataset_files.append(file)
    return sorted(dataset_files)

def pre_train_read_dataset(data_dir, dataset_files):
    dfs = []

    for dataset_file in dataset_files:
        df = pd.read_csv(os.path.join(data_dir, dataset_file), encoding='utf-8', header=0)
        df = df[config.INDICATORS].ffill().bfill().fillna(df[config.INDICATORS].mean())
        data = df.to_numpy(dtype=np.float32)

        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)

        dfs.append(data)

    if len(dfs) == 0:
        raise ValueError("没有有效的站点数据!")

    df_merged = np.concatenate(dfs, axis=0)
    scaler_data = df_merged.reshape(-1)
    scaler_data = scaler_data[~np.isnan(scaler_data)]

    if len(scaler_data) == 0:
        raise ValueError("所有数据都是NaN!")

    scaler = StandardScaler(mean=scaler_data.mean(), std=scaler_data.std())

    for i in range(len(dfs)):
        dfs[i] = scaler.transform(dfs[i])

        if np.isnan(dfs[i]).any() or np.isinf(dfs[i]).any():
            print(f"⚠️  在 {dataset_files[i]} 发现 NaN/Inf, 替换为0")
            dfs[i] = np.nan_to_num(dfs[i], nan=0.0, posinf=0.0, neginf=0.0)

    return dfs, scaler

class EarlyStopping:
    def __init__(self, patience=20, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        return self.early_stop

class SnapshotEnsemble:
    def __init__(self, save_interval=50):
        self.save_interval = save_interval
        self.snapshots = []

    def save_snapshot(self, model, epoch):
        if (epoch + 1) % self.save_interval == 0:
            snapshot = copy.deepcopy(model.state_dict())
            self.snapshots.append(snapshot)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# 训练/验证
def train_epoch(model, teacher_model, train_loader, optimizer, scheduler,
                epoch, config, curriculum_masking, logger):
    model.train()
    epoch_loss_norm = 0      # 归一化空间损失（用于训练）
    epoch_loss_orig = 0      # 原始空间损失（用于监控对比）

    mask_ratio = curriculum_masking.get_mask_ratio(epoch)

    # 加权随机选择策略
    # 策略权重：简单策略多，困难策略少
    strategy_weights = [0.40, 0.25, 0.20, 0.15]  # random, parameter, station, temporal
    strategy_names = ['random', 'parameter', 'station', 'temporal']

    # 统计每种策略的使用次数（用于监控）
    strategy_counts = {name: 0 for name in strategy_names}

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS}')

    for batch_idx, (x, y) in enumerate(progress_bar):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # 从原始目标 y 计算统计量
        mean = y.mean(dim=1, keepdim=True)  # [B, 1, F, N]
        std = y.std(dim=1, keepdim=True) + 1e-5  # [B, 1, F, N]

        # 归一化
        x_norm = (x - mean) / std
        y_norm = (y - mean) / std

        # Mixup（在归一化空间）
        if torch.rand(1).item() < config.MIXUP_PROB:
            x_norm, y_norm = mixup_data(x_norm, y_norm, config.MIXUP_ALPHA)

        # 加权随机选择掩码策略
        strategy_idx = random.choices([0, 1, 2, 3], weights=strategy_weights, k=1)[0]
        masking_fn = [random_masking, parameter_masking, station_masking, temporal_masking][strategy_idx]
        masked_x_norm = masking_fn(x_norm, mask_ratio)

        # 统计策略使用
        strategy_counts[strategy_names[strategy_idx]] += 1

        # 模型输出归一化空间的重构
        recon_norm = model(masked_x_norm)

        # 在归一化空间计算损失（用于反向传播）
        loss_norm = F.mse_loss(recon_norm, y_norm)

        # 反归一化后计算原始空间损失（仅用于监控）
        with torch.no_grad():
            recon = recon_norm * std + mean
            loss_orig = F.mse_loss(recon, y)

        optimizer.zero_grad()
        loss_norm.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        epoch_loss_norm += loss_norm.item()
        epoch_loss_orig += loss_orig.item()

        progress_bar.set_postfix({
            'norm': f'{loss_norm.item():.4f}',
            'orig': f'{loss_orig.item():.4f}',
            'strategy': strategy_names[strategy_idx],
            'mask': f'{mask_ratio:.2f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })

    n_batches = len(train_loader)

    # 打印策略分布统计
    if epoch % 10 == 0 or epoch == 0:
        logger.info(f"Epoch {epoch+1} 策略分布: " +
                   ", ".join([f"{name}={count}({count/n_batches*100:.1f}%)"
                             for name, count in strategy_counts.items()]))

    return {
        'total': epoch_loss_norm / n_batches,      # 归一化空间（训练用）
        'recon': epoch_loss_norm / n_batches,      # 保持兼容性
        'orig': epoch_loss_orig / n_batches        # 原始空间（监控用）
    }

def validate(model, val_loader, config, mask_ratio=None):
    """
    验证函数
    1. 固定使用最终掩码比例（不随epoch变化）
    2. 加权随机选择掩码策略（与训练一致）
    3. 反归一化后在原始空间计算损失
    """


    model.eval()
    total_loss = 0

    # 使用固定的最终掩码比例（确保可比较）
    if mask_ratio is None:
        mask_ratio = config.CURRICULUM_END_MASK

    # 加权随机选择策略
    strategy_weights = [0.40, 0.25, 0.20, 0.15]  # random, parameter, station, temporal

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            # 从原始目标 y 计算统计量
            mean = y.mean(dim=1, keepdim=True)
            std = y.std(dim=1, keepdim=True) + 1e-5

            # 归一化
            x_norm = (x - mean) / std
            y_norm = (y - mean) / std

            # 加权随机选择掩码策略
            strategy_idx = random.choices([0, 1, 2, 3], weights=strategy_weights, k=1)[0]
            masking_fn = [random_masking, parameter_masking,
                          station_masking, temporal_masking][strategy_idx]
            masked_x_norm = masking_fn(x_norm, mask_ratio)

            # 模型输出归一化空间的重构
            recon_norm = model(masked_x_norm)

            # 反归一化后在原始空间计算损失
            recon = recon_norm * std + mean
            loss = F.mse_loss(recon, y)  # 原始空间比较，可与修改前的0.022对比
            total_loss += loss.item()

    return total_loss / len(val_loader)

# ===================================================================
# 主函数
# ===================================================================
def main():
    logger = setup_logger(config.MODEL_SAVE_PATH)

    print("\n" + "=" * 70)
    print(" 跨站点Transformer - 掩码重构任务 ".center(70, "="))
    print("=" * 70)
    print("\n核心创新:")
    print(" 站点在编码阶段就相互感知（而非独立）")
    print(" 多层跨站点注意力机制")
    print(" 时空融合解码")
    print(" 任务：掩码重构（Masked Reconstruction）\n")

    logger.info("=" * 70)
    logger.info("CrossStationTransformer 训练开始 - 掩码重构任务")
    logger.info("=" * 70)
    logger.info(f"时间维度: D_MODEL={config.D_MODEL}, N_HEADS={config.N_HEADS}")
    logger.info(f"空间维度: SPATIAL_D_MODEL={config.SPATIAL_D_MODEL}, SPATIAL_N_HEADS={config.SPATIAL_N_HEADS}")
    logger.info(f"空间层数: SPATIAL_LAYERS={config.SPATIAL_LAYERS}")

    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)

    dataset_files = get_dataset_files(config.DATA_DIR)
    print(f"✓ 找到 {len(dataset_files)} 个站点文件")
    logger.info(f"找到 {len(dataset_files)} 个站点文件")

    df_list, scaler = pre_train_read_dataset(config.DATA_DIR, dataset_files)
    num_stations = len(df_list)

    print(f"✓ 流域: {num_stations} 个站点")
    print(f"✓ 数据形状: {[df.shape for df in df_list]}")
    logger.info(f"流域: {num_stations} 个站点")

    scaler_path = os.path.join(config.MODEL_SAVE_PATH, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    train_dataset = PreTrainDataset(
        df_list,
        seq_len=config.SEQ_LEN,
        num_feat=config.INPUT_DIM,
        mode="train",
        train_split=config.TRAIN_SPLIT
    )
    test_dataset = PreTrainDataset(
        df_list,
        seq_len=config.SEQ_LEN,
        num_feat=config.INPUT_DIM,
        mode="test",
        train_split=config.TRAIN_SPLIT
    )

    print(f"✓ 基础数据集: {len(train_dataset)} 训练, {len(test_dataset)} 验证")
    logger.info(f"基础数据集: {len(train_dataset)} 训练, {len(test_dataset)} 验证")

    train_augmented = augment_dataset(train_dataset, config)
    test_augmented = augment_dataset(test_dataset, config)

    train_dataset_aug = AugmentedDataset(train_augmented)
    test_dataset_aug = AugmentedDataset(test_augmented)

    print(f"✓ 增强后数据集: {len(train_dataset_aug)} 训练, {len(test_dataset_aug)} 验证")
    logger.info(f"增强后数据集: {len(train_dataset_aug)} 训练, {len(test_dataset_aug)} 验证")

    train_loader = DataLoader(train_dataset_aug, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset_aug, batch_size=config.BATCH_SIZE, shuffle=False)

    for x, y in train_loader:
        print(f"✓ Batch形状: x={x.shape}, y={y.shape}")
        print(f"✓ 数据范围: x=[{x.min():.3f}, {x.max():.3f}]")
        break

    # 创建 CrossStationTransformer
    model = CrossStationTransformer(
        num_stations=num_stations,
        num_feat=config.INPUT_DIM,
        seq_len=config.SEQ_LEN,
        pred_len=config.PRED_LEN,
        d_model=config.D_MODEL,
        spatial_d_model=config.SPATIAL_D_MODEL,
        n_heads=config.N_HEADS,
        spatial_n_heads=config.SPATIAL_N_HEADS,
        spatial_layers=config.SPATIAL_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ CrossStationTransformer 模型创建成功")
    print(f"✓ 总参数量: {total_params:,}")
    logger.info(f"模型参数量: {total_params:,}")

    # 统计各模块参数
    temporal_params = sum(p.numel() for p in model.temporal_encoder.parameters())
    spatial_params = sum(p.numel() for p in model.cross_station_layers.parameters())
    decoder_params = sum(p.numel() for p in model.decoders.parameters())

    print(f"  - 时间编码器: {temporal_params:,}")
    print(f"  - 跨站点交互: {spatial_params:,}")
    print(f"  - 解码器: {decoder_params:,}")
    logger.info(f"时间编码器参数: {temporal_params:,}")
    logger.info(f"跨站点交互参数: {spatial_params:,}")
    logger.info(f"解码器参数: {decoder_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.BASE_LR,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    num_training_steps = len(train_loader) * config.EPOCHS
    num_warmup_steps = len(train_loader) * config.WARMUP_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    curriculum_masking = CurriculumMasking(
        config.CURRICULUM_START_MASK,
        config.CURRICULUM_END_MASK,
        config.CURRICULUM_WARMUP
    )
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    snapshot_ensemble = SnapshotEnsemble(save_interval=config.SNAPSHOT_INTERVAL)

    teacher_model = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon': [],
        'train_orig': []  # 原始空间损失（可与修改前对比）
    }

    best_val_loss = float('inf')

    print("\n" + "=" * 70)
    print(" 开始训练 ".center(70, "="))
    print("=" * 70 + "\n")

    for epoch in range(config.EPOCHS):
        train_losses = train_epoch(
            model, teacher_model, train_loader, optimizer, scheduler,
            epoch, config, curriculum_masking, logger
        )

        # 验证固定使用最终掩码比例
        # 训练使用课程掩码（0.2→0.5），但验证始终用最终难度（0.5）
        train_mask_ratio = curriculum_masking.get_mask_ratio(epoch)  # 用于记录
        val_loss = validate(model, val_loader, config)  # 不传mask_ratio，使用默认的CURRICULUM_END_MASK

        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_loss)
        history['train_recon'].append(train_losses['recon'])
        history['train_orig'].append(train_losses['orig'])

        gap = abs(train_losses['orig'] - val_loss)

        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print(f"  训练掩码比例: {train_mask_ratio:.2f}")
        print(f"  训练损失(归一化): {train_losses['total']:.6f}")
        print(f"  训练损失(原始): {train_losses['orig']:.6f}")
        print(f"  验证损失(原始): {val_loss:.6f} [固定mask={config.CURRICULUM_END_MASK}]")
        print(f"  训练-验证差距: {gap:.6f}")

        logger.info(f"Epoch {epoch+1} - TrainMask: {train_mask_ratio:.2f}, Train(norm): {train_losses['total']:.6f}, Train(orig): {train_losses['orig']:.6f}, Val: {val_loss:.6f}, Gap: {gap:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✅ 保存最佳模型")
            logger.info(f"  保存最佳模型 - 验证损失: {best_val_loss:.6f}")

        snapshot_ensemble.save_snapshot(model, epoch)

        if early_stopping(val_loss, model):
            print(f"\n⚠️  早停触发! (Epoch {epoch+1})")
            print(f"  最佳验证损失: {early_stopping.best_loss:.6f}")
            logger.info(f"早停触发 - Epoch {epoch+1}, 最佳验证损失: {early_stopping.best_loss:.6f}")
            model.load_state_dict(early_stopping.best_model)
            break

    history_path = os.path.join(config.MODEL_SAVE_PATH, "training_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)

    print("\n" + "=" * 70)
    print(" 训练完成 ".center(70, "="))
    print("=" * 70)
    print(f"\n最终统计:")
    print(f"  - 训练轮数: {len(history['train_loss'])}")
    print(f"  - 最佳验证损失: {best_val_loss:.6f}")
    print(f"\n模型保存在: {config.MODEL_SAVE_PATH}")

    logger.info("=" * 70)
    logger.info("训练完成")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")
    logger.info("=" * 70)

    config_dict = {
        'SEQ_LEN': config.SEQ_LEN,
        'PRED_LEN': config.PRED_LEN,
        'D_MODEL': config.D_MODEL,
        'SPATIAL_D_MODEL': config.SPATIAL_D_MODEL,
        'N_HEADS': config.N_HEADS,
        'SPATIAL_N_HEADS': config.SPATIAL_N_HEADS,
        'SPATIAL_LAYERS': config.SPATIAL_LAYERS,
        'num_stations': num_stations,
    }
    config_path = os.path.join(config.MODEL_SAVE_PATH, "config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(config_dict, f)

if __name__ == '__main__':
    main()
