import os
import time
import pickle
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from math import sqrt
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "results", "base")
PRETRAIN_RUNS_DIR = os.path.join(ARTIFACTS_ROOT, "pretrain", "runs")

# ==========================================
# 配置
# ==========================================
class PretrainConfig:
    def __init__(self):
        # 路径配置
        self.water_data_dir = os.path.join(REPO_ROOT, "data", "data_cleaned", "yangzte")
        self.save_dir = PRETRAIN_RUNS_DIR
        self.river_name = "长江"

        # 设备配置
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple MPS acceleration.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            print("Using CPU.")

        # 模型超参数
        self.n_in = 8
        self.n_out = 8
        self.input_dim = 4
        self.hidden_size = 256
        self.num_heads = 8
        self.e_layer = 3

        self.batch_size = 32
        self.pretrain_epochs = 50

        self.base_lr = 0.009737183935075031
        self.epsilon = 1e-8
        self.train_ratio = 0.8
        self.mask_ratio = 0.7  # 对齐论文参数

        self.lr_milestones = [40, 60, 80, 120, 160, 200, 240, 280]
        self.lr_decay_ratio = 0.5
        self.max_grad_norm = 1.0

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

# ==========================================
# 工具函数
# ==========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class StandardScaler:
    """标准化器 - 对齐仓库实现"""
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

# ==========================================
# 掩码策略 - 对齐仓库的4种掩码
# ==========================================
def random_masking(data, mask_ratio):
    """随机掩码"""
    mask = torch.rand(data.shape) >= mask_ratio
    masked_data = data.clone()
    masked_data[~mask] = 0
    return masked_data

def parameter_masking(data, mask_ratio):
    """参数维度掩码"""
    masked_data = data.clone()
    batch_size, seq_len, num_stations, num_feat = data.shape
    for i in range(batch_size):
        for station_idx in range(num_stations):
            for param_idx in range(num_feat):
                if torch.rand(1).item() < mask_ratio:
                    masked_data[i, :, station_idx, param_idx] = 0
    return masked_data

def station_masking(data, mask_ratio):
    """站点维度掩码"""
    masked_data = data.clone()
    batch_size, seq_len, num_stations, num_feat = data.shape
    for i in range(batch_size):
        for station_idx in range(num_stations):
            if torch.rand(1).item() < mask_ratio:
                masked_data[i, :, station_idx, :] = 0
    return masked_data

def temporal_masking(data, mask_ratio):
    """时间维度掩码"""
    masked_data = data.clone()
    batch_size, seq_len, num_stations, num_feat = data.shape
    for i in range(batch_size):
        for time_step in range(seq_len):
            if torch.rand(1).item() < mask_ratio:
                masked_data[i, time_step, :, :] = 0
    return masked_data

# ==========================================
# 数据集 - 对齐仓库的5倍数据增强
# ==========================================
class PretrainMaskedDataset(Dataset):
    """预训练数据集 - 使用4种掩码策略进行5倍数据增强"""
    def __init__(self, data, seq_len=8, mask_ratio=0.7, mode="train", train_split=0.8):
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio

        # data shape: (Time, Stations, Features)
        dataset_len = len(data)
        train_len = int(train_split * dataset_len)

        if mode == "train":
            self.data = data[:train_len]
        else:
            self.data = data[train_len:]

        self.data = torch.FloatTensor(self.data)

        # 计算有效样本数
        self.num_samples = len(self.data) - self.seq_len + 1

    def __getitem__(self, index):
        # 确定是原始样本还是增强样本
        # index 范围: [0, num_samples * 5)
        original_idx = index % self.num_samples
        mask_type = index // self.num_samples  # 0:原始, 1-4:四种掩码

        s_begin = original_idx
        s_end = s_begin + self.seq_len

        # 取出原始序列 (Seq, Stations, Feat)
        x_origin = self.data[s_begin:s_end]

        # 调整形状为 (1, Seq, Stations, Feat) 以适配掩码函数
        x_origin = x_origin.unsqueeze(0)

        # 根据mask_type应用不同掩码
        if mask_type == 0:
            # 原始数据，不掩码
            x_masked = x_origin
        elif mask_type == 1:
            x_masked = random_masking(x_origin, self.mask_ratio)
        elif mask_type == 2:
            x_masked = parameter_masking(x_origin, self.mask_ratio)
        elif mask_type == 3:
            x_masked = station_masking(x_origin, self.mask_ratio)
        else:  # mask_type == 4
            x_masked = temporal_masking(x_origin, self.mask_ratio)

        # 移除batch维度
        x_masked = x_masked.squeeze(0)
        y = x_origin.squeeze(0)

        return x_masked, y

    def __len__(self):
        # 5倍数据增强：原始 + 4种掩码
        return self.num_samples * 5

def load_pretrain_data(config, river_name):
    """加载预训练数据 - 对齐仓库的归一化方式"""
    dataset_files = sorted([f for f in os.listdir(config.water_data_dir)
                           if f.endswith('.csv') and (river_name in f)])

    if not dataset_files:
        print("没有找到预训练水质文件。")
        return None, None, None, None

    dfs_w = []
    for w_file in dataset_files:
        w_path = os.path.join(config.water_data_dir, w_file)
        df_w = pd.read_csv(w_path, header=0)
        if {'CODMn', 'DO', 'NH4N', 'pH'}.issubset(df_w.columns):
            data_w = df_w[['CODMn', 'DO', 'NH4N', 'pH']].values.astype(np.float32)
            dfs_w.append(data_w)
        else:
            print(f"警告: {w_file} 缺少必要的列，跳过。")

    if not dfs_w:
        return None, None, None, None

    # 对齐时间长度
    min_len = min(len(d) for d in dfs_w)
    dfs_w = [d[:min_len] for d in dfs_w]

    # Stack -> (Time, Stations, Features)
    stacked = np.stack(dfs_w, axis=1)

    # 【关键修改】对齐仓库：reshape成1D进行全局归一化
    scaler_data = stacked.reshape(-1)
    scaler = StandardScaler(mean=scaler_data.mean(), std=scaler_data.std())

    # 归一化整个数据
    stacked_normalized = scaler.transform(stacked)

    train_dataset = PretrainMaskedDataset(
        stacked_normalized, config.n_in, config.mask_ratio, "train", config.train_ratio
    )
    test_dataset = PretrainMaskedDataset(
        stacked_normalized, config.n_in, config.mask_ratio, "test", config.train_ratio
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader, scaler, dataset_files

# ==========================================
# 模型定义
# ==========================================
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(seq_len, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-1)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, L, H, -1)
        values = self.value_projection(values).view(B, L, H, -1)
        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class Transformer_Layer(nn.Module):
    def __init__(self, num_heads, e_layer, hidden_size, num_feat, seq_len, pred_len):
        super(Transformer_Layer, self).__init__()
        self.d_model = hidden_size
        self.embedding = TimeFeatureEmbedding(self.d_model, seq_len)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(attention_dropout=0.1), self.d_model, num_heads),
                    self.d_model, 2048, dropout=0.1
                ) for _ in range(e_layer)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.head = FlattenHead(num_feat, self.d_model, pred_len, head_dropout=0.1)
        self.pred_len = pred_len

    def forward(self, x_enc):
        # x_enc shape: (Batch, Seq, Feat)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.embedding(x_enc)
        enc_out = self.encoder(enc_out)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

class MultiTransformer(nn.Module):
    def __init__(self, num_heads, e_layer, hidden_size, num_stations, num_feat, seq_len, pred_len):
        super(MultiTransformer, self).__init__()
        self.transformers = nn.ModuleList([
            Transformer_Layer(num_heads, e_layer, hidden_size, num_feat, seq_len, pred_len)
            for _ in range(num_stations)
        ])
        self.mlp = nn.Linear(num_stations, num_stations)

    def forward(self, x):
        # x shape: (Batch, Seq, Num_Stations, Feat)
        outputs = []
        for i in range(len(self.transformers)):
            channel_data = x[:, :, i, :]
            channel_out = self.transformers[i](channel_data)
            outputs.append(channel_out)

        out = torch.stack(outputs, dim=2)
        out = out.permute(0, 1, 3, 2)
        out = self.mlp(out)
        out = out.permute(0, 1, 3, 2)

        return out

# ==========================================
# 训练函数
# ==========================================
def pre_train_model(model, train_loader, val_loader, config):
    model = model.to(config.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.base_lr,
        eps=config.epsilon,
        weight_decay=0.0,
        amsgrad=True
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_ratio
    )
    criterion = nn.MSELoss()

    history = []
    best_val_loss = float('inf')

    for epoch in range(config.pretrain_epochs):
        model.train()
        total_loss = 0.0

        for data, target in train_loader:
            data = data.to(config.device)
            target = target.to(config.device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        # 验证
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(config.device)
                target = target.to(config.device)
                output = model(data)
                val_loss += criterion(output, target).item()

        val_loss = val_loss / len(val_loader)

        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": val_loss})

        if (epoch + 1) % 10 == 0:
            print(f"[Pretrain] Epoch {epoch+1}/{config.pretrain_epochs} | Loss: {avg_train_loss:.5f} | Val: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return model, history, best_val_loss

# ==========================================
# 主函数
# ==========================================
def main(seed=42):
    set_seed(seed)

    config = PretrainConfig()

    print(f"\n{'='*60}")
    print(f"预训练开始 - 种子: {seed}")
    print(f"{'='*60}")
    print(f"超参数: hidden={config.hidden_size}, heads={config.num_heads}, mask_ratio={config.mask_ratio}")
    print(f"数据增强: 5倍扩展（原始 + 4种掩码策略）")

    print("\n>>> 加载预训练数据...")
    pt_loader, pt_val_loader, scaler, pt_files = load_pretrain_data(config, config.river_name)

    if pt_loader is None:
        raise ValueError("预训练数据加载失败")

    print(f">>> 找到 {len(pt_files)} 个站点")
    print(f">>> 训练样本: {len(pt_loader.dataset)} (包含5倍增强)")

    print("\n>>> 创建模型...")
    model = MultiTransformer(
        num_heads=config.num_heads,
        e_layer=config.e_layer,
        hidden_size=config.hidden_size,
        num_stations=len(pt_files),
        num_feat=config.input_dim,
        seq_len=config.n_in,
        pred_len=config.n_out
    )

    print("\n>>> 开始训练...")
    train_start = time.time()
    model, history, best_val_loss = pre_train_model(model, pt_loader, pt_val_loader, config)
    train_end = time.time()

    # 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"pretrain_{config.river_name}_seed{seed}_{timestamp}"
    save_dir = os.path.join(config.save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    config_dict = vars(config)
    config_dict['station_files'] = pt_files
    config_dict['num_stations'] = len(pt_files)
    config_dict['device'] = str(config.device)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "history.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"预训练完成 | 最佳Val Loss: {best_val_loss:.6f}")
    print(f"训练耗时: {train_end - train_start:.2f}秒")
    print(f"模型已保存至: {save_dir}")

    return best_val_loss, save_dir

if __name__ == '__main__':
    main(seed=42)
