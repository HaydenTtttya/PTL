import os
import re
import time
import math
import copy
import datetime
import argparse
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "results", "base")
PRETRAIN_RUNS_DIR = os.path.join(ARTIFACTS_ROOT, "pretrain", "runs")
FINETUNE_RUNS_DIR = os.path.join(ARTIFACTS_ROOT, "finetune", "runs")
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}


def should_save_finetune_model_weights(default=False):
    value = os.environ.get("BASE_FINETUNE_SAVE_MODEL_WEIGHTS")
    if value is None:
        return default
    return value.strip().lower() in TRUTHY_ENV_VALUES

# ==========================================
# 1. 配置部分 (Configuration)
# ==========================================

class Config:
    def __init__(self):
        # --- 路径配置 (请确保路径正确) ---
        self.water_data_dir = os.path.join(REPO_ROOT, "data", "data_cleaned", "yangzte")
        self.meteo_data_dir = os.path.join(REPO_ROOT, "data", "meteorology_weekly")
        self.save_dir = PRETRAIN_RUNS_DIR

        # 微调的目标文件
        self.target_water_file = os.path.join(REPO_ROOT, "data", "data_cleaned", "zhu", "珠江136.csv")
        self.river_name = "长江" # 用于预训练筛选文件
        self.target_station_id = "136" # 如果没从文件名读出来，则用这个

        self.seeds = [42]
        self.results_dir = FINETUNE_RUNS_DIR
        self.save_finetune_model_weights = should_save_finetune_model_weights()

        # --- 设备配置 (适配 M1 Mac) ---
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple MPS (Metal Performance Shaders) acceleration.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            print("Using CPU.")

        # --- 模型超参数 ---
        self.model_name = 'Prompt_Transformer'
        self.n_in = 8          # 输入序列长度
        self.n_out = 1         # 预测序列长度
        self.input_dim = 4     # 水质特征数量 (CODMn, DO, NH4N, pH)
        self.feature_dim = 7   # 气象特征数量
        self.hidden_size = 128
        self.num_heads = 8
        self.e_layer = 3
        self.batch_size = 8
        self.epochs = 100       # 微调 Epoch
        self.pretrain_epochs = 300 # 预训练 Epoch
        self.base_lr = 0.00034738360143117493
        self.epsilon = 1e-8
        self.weight_decay = 0.0003066732568374593
        self.train_ratio = 0.8
        self.prompt_num = 1
        self.mask_ratio = 0.5
        self.lr_milestones = [40, 60, 80]
        self.lr_decay_ratio = 0.5
        self.max_grad_norm = 1.0
        self.freeze_ratio = 0.4961191262276459

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

config = Config()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def _flatten_preds_targets(preds: np.ndarray, targets: np.ndarray):
    preds_2d = preds.reshape(-1, preds.shape[-1])
    targets_2d = targets.reshape(-1, targets.shape[-1])
    return preds_2d, targets_2d

def compute_per_feature_metrics(preds: np.ndarray, targets: np.ndarray, feature_names):
    """
    按论文公式计算指标
    M: 特征数量, N: 样本数量
    """
    preds_2d, targets_2d = _flatten_preds_targets(preds, targets)
    M = preds_2d.shape[1]  # 特征数
    N = preds_2d.shape[0]  # 样本数

    out = {}

    # 每个特征单独计算
    for j, name in enumerate(feature_names):
        y_pred = preds_2d[:, j]
        y_true = targets_2d[:, j]

        # MAE: 平均绝对误差
        mae = float(np.mean(np.abs(y_true - y_pred)))

        # RMSE: 均方根误差
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # NSE: Nash-Sutcliffe效率系数
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        if denominator == 0.0:
            nse = float("nan")
        else:
            nse = float(1.0 - (numerator / denominator))

        # MAPE: 平均绝对百分比误差
        # 避免除零
        mask = y_true != 0
        if np.any(mask):
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = float("nan")

        out[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "NSE": nse,
            "MAPE": mape
        }

    # 总体指标（按论文公式2.21-2.25）
    # MSE (2.21)
    overall_mse = float(np.mean((preds_2d - targets_2d) ** 2))

    # MAE (2.24)
    overall_mae = float(np.mean(np.abs(preds_2d - targets_2d)))

    # RMSE (2.22): 先对每个特征计算RMSE，再平均
    rmse_per_feature = []
    for j in range(M):
        rmse_j = np.sqrt(np.mean((preds_2d[:, j] - targets_2d[:, j]) ** 2))
        rmse_per_feature.append(rmse_j)
    overall_rmse = float(np.mean(rmse_per_feature))

    # NSE (2.23): 先对每个特征计算NSE，再平均
    nse_per_feature = []
    for j in range(M):
        y_true = targets_2d[:, j]
        y_pred = preds_2d[:, j]
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        if denominator != 0:
            nse_j = 1.0 - (numerator / denominator)
            nse_per_feature.append(nse_j)
    overall_nse = float(np.mean(nse_per_feature)) if nse_per_feature else float("nan")

    # MAPE (2.25)
    mask = targets_2d != 0
    if np.any(mask):
        overall_mape = float(np.mean(np.abs((targets_2d[mask] - preds_2d[mask]) / targets_2d[mask])) * 100)
    else:
        overall_mape = float("nan")

    out["__overall__"] = {
        "MSE": overall_mse,
        "MAE": overall_mae,
        "RMSE": overall_rmse,
        "NSE": overall_nse,
        "MAPE": overall_mape
    }

    return out

def _jsonify(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return str(obj)

# ==========================================
# 2. 工具与数据处理 (Utils & Data Process)
# ==========================================

class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean(axis=0)  # 修复：按特征计算
        self.std = data.std(axis=0)    # 修复：按特征计算

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

# --- 微调用的 Dataset (返回水质和气象) ---
class WaterMeteoDataset(Dataset):
    """微调用数据集，同时返回水质和气象数据"""
    def __init__(self, w_data, m_data, seq_len=6, pred_len=1, mode="train", train_split=0.8):
        self.seq_len = seq_len
        self.pred_len = pred_len

        dataset_len = len(w_data)
        train_len = int(train_split * dataset_len)

        if mode == "train":
            self.w_data = w_data[:train_len]
            self.m_data = m_data[:train_len]
        else:
            self.w_data = w_data[train_len:]
            self.m_data = m_data[train_len:]

        self.w_data = torch.FloatTensor(self.w_data)
        self.m_data = torch.FloatTensor(self.m_data)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        w_seq_x = self.w_data[s_begin:s_end]
        m_seq_x = self.m_data[s_begin:s_end]
        w_seq_y = self.w_data[r_begin:r_end]

        return w_seq_x, m_seq_x, w_seq_y

    def __len__(self):
        return len(self.w_data) - self.seq_len - self.pred_len + 1

# --- 预训练用的 Dataset (Masked，只返回水质) ---
class PretrainMaskedDataset(Dataset):
    def __init__(self, data, seq_len=6, pred_len=1, mask_ratio=0.5, mode="train", train_split=0.8):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mask_ratio = mask_ratio

        dataset_len = len(data)
        train_len = int(train_split * dataset_len)
        if mode == "train":
            self.data = data[:train_len]
        else:
            self.data = data[train_len:]

        self.data = torch.FloatTensor(self.data)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        x = self.data[s_begin:s_end]
        y = self.data[r_begin:r_end]

        mask = torch.rand_like(x) > self.mask_ratio
        x_masked = x * mask
        return x_masked, y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

def get_dataset_files(dataset_dir, river_name):
    return sorted([f for f in os.listdir(dataset_dir) if f.endswith('.csv') and (river_name in f)])

def load_pretrain_data(config, river_name, pred_len):
    dataset_files = get_dataset_files(config.water_data_dir, river_name)
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

    min_len = min(len(d) for d in dfs_w)
    dfs_w = [d[:min_len] for d in dfs_w]
    stacked = np.stack(dfs_w, axis=-1)  # [T, feat, stations]

    scaler = StandardScaler()
    scaler.fit(stacked)
    stacked = scaler.transform(stacked)

    train_dataset = PretrainMaskedDataset(stacked, config.n_in, pred_len, config.mask_ratio, "train", config.train_ratio)
    test_dataset = PretrainMaskedDataset(stacked, config.n_in, pred_len, config.mask_ratio, "test", config.train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader, scaler, dataset_files


def load_finetune_data(config, river_name, target_station_id):
    """加载微调数据，返回水质+气象的loader"""
    if config.target_water_file:
        water_filename = os.path.basename(config.target_water_file)
        w_path = config.target_water_file
        match = re.search(r"(\d+)", water_filename)
        station_id = match.group(1) if match else target_station_id
    else:
        station_id = target_station_id
        w_path = os.path.join(config.water_data_dir, f"{river_name}{station_id}.csv")

    meteo_file = f"Meteorology_{station_id}.0.csv"
    m_path = os.path.join(config.meteo_data_dir, meteo_file)

    if not os.path.exists(w_path) or not os.path.exists(m_path):
        print(f"找不到文件: \n{w_path} \n{m_path}")
        return None, None, None

    df_w = pd.read_csv(w_path, header=0)
    df_m = pd.read_csv(m_path, header=0)

    if not {'CODMn', 'DO', 'NH4N', 'pH'}.issubset(df_w.columns):
        print("水质文件缺少列")
        return None, None, None

    m_cols = ['lrad', 'temp', 'pres', 'shum', 'wind', 'srad', 'prec']
    if not set(m_cols).issubset(df_m.columns):
        print("气象文件缺少列")
        return None, None, None

    data_w = df_w[['CODMn', 'DO', 'NH4N', 'pH']].values.astype(np.float32)
    data_m = df_m[m_cols].values.astype(np.float32)

    min_len = min(len(data_w), len(data_m))
    data_w = data_w[:min_len]
    data_m = data_m[:min_len]

    scaler_w = StandardScaler()
    scaler_w.fit(data_w)
    data_w = scaler_w.transform(data_w)

    scaler_m = StandardScaler()
    scaler_m.fit(data_m)
    data_m = scaler_m.transform(data_m)

    train_dataset = WaterMeteoDataset(data_w, data_m, config.n_in, config.n_out, "train", config.train_ratio)
    test_dataset = WaterMeteoDataset(data_w, data_m, config.n_in, config.n_out, "test", config.train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader, scaler_w

# ==========================================
# 3. 模型定义 (Model Architecture)
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

# --- Memory & Prompt Modules ---

class Memory(nn.Module):
    def __init__(self, num_memory, memory_dim):
        super().__init__()
        self.memMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))
        self.keyMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))
        self.x_proj = nn.Linear(memory_dim, memory_dim)
        torch.nn.init.xavier_uniform_(self.memMatrix)
        torch.nn.init.xavier_uniform_(self.keyMatrix)

    def forward(self, x):
        x_query = torch.tanh(self.x_proj(x))
        att_weight = F.linear(input=x_query, weight=self.keyMatrix)
        att_weight = F.softmax(att_weight, dim=-1)
        out = F.linear(att_weight, self.memMatrix.permute(1, 0))
        return out

class MeteorologyAttention(nn.Module):
    def __init__(self, num_memory, memory_dim, nhead):
        super(MeteorologyAttention, self).__init__()
        self.memory = Memory(num_memory, memory_dim)
        # MPS兼容：确保nhead合理
        actual_nhead = max(1, min(nhead, memory_dim))
        while memory_dim % actual_nhead != 0 and actual_nhead > 1:
            actual_nhead -= 1

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=memory_dim,
            nhead=actual_nhead,
            dim_feedforward=memory_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)

    def forward(self, water_data, weather_data):
        combined = torch.cat((water_data, weather_data), dim=-1)
        modulated = self.transformer_encoder(combined)
        ma_output = self.memory(modulated)
        return ma_output

class CombinedAttentionConcat(nn.Module):
    def __init__(self, input_dim_total, w_feat, seq_len, pred_len):
        super(CombinedAttentionConcat, self).__init__()
        # MPS兼容调整
        nhead = min(input_dim_total, 4)
        while input_dim_total % nhead != 0 and nhead > 1:
            nhead -= 1

        self.meteorology_attention = MeteorologyAttention(
            num_memory=input_dim_total * 2,
            memory_dim=input_dim_total,
            nhead=nhead
        )
        self.mlp = nn.Sequential(nn.Linear(input_dim_total, 64), nn.ReLU(), nn.Linear(64, w_feat))
        self.fc = nn.Linear(seq_len, pred_len)

    def forward(self, water_data, weather_data):
        met_out = self.meteorology_attention(water_data, weather_data)
        combined = self.mlp(met_out)
        combined = combined.permute(0, 2, 1)
        combined = self.fc(combined)
        return combined.permute(0, 2, 1)

# --- 预训练模型 (多站点) ---
class MultiTransformer(nn.Module):
    def __init__(self, num_heads, e_layer, hidden_size, num_stations, num_feat, seq_len, pred_len):
        super(MultiTransformer, self).__init__()
        self.transformers = nn.ModuleList([
            Transformer_Layer(num_heads, e_layer, hidden_size, num_feat, seq_len, pred_len)
            for _ in range(num_stations)
        ])
        self.mlp = nn.Linear(num_stations, num_stations)

    def forward(self, x):
        outputs = []
        for i in range(len(self.transformers)):
            channel_data = x[..., i]
            channel_out = self.transformers[i](channel_data)
            outputs.append(channel_out)
        out = torch.stack(outputs, dim=-1)
        out = self.mlp(out)
        return out

# --- 微调模型 (继承预训练结构 + Prompt) ---
class Prompt_MultiTransformer(nn.Module):
    """
    微调模型：继承预训练的多站点Transformer结构
    学习consolidated的方法：保持结构一致性以便权重迁移
    """
    def __init__(self, config, num_pretrain_stations):
        super(Prompt_MultiTransformer, self).__init__()

        # 1. 保留预训练的transformers结构（用于加载权重）
        self.transformers = nn.ModuleList([
            Transformer_Layer(
                num_heads=config.num_heads,
                e_layer=config.e_layer,
                hidden_size=config.hidden_size,
                num_feat=config.input_dim,
                seq_len=config.n_in,
                pred_len=config.n_out
            ) for _ in range(num_pretrain_stations)
        ])

        # 2. MLP层（映射多站点到单站点输出）
        self.mlp = nn.Linear(num_pretrain_stations, config.prompt_num)

        # 3. Prompt模块（融合气象信息）
        self.combined_attention = CombinedAttentionConcat(
            input_dim_total=config.input_dim + config.feature_dim,
            w_feat=config.input_dim,
            seq_len=config.n_in,
            pred_len=config.n_out
        )

        # 4. 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.prompt_num * 2, 32),
            nn.ReLU(),
            nn.Linear(32, config.prompt_num)
        )

    def forward(self, x_water, x_weather):
        # 主分支：使用所有预训练transformers的集成
        outputs = []
        for transformer in self.transformers:
            out = transformer(x_water)
            outputs.append(out)

        # [B, T, F, num_stations] -> [B, T, F, 1]
        out_main = torch.stack(outputs, dim=-1)
        out_main = self.mlp(out_main)

        # Prompt分支：融合气象信息
        out_prompt = self.combined_attention(x_water, x_weather).unsqueeze(-1)

        # 融合两个分支
        combined = torch.cat([out_main, out_prompt], dim=-1)
        final = self.fusion_layer(combined).squeeze(-1)

        return final

# ==========================================
# 4. 训练与评估函数 (Train & Eval)
# ==========================================

def pre_train_model(model, train_loader, val_loader, config):
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr, eps=config.epsilon)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_ratio)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(config.pretrain_epochs):
        model.train()
        total_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.device), target.to(config.device)
                output = model(data)
                val_loss += criterion(output, target).item()
        val_loss = val_loss / len(val_loader)
        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": val_loss})
        if (epoch + 1) % 5 == 0:
            print(f"[Pretrain] Epoch {epoch+1}/{config.pretrain_epochs} | Loss: {avg_train_loss:.5f} | Val: {val_loss:.5f}")

    return model, history


def evaluate_finetune(model, loader, device, scaler):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for w_x, m_x, w_y in loader:
            w_x, m_x, w_y = w_x.to(device), m_x.to(device), w_y.to(device)
            output = model(w_x, m_x)
            loss = criterion(output, w_y)
            total_loss += loss.item()

            pred_np = output.cpu().numpy()
            target_np = w_y.cpu().numpy()

            if scaler:
                pred_np = scaler.inverse_transform(pred_np)
                target_np = scaler.inverse_transform(target_np)

            all_preds.append(pred_np)
            all_targets.append(target_np)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_targets


def fine_tune_model(model, train_loader, val_loader, config):
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr, eps=config.epsilon, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_ratio)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0

        for w_x, m_x, w_y in train_loader:
            w_x, m_x, w_y = w_x.to(config.device), m_x.to(config.device), w_y.to(config.device)

            optimizer.zero_grad()
            output = model(w_x, m_x)
            loss = criterion(output, w_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        val_loss, _, _ = evaluate_finetune(model, val_loader, config.device, None)

        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": val_loss})
        if (epoch + 1) % 5 == 0:
            print(f"[Finetune] Epoch {epoch+1}/{config.epochs} | Loss: {avg_train_loss:.5f} | Val: {val_loss:.5f}")

    return model, history

# ==========================================
# 5. 主程序 (Main)
# ==========================================

if __name__ == '__main__':
    feature_names = ['CODMn', 'DO', 'NH4N', 'pH']

    for seed in config.seeds:
        set_seed(int(seed))
        run_start_dt = datetime.datetime.now()
        run_id = f"{config.model_name}_{run_start_dt.strftime('%Y%m%d_%H%M%S')}_seed{seed}"

        run_dir = os.path.join(config.results_dir, run_id)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # -----------------------------------------------------
        # 1. 预训练阶段 (Pre-training)
        # -----------------------------------------------------
        print(">>> [Stage 1] 加载预训练数据...")
        pt_loader, pt_val_loader, scaler_pre, pt_files = load_pretrain_data(config, config.river_name, config.n_in)

        if pt_loader is None:
            print("预训练数据加载失败。")
            continue

        print(f">>> [Stage 1] 开始预训练 ({len(pt_files)}个站点)...")
        pretrain_model = MultiTransformer(
            num_heads=config.num_heads,
            e_layer=config.e_layer,
            hidden_size=config.hidden_size,
            num_stations=len(pt_files),
            num_feat=config.input_dim,
            seq_len=config.n_in,
            pred_len=config.n_in
        )

        pretrain_model, _ = pre_train_model(pretrain_model, pt_loader, pt_val_loader, config)

        pretrain_ckpt = os.path.join(config.save_dir, f"pretrain_{config.river_name}.pth")
        torch.save(pretrain_model.state_dict(), pretrain_ckpt)
        print(f">>> 预训练完成，权重已保存: {pretrain_ckpt}")

        # -----------------------------------------------------
        # 2. 微调阶段 (Fine-tuning)
        # -----------------------------------------------------
        print(">>> [Stage 2] 加载微调数据 (Target Station)...")
        ft_train_loader, ft_test_loader, scaler_ft = load_finetune_data(
            config, config.river_name, config.target_station_id
        )

        if ft_train_loader is None:
            print("微调数据加载失败。")
            continue

        print(">>> [Stage 2] 初始化微调模型并加载迁移权重...")
        # 使用新的模型结构，传入预训练站点数量
        ft_model = Prompt_MultiTransformer(config, num_pretrain_stations=len(pt_files))

        # --- 学习consolidated的权重加载方法 ---
        print(">>> 加载预训练权重...")
        pretrained_dict = torch.load(pretrain_ckpt, map_location=config.device)
        ft_model_dict = ft_model.state_dict()

        # 直接匹配键和形状，自动过滤不匹配的
        matched_dict = {k: v for k, v in pretrained_dict.items()
                       if k in ft_model_dict and v.shape == ft_model_dict[k].shape}

        ft_model_dict.update(matched_dict)
        ft_model.load_state_dict(ft_model_dict, strict=False)

        print(f">>> 成功加载 {len(matched_dict)} 层权重到微调模型")

        # # --- 可选：冻结预训练的transformers参数 ---
        # frozen_layers = ['mlp.weight', 'mlp.bias']
        # frozen_count = 0
        # for name, param in ft_model.named_parameters():
        #     if name.startswith('transformers') and name not in frozen_layers:
        #         param.requires_grad = False
        #         frozen_count += 1
        # print(f">>> 冻结了 {frozen_count} 个transformers参数，只训练Prompt模块")


        # --- 按比例冻结预训练的transformers参数 ---
        frozen_layers = ['mlp.weight', 'mlp.bias']
        frozen_count = 0

        # 获取所有transformers相关参数
        transformer_params = [(name, param) for name, param in ft_model.named_parameters()
                            if name.startswith('transformers') and name not in frozen_layers]

        # 计算需要冻结的参数数量
        num_to_freeze = int(len(transformer_params) * config.freeze_ratio)

        # 冻结前 num_to_freeze 个参数
        for i, (name, param) in enumerate(transformer_params):
            if i < num_to_freeze:
                param.requires_grad = False
                frozen_count += 1

        print(f">>> 冻结了 {frozen_count}/{len(transformer_params)} 个transformer参数 (比例: {config.freeze_ratio})")

        print(">>> [Stage 2] 开始微调训练...")
        train_start = time.time()
        ft_model, history = fine_tune_model(ft_model, ft_train_loader, ft_test_loader, config)
        train_end = time.time()

        # -----------------------------------------------------
        # 3. 保存与输出
        # -----------------------------------------------------
        save_path = os.path.join(run_dir, f"finetune_final.pth")
        finetune_weights_saved = bool(config.save_finetune_model_weights)
        if finetune_weights_saved:
            torch.save(ft_model.state_dict(), save_path)
        else:
            print(f">>> 跳过保存微调模型权重: {save_path}")

        test_loss, preds, targets = evaluate_finetune(ft_model, ft_test_loader, config.device, scaler_ft)

        # 保存CSV
        preds_flat = preds.squeeze(1)
        targets_flat = targets.squeeze(1)
        df_res = pd.DataFrame(preds_flat, columns=[f"Pred_{n}" for n in feature_names])
        df_tgt = pd.DataFrame(targets_flat, columns=[f"True_{n}" for n in feature_names])
        pd.concat([df_tgt, df_res], axis=1).to_csv(os.path.join(run_dir, "prediction_results.csv"), index=False)

        # 保存Metrics
        metrics = compute_per_feature_metrics(preds, targets, feature_names)
        pd.DataFrame.from_dict(metrics, orient="index").to_csv(os.path.join(run_dir, "metrics.csv"))

        # 保存Meta
        meta = {
            "run_id": run_id,
            "seed": seed,
            "device": str(config.device),

            # 模型结构参数
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "e_layer": config.e_layer,
            "input_dim": config.input_dim,
            "feature_dim": config.feature_dim,
            "prompt_num": config.prompt_num,

            # 时间序列参数
            "n_in": config.n_in,
            "n_out": config.n_out,

            # 训练参数
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "pretrain_epochs": config.pretrain_epochs,
            "base_lr": config.base_lr,
            "weight_decay": config.weight_decay,
            "epsilon": config.epsilon,
            "max_grad_norm": config.max_grad_norm,

            # 数据参数
            "train_ratio": config.train_ratio,
            "mask_ratio": config.mask_ratio,
            "lr_milestones": config.lr_milestones,
            "lr_decay_ratio": config.lr_decay_ratio,

            # 运行结果
            "duration": train_end - train_start,
            "test_loss": test_loss,
            "transferred_layers": len(matched_dict),
            "frozen_params": frozen_count,
            "pretrain_stations": len(pt_files),
            "pretrain_files": pt_files,
            "target_station_id": config.target_station_id,
            "river_name": config.river_name,
            "finetune_weights_saved": finetune_weights_saved,
            "finetune_weights_path": save_path if finetune_weights_saved else None,
        }
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump(_jsonify(meta), f, indent=2)

        print(f"\n>>> 运行结束，结果保存在: {run_dir}")
        print(f">>> 测试损失: {test_loss:.5f}")
        print(f">>> 训练耗时: {train_end - train_start:.2f}秒")
