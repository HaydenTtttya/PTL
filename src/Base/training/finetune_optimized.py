import os
import re
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
FINETUNE_RUNS_DIR = os.path.join(ARTIFACTS_ROOT, "finetune", "runs")
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}


def should_save_finetune_model_weights(default=False):
    value = os.environ.get("BASE_FINETUNE_SAVE_MODEL_WEIGHTS")
    if value is None:
        return default
    return value.strip().lower() in TRUTHY_ENV_VALUES

# ==========================================
# 配置 (已根据论文参数调整)
# ==========================================
class FinetuneConfig:
    def __init__(self):
        # 路径配置 (请根据实际情况修改)
        self.water_data_dir = os.path.join(REPO_ROOT, "data", "data_cleaned", "yangzte")
        self.meteo_data_dir = os.path.join(REPO_ROOT, "data", "meteorology_weekly")
        self.save_dir = FINETUNE_RUNS_DIR
        self.save_model_weights = should_save_finetune_model_weights()

        # 预训练模型路径（必须指定）
        self.pretrain_model_dir = None  # 需要在运行时设置

        # 微调目标文件
        self.target_data_folder = os.path.join(REPO_ROOT, "data", "data_cleaned", "zhu")

        # self.target_water_file 可按需指定到 data/data_cleaned/zhu 下的单个站点文件
        self.river_name = "长江"
        self.target_station_id = "72"

        # [新增] 这些变量将在循环中动态更新，这里初始化为空即可
        self.target_water_file = None
        self.target_station_id = None
        self.river_name = "珠江" # 用于日志显示

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
        self.n_out = 1
        self.input_dim = 4
        self.feature_dim = 7
        self.hidden_size = 256  # 从预训练继承
        self.num_heads = 8      # 从预训练继承
        self.e_layer = 3        # 从预训练继承
        self.prompt_num = 1

        # 微调训练参数（已对齐论文）
        self.batch_size = 32
        self.epochs = 50             # 论文明确: "only 50 epochs"
        self.base_lr = 1e-2          # 论文原始代码 fine_tuning.py: base_lr = 1e-2
        self.epsilon = 1e-8

        self.weight_decay = 0.0      # 论文原始代码: weight_decay = 0.0
        self.train_ratio = 0.8
        self.lr_milestones = [40, 60, 80]
        self.lr_decay_ratio = 0.5
        self.max_grad_norm = 1.0

        # [修改] 不再使用 freeze_ratio，改为基于名称的策略
        # self.freeze_ratio = 0.38

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
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_latest_pretrain_run(pretrain_runs_dir):
    if not os.path.isdir(pretrain_runs_dir):
        return None

    candidates = []
    for entry in os.scandir(pretrain_runs_dir):
        if not entry.is_dir():
            continue

        config_path = os.path.join(entry.path, "config.json")
        model_path = os.path.join(entry.path, "model.pth")
        if os.path.exists(config_path) and os.path.exists(model_path):
            candidates.append(entry.path)

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)

def _flatten_preds_targets(preds: np.ndarray, targets: np.ndarray):
    preds_2d = preds.reshape(-1, preds.shape[-1])
    targets_2d = targets.reshape(-1, targets.shape[-1])
    return preds_2d, targets_2d

def compute_per_feature_metrics(preds: np.ndarray, targets: np.ndarray, feature_names):
    preds_2d, targets_2d = _flatten_preds_targets(preds, targets)
    M = preds_2d.shape[1]

    out = {}

    for j, name in enumerate(feature_names):
        y_pred = preds_2d[:, j]
        y_true = targets_2d[:, j]

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        nse = float(1.0 - (numerator / denominator)) if denominator != 0.0 else float("nan")

        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if np.any(mask) else float("nan")

        out[name] = {"MAE": mae, "RMSE": rmse, "NSE": nse, "MAPE": mape}

    # 总体指标
    overall_mse = float(np.mean((preds_2d - targets_2d) ** 2))
    overall_mae = float(np.mean(np.abs(preds_2d - targets_2d)))

    rmse_per_feature = [np.sqrt(np.mean((preds_2d[:, j] - targets_2d[:, j]) ** 2)) for j in range(M)]
    overall_rmse = float(np.mean(rmse_per_feature))

    nse_per_feature = []
    for j in range(M):
        y_true = targets_2d[:, j]
        y_pred = preds_2d[:, j]
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        if denominator != 0:
            nse_per_feature.append(1.0 - (numerator / denominator))
    overall_nse = float(np.mean(nse_per_feature)) if nse_per_feature else float("nan")

    mask = targets_2d != 0
    overall_mape = float(np.mean(np.abs((targets_2d[mask] - preds_2d[mask]) / targets_2d[mask])) * 100) if np.any(mask) else float("nan")

    out["__overall__"] = {
        "MSE": overall_mse,
        "MAE": overall_mae,
        "RMSE": overall_rmse,
        "NSE": overall_nse,
        "MAPE": overall_mape
    }

    return out

class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

# ==========================================
# 数据集
# ==========================================
class WaterMeteoDataset(Dataset):
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

def load_finetune_data(config, river_name, target_station_id):
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
# 模型定义（与原代码保持一致）
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

class MPS_TransformerEncoderLayer(nn.Module):
    """MPS兼容的编码器层 — 纯Linear+matmul实现，避免MPS backward的channels_last崩溃"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.nhead     = nhead
        self.head_dim  = d_model // nhead
        self.d_model   = d_model
        self.scale     = self.head_dim ** -0.5
        self.q_proj    = nn.Linear(d_model, d_model)
        self.k_proj    = nn.Linear(d_model, d_model)
        self.v_proj    = nn.Linear(d_model, d_model)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        H, D    = self.nhead, self.head_dim
        q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, D).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out  = torch.matmul(self.dropout(attn), v).transpose(1, 2).contiguous().view(B, L, self.d_model)
        x = self.norm1(x + self.dropout(self.out_proj(out)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class MeteorologyAttention(nn.Module):
    def __init__(self, num_memory, memory_dim, nhead):
        super(MeteorologyAttention, self).__init__()
        self.memory = Memory(num_memory, memory_dim)
        actual_nhead = max(1, min(nhead, memory_dim))
        while memory_dim % actual_nhead != 0 and actual_nhead > 1:
            actual_nhead -= 1
        self.transformer_encoder = MPS_TransformerEncoderLayer(
            d_model=memory_dim, nhead=actual_nhead, dim_feedforward=memory_dim * 4
        )

    def forward(self, water_data, weather_data):
        combined  = torch.cat((water_data, weather_data), dim=-1)
        modulated = self.transformer_encoder(combined)
        return self.memory(modulated)

class CombinedAttentionConcat(nn.Module):
    def __init__(self, input_dim_total, w_feat, seq_len, pred_len):
        super(CombinedAttentionConcat, self).__init__()
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

class Prompt_MultiTransformer(nn.Module):
    def __init__(self, config, num_pretrain_stations):
        super(Prompt_MultiTransformer, self).__init__()

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

        self.mlp = nn.Linear(num_pretrain_stations, config.prompt_num)

        self.combined_attention = CombinedAttentionConcat(
            input_dim_total=config.input_dim + config.feature_dim,
            w_feat=config.input_dim,
            seq_len=config.n_in,
            pred_len=config.n_out
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(config.prompt_num * 2, 32),
            nn.ReLU(),
            nn.Linear(32, config.prompt_num)
        )

    def forward(self, x_water, x_weather):
        outputs = []
        for transformer in self.transformers:
            out = transformer(x_water)
            outputs.append(out)

        out_main = torch.stack(outputs, dim=-1)
        out_main = self.mlp(out_main)

        out_prompt = self.combined_attention(x_water, x_weather).unsqueeze(-1)

        combined = torch.cat([out_main, out_prompt], dim=-1)
        final = self.fusion_layer(combined).squeeze(-1)

        return final

# ==========================================
# 训练和评估
# ==========================================
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

            # [修改] 确保维度对齐
            if output.shape != w_y.shape:
                w_y_aligned = w_y.view_as(output)
            else:
                w_y_aligned = w_y

            loss = criterion(output, w_y_aligned)
            total_loss += loss.item()

            pred_np = output.cpu().numpy()
            target_np = w_y_aligned.cpu().numpy()

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

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.base_lr,
        eps=config.epsilon,
        weight_decay=config.weight_decay,
        amsgrad=True
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_ratio)
    criterion = nn.MSELoss()

    history = []
    best_val_loss  = float('inf')
    best_state     = None          # 保存最佳模型权重

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0

        for w_x, m_x, w_y in train_loader:
            w_x, m_x, w_y = w_x.to(config.device), m_x.to(config.device), w_y.to(config.device)

            optimizer.zero_grad()
            output = model(w_x, m_x)

            if output.shape != w_y.shape:
                w_y = w_y.view_as(output)

            loss = criterion(output, w_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        if avg_train_loss > 1e5 or np.isnan(avg_train_loss):
            print(f"[错误] Epoch {epoch+1}: 检测到损失异常 (Loss: {avg_train_loss}). 停止训练。")
            break

        val_loss, _, _ = evaluate_finetune(model, val_loader, config.device, None)

        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": val_loss})

        if (epoch + 1) % 10 == 0:
            print(f"[Finetune] Epoch {epoch+1}/{config.epochs} | Loss: {avg_train_loss:.5f} | Val: {val_loss:.5f}")

        # ── best model checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── 恢复最佳模型 ──
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(config.device)
        print(f">>> 已恢复最佳模型 (best_val_loss={best_val_loss:.5f})")

    return model, history, best_val_loss

# ==========================================
# 主函数
# ==========================================
def main(pretrain_model_dir, custom_config=None, seed=42):
    set_seed(seed)

    # 1. 初始化基础配置
    config = FinetuneConfig()
    config.pretrain_model_dir = pretrain_model_dir

    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # 2. 加载预训练配置（只加载一次配置，权重后面循环加载）
    print(f"\n{'='*60}")
    print(f"批量微调开始 - 种子: {seed}")
    print(f"预训练模型: {pretrain_model_dir}")
    print(f"{'='*60}")

    pretrain_config_path = os.path.join(pretrain_model_dir, "config.json")
    with open(pretrain_config_path, "r") as f:
        pretrain_config = json.load(f)

    config.hidden_size = pretrain_config['hidden_size']
    config.num_heads = pretrain_config['num_heads']
    config.e_layer = pretrain_config['e_layer']
    num_pretrain_stations = pretrain_config['num_stations']

    # 3. 扫描目标文件夹下的所有CSV文件
    if not os.path.exists(config.target_data_folder):
        raise ValueError(f"目标文件夹不存在: {config.target_data_folder}")

    target_files = [f for f in os.listdir(config.target_data_folder)
                    if f.endswith('.csv') and not f.startswith('.')] # 过滤隐藏文件

    # 按文件名排序，保证顺序一致
    target_files.sort()

    print(f">>> 在 '{config.target_data_folder}' 下发现 {len(target_files)} 个站点文件: {target_files}")

    # ==========================================
    # 开始循环遍历每个站点
    # ==========================================
    results_summary = [] # 记录每个站点的最终结果

    for file_idx, filename in enumerate(target_files):
        print(f"\n\n{'#'*40}")
        print(f"正在处理第 {file_idx+1}/{len(target_files)} 个站点: {filename}")
        print(f"{'#'*40}")

        # --- A. 解析站点ID和路径 ---
        # 使用正则提取数字ID (例如 "珠江 3.csv" -> "3", "珠江72.csv" -> "72")
        match = re.search(r"(\d+)", filename)
        if not match:
            print(f"[跳过] 无法从文件名 {filename} 中提取站点ID")
            continue

        station_id = match.group(1)
        config.target_station_id = station_id
        config.target_water_file = os.path.join(config.target_data_folder, filename)

        print(f">>> 目标ID: {station_id} | 文件路径: {config.target_water_file}")

        # --- B. 加载该站点的数据 ---
        ft_train_loader, ft_test_loader, scaler_ft = load_finetune_data(
            config, config.river_name, config.target_station_id
        )

        if ft_train_loader is None:
            print(f"[错误] 数据加载失败: {filename}，跳过该站点")
            continue

        # --- C. 初始化模型 (每次循环都重新创建，保证权重重置) ---
        ft_model = Prompt_MultiTransformer(config, num_pretrain_stations=num_pretrain_stations)

        # --- D. 加载预训练权重 ---
        pretrain_model_path = os.path.join(pretrain_model_dir, "model.pth")
        pretrained_dict = torch.load(pretrain_model_path, map_location=config.device)
        ft_model_dict = ft_model.state_dict()

        matched_dict = {k: v for k, v in pretrained_dict.items()
                       if k in ft_model_dict and v.shape == ft_model_dict[k].shape}

        ft_model_dict.update(matched_dict)
        ft_model.load_state_dict(ft_model_dict, strict=False)

        # --- E. 实施冻结策略 ---
        for param in ft_model.parameters():
            param.requires_grad = False

        trainable_keywords = ['prompt', 'memory', 'mlp', 'fusion_layer', 'combined_attention']
        for name, param in ft_model.named_parameters():
            for kw in trainable_keywords:
                if kw in name:
                    param.requires_grad = True
                    break
            if 'transformers' in name:
                # head: Linear(hidden_size, pred_len) — pretrain用pred_len=8, finetune用1
                # shape mismatch → 未被转移 → 随机初始化 → 必须训练
                # encoder.norm 等其他权重均已从pretrain转移 → 冻结
                if 'head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # --- 打印可训练参数信息 ---
        trainable_params = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in ft_model.parameters())
        print(f">>> 可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.3f}%)")
        for n, p in ft_model.named_parameters():
            if p.requires_grad:
                print(f"    ✓ {n} ({p.numel()})")

        # --- F. 开始微调 ---
        train_start = time.time()
        ft_model, history, best_val_loss = fine_tune_model(ft_model, ft_train_loader, ft_test_loader, config)
        train_end = time.time()

        # --- G. 评估与保存 ---
        test_loss, preds, targets = evaluate_finetune(ft_model, ft_test_loader, config.device, scaler_ft)

        # 创建带站点ID的保存目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # [修改] 文件夹名包含具体的站点ID
        save_name = f"finetune_station{station_id}_seed{seed}_{timestamp}"
        current_save_dir = os.path.join(config.save_dir, save_name)
        os.makedirs(current_save_dir, exist_ok=True)

        # 保存模型权重默认关闭，避免批量微调 run 反复生成大体积 .pth。
        model_path = os.path.join(current_save_dir, "model.pth")
        model_weights_saved = bool(config.save_model_weights)
        if model_weights_saved:
            torch.save(ft_model.state_dict(), model_path)
        else:
            print(f">>> 跳过保存模型权重: {model_path}")

        # 保存预测结果 (CSV)
        feature_names = ['CODMn', 'DO', 'NH4N', 'pH']
        preds_flat = preds.squeeze(1)
        targets_flat = targets.squeeze(1)
        df_res = pd.DataFrame(preds_flat, columns=[f"Pred_{n}" for n in feature_names])
        df_tgt = pd.DataFrame(targets_flat, columns=[f"True_{n}" for n in feature_names])
        pd.concat([df_tgt, df_res], axis=1).to_csv(os.path.join(current_save_dir, "predictions.csv"), index=False)

        # 保存指标
        metrics = compute_per_feature_metrics(preds, targets, feature_names)
        pd.DataFrame.from_dict(metrics, orient="index").to_csv(os.path.join(current_save_dir, "metrics.csv"))

        # 保存历史
        pd.DataFrame(history).to_csv(os.path.join(current_save_dir, "history.csv"), index=False)

        print(f">>> 站点 {station_id} 完成。最佳Val Loss: {best_val_loss:.5f}")
        results_summary.append({
            "station": station_id,
            "filename": filename,
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "save_dir": current_save_dir,
            "model_weights_saved": model_weights_saved,
            "model_weights_path": model_path if model_weights_saved else None,
        })

    # ==========================================
    # 所有站点循环结束
    # ==========================================
    print(f"\n{'='*60}")
    print(f"批量任务全部完成 ({len(results_summary)}/{len(target_files)})")
    print(f"{'='*60}")

    # 打印简报
    print("结果汇总:")
    for res in results_summary:
        print(f"  - 站点 {res['station']}: Val Loss={res['best_val_loss']:.5f} | Test Loss={res['test_loss']:.5f}")

    return results_summary
# def main(pretrain_model_dir, custom_config=None, seed=42):
#     set_seed(seed)

#     # 初始化配置
#     config = FinetuneConfig()
#     config.pretrain_model_dir = pretrain_model_dir

#     # 应用自定义配置
#     if custom_config:
#         for key, value in custom_config.items():
#             if hasattr(config, key):
#                 setattr(config, key, value)

#     print(f"\n{'='*60}")
#     print(f"微调开始 - 种子: {seed}")
#     print(f"{'='*60}")

#     # 加载预训练配置
#     print("\n>>> 加载预训练模型配置...")
#     pretrain_config_path = os.path.join(pretrain_model_dir, "config.json")
#     with open(pretrain_config_path, "r") as f:
#         pretrain_config = json.load(f)

#     # 从预训练继承关键参数
#     config.hidden_size = pretrain_config['hidden_size']
#     config.num_heads = pretrain_config['num_heads']
#     config.e_layer = pretrain_config['e_layer']
#     num_pretrain_stations = pretrain_config['num_stations']

#     print(f">>> 预训练模型: {pretrain_model_dir}")
#     print(f">>> 预训练站点数: {num_pretrain_stations}")
#     print(f">>> 继承超参数: hidden_size={config.hidden_size}, num_heads={config.num_heads}, e_layer={config.e_layer}")

#     # 加载微调数据
#     print("\n>>> 加载微调数据...")
#     ft_train_loader, ft_test_loader, scaler_ft = load_finetune_data(
#         config, config.river_name, config.target_station_id
#     )

#     if ft_train_loader is None:
#         raise ValueError("微调数据加载失败")

#     # 创建微调模型
#     print("\n>>> 创建微调模型...")
#     ft_model = Prompt_MultiTransformer(config, num_pretrain_stations=num_pretrain_stations)

#     # 加载预训练权重
#     print(">>> 加载预训练权重...")
#     pretrain_model_path = os.path.join(pretrain_model_dir, "model.pth")
#     pretrained_dict = torch.load(pretrain_model_path, map_location=config.device)
#     ft_model_dict = ft_model.state_dict()

#     matched_dict = {k: v for k, v in pretrained_dict.items()
#                    if k in ft_model_dict and v.shape == ft_model_dict[k].shape}

#     ft_model_dict.update(matched_dict)
#     ft_model.load_state_dict(ft_model_dict, strict=False)
#     print(f">>> 成功加载 {len(matched_dict)} 层权重")

#     # [修改] 实施论文的 Prompt Tuning 冻结策略
#     print(">>> 实施冻结策略 (Prompt Tuning Style)...")

#     # 1. 默认冻结所有参数
#     for param in ft_model.parameters():
#         param.requires_grad = False

#     # 2. 解冻 Prompt 模块、MLP 头和融合层
#     # 这些关键词涵盖了 prompt-tuning 需要训练的部分
#     trainable_keywords = ['prompt', 'memory', 'mlp', 'fusion_layer', 'combined_attention']

#     trainable_count = 0
#     total_count = 0

#     for name, param in ft_model.named_parameters():
#         total_count += 1
#         # 如果名字里包含可训练关键词，解冻
#         for kw in trainable_keywords:
#             if kw in name:
#                 param.requires_grad = True
#                 trainable_count += 1
#                 break

#         # [双重保险] 强制冻结 transformer 主干，防止 keywords 误伤
#         if 'transformers' in name:
#              param.requires_grad = False

#     # 重新统计
#     final_trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
#     final_total = sum(p.numel() for p in ft_model.parameters())

#     print(f">>> 冻结完成。可训练参数: {final_trainable} / {final_total} ({final_trainable/final_total:.2%})")

#     # 微调训练
#     print("\n>>> 开始微调训练...")
#     train_start = time.time()
#     ft_model, history, best_val_loss = fine_tune_model(ft_model, ft_train_loader, ft_test_loader, config)
#     train_end = time.time()

#     # 最终评估
#     print("\n>>> 最终评估...")
#     test_loss, preds, targets = evaluate_finetune(ft_model, ft_test_loader, config.device, scaler_ft)

#     # 保存结果
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_name = f"finetune_station{config.target_station_id}_seed{seed}_{timestamp}"
#     save_dir = os.path.join(config.save_dir, save_name)
#     os.makedirs(save_dir, exist_ok=True)

#     # 保存模型
#     model_path = os.path.join(save_dir, "model.pth")
#     torch.save(ft_model.state_dict(), model_path)

#     # 保存预测结果
#     feature_names = ['CODMn', 'DO', 'NH4N', 'pH']
#     preds_flat = preds.squeeze(1)
#     targets_flat = targets.squeeze(1)
#     df_res = pd.DataFrame(preds_flat, columns=[f"Pred_{n}" for n in feature_names])
#     df_tgt = pd.DataFrame(targets_flat, columns=[f"True_{n}" for n in feature_names])
#     pd.concat([df_tgt, df_res], axis=1).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

#     # 保存指标
#     metrics = compute_per_feature_metrics(preds, targets, feature_names)
#     pd.DataFrame.from_dict(metrics, orient="index").to_csv(os.path.join(save_dir, "metrics.csv"))

#     # 保存元信息
#     meta = {
#         'pretrain_model_dir': pretrain_model_dir,
#         'target_station_id': config.target_station_id,
#         'seed': seed,
#         'epochs': config.epochs,
#         'base_lr': config.base_lr,
#         'weight_decay': config.weight_decay,
#         'best_val_loss': best_val_loss,
#         'test_loss': test_loss,
#         'metrics': metrics['__overall__']
#     }

#     with open(os.path.join(save_dir, "meta.json"), "w") as f:
#         json.dump(meta, f, indent=2)

#     pd.DataFrame(history).to_csv(os.path.join(save_dir, "history.csv"), index=False)

#     print(f"\n{'='*60}")
#     print(f"微调完成")
#     print(f"最佳验证损失: {best_val_loss:.6f}")
#     print(f"结果已保存至: {save_dir}")

#     return best_val_loss, save_dir

if __name__ == '__main__':
    pretrain_dir = find_latest_pretrain_run(PRETRAIN_RUNS_DIR)

    if pretrain_dir is None:
        print(
            f"Warning: 未找到可用的预训练目录，请检查 {PRETRAIN_RUNS_DIR} 下是否存在包含 "
            "config.json 和 model.pth 的目录。"
        )
    else:
        main(pretrain_model_dir=pretrain_dir, seed=42)
