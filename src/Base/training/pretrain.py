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

        # 模型超参数（可调）
        self.n_in = 8
        self.input_dim = 4
        self.hidden_size = 256
        self.num_heads = 16
        self.e_layer = 3
        self.batch_size = 16
        self.pretrain_epochs = 50
        self.base_lr = 0.0009
        self.epsilon = 1e-8
        self.train_ratio = 0.8
        self.mask_ratio = 0.5
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
    os.environ["PYTHONHASHSEED"] = str(seed)

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
    stacked = np.stack(dfs_w, axis=-1)

    scaler = StandardScaler()
    scaler.fit(stacked)
    stacked = scaler.transform(stacked)

    train_dataset = PretrainMaskedDataset(stacked, config.n_in, pred_len, config.mask_ratio, "train", config.train_ratio)
    test_dataset = PretrainMaskedDataset(stacked, config.n_in, pred_len, config.mask_ratio, "test", config.train_ratio)

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
        outputs = []
        for i in range(len(self.transformers)):
            channel_data = x[..., i]
            channel_out = self.transformers[i](channel_data)
            outputs.append(channel_out)
        out = torch.stack(outputs, dim=-1)
        out = self.mlp(out)
        return out

# ==========================================
# 训练函数
# ==========================================
def pre_train_model(model, train_loader, val_loader, config):
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr, eps=config.epsilon)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_ratio)
    criterion = nn.MSELoss()

    history = []
    best_val_loss = float('inf')

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

        # 验证
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.device), target.to(config.device)
                output = model(data)
                val_loss += criterion(output, target).item()
        val_loss = val_loss / len(val_loader)

        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": val_loss})

        if (epoch + 1) % 10 == 0:
            print(f"[Pretrain] Epoch {epoch+1}/{config.pretrain_epochs} | Loss: {avg_train_loss:.5f} | Val: {val_loss:.5f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return model, history, best_val_loss

# ==========================================
# 主函数
# ==========================================
def main(custom_config=None, seed=42):
    """
    主训练函数
    Args:
        custom_config: 自定义配置字典（用于贝叶斯调参）
        seed: 随机种子
    Returns:
        best_val_loss: 最佳验证损失（用于调参）
        save_path: 模型保存路径
    """
    set_seed(seed)

    # 初始化配置
    config = PretrainConfig()

    # 应用自定义配置（调参用）
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    print(f"\n{'='*60}")
    print(f"预训练开始 - 种子: {seed}")
    print(f"{'='*60}")
    print(f"数据目录: {config.water_data_dir}")
    print(f"流域: {config.river_name}")
    print(f"超参数: hidden_size={config.hidden_size}, num_heads={config.num_heads}, "
          f"e_layer={config.e_layer}, lr={config.base_lr}")

    # 加载数据
    print("\n>>> 加载预训练数据...")
    pt_loader, pt_val_loader, scaler, pt_files = load_pretrain_data(config, config.river_name, config.n_in)

    if pt_loader is None:
        raise ValueError("预训练数据加载失败")

    print(f">>> 找到 {len(pt_files)} 个站点")
    print(f">>> 站点文件: {pt_files}")

    # 创建模型
    print("\n>>> 创建模型...")
    model = MultiTransformer(
        num_heads=config.num_heads,
        e_layer=config.e_layer,
        hidden_size=config.hidden_size,
        num_stations=len(pt_files),
        num_feat=config.input_dim,
        seq_len=config.n_in,
        pred_len=config.n_in
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> 模型参数量: {total_params:,}")

    # 训练
    print("\n>>> 开始训练...")
    train_start = time.time()
    model, history, best_val_loss = pre_train_model(model, pt_loader, pt_val_loader, config)
    train_end = time.time()

    # 保存模型和配置
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"pretrain_{config.river_name}_seed{seed}_{timestamp}"
    save_dir = os.path.join(config.save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    # 保存权重
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    # 保存配置
    config_dict = {
        'num_stations': len(pt_files),
        'station_files': pt_files,
        'n_in': config.n_in,
        'input_dim': config.input_dim,
        'hidden_size': config.hidden_size,
        'num_heads': config.num_heads,
        'e_layer': config.e_layer,
        'batch_size': config.batch_size,
        'pretrain_epochs': config.pretrain_epochs,
        'base_lr': config.base_lr,
        'epsilon': config.epsilon,
        'train_ratio': config.train_ratio,
        'mask_ratio': config.mask_ratio,
        'lr_milestones': config.lr_milestones,
        'lr_decay_ratio': config.lr_decay_ratio,
        'max_grad_norm': config.max_grad_norm,
        'best_val_loss': best_val_loss,
        'train_time': train_end - train_start,
        'seed': seed
    }

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # 保存scaler
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # 保存训练历史
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "history.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"预训练完成")
    print(f"{'='*60}")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"训练耗时: {train_end - train_start:.2f}秒")
    print(f"模型已保存至: {save_dir}")
    print(f"  - 模型权重: {model_path}")
    print(f"  - 配置文件: {os.path.join(save_dir, 'config.json')}")
    print(f"  - Scaler: {os.path.join(save_dir, 'scaler.pkl')}")

    return best_val_loss, save_dir

if __name__ == '__main__':
    # 直接运行（不调参）
    best_val_loss, save_path = main(seed=42)

    # 贝叶斯调参示例（注释掉）
    # from bayes_opt import BayesianOptimization
    #
    # def objective(**params):
    #     config = {
    #         'hidden_size': int(params['hidden_size']),
    #         'num_heads': int(params['num_heads']),
    #         'e_layer': int(params['e_layer']),
    #         'base_lr': params['base_lr'],
    #         'mask_ratio': params['mask_ratio']
    #     }
    #     val_loss, _ = main(custom_config=config, seed=42)
    #     return -val_loss  # 返回负值因为要最大化
    #
    # optimizer = BayesianOptimization(
    #     f=objective,
    #     pbounds={
    #         'hidden_size': (128, 512),
    #         'num_heads': (4, 16),
    #         'e_layer': (2, 6),
    #         'base_lr': (1e-4, 1e-3),
    #         'mask_ratio': (0.3, 0.7)
    #     },
    #     random_state=42
    # )
    #
    # optimizer.maximize(init_points=5, n_iter=20)
    # print(f"\n最佳参数: {optimizer.max}")
