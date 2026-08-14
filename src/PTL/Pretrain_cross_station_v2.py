import argparse
import copy
import datetime
import json
import os
import pickle
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from progressive_core import (
    WaterQualityTransformer,
    build_start_indices,
    fit_scaler_on_train_slices,
    infer_device,
    resize_sequence_length,
    set_seed,
    snapshot_state_dict,
)
from model_agnostic_backbones import (
    SUPPORTED_MODEL_AGNOSTIC_BACKBONES,
    TemporalFeatureTokenPretrainAdapter,
    build_model_agnostic_forecaster,
    build_temporal_forecasting_backbone,
    normalize_backbone_name,
    normalize_model_agnostic_interface,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]


@dataclass
class CrossStationPretrainConfig:
    basin_data_root: str = str(REPO_ROOT / "data" / "data_cleaned")
    basin_key: str = "yangzte"
    river_name: str = "长江"
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
    selected_stations: tuple[str, ...] | None = None
    max_stations: int | None = None
    min_series_length: int = 200

    raw_seq_len: int = 168
    model_seq_len: int = 168
    input_dim: int = len(FEATURE_COLUMNS)
    backbone_name: str = "transformer"
    model_agnostic_interface: str = "legacy"
    hidden_size: int = 256
    num_heads: int = 8
    e_layer: int = 3
    cross_station_heads: int = 4
    cross_station_layers: int = 1
    cross_station_dropout: float = 0.1
    station_identity_enabled: bool = True
    station_embedding_init_std: float = 0.02

    cnn_channels: tuple[int, ...] = (64, 128, 128)
    cnn_kernel_sizes: tuple[int, ...] = (3, 3, 3)
    cnn_dilations: tuple[int, ...] = (1, 2, 4)
    cnn_use_batch_norm: bool = True
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_use_input_layer_norm: bool = True
    backbone_dropout: float = 0.15
    backbone_activation: str = "gelu"
    backbone_head_hidden_dim: int = 128
    mlp_hidden_dims: tuple[int, ...] = (256, 128)
    mlp_dropout: float = 0.1
    mlp_use_layer_norm: bool = True
    cnn_lstm_conv_channels: tuple[int, ...] = (64, 128)
    cnn_lstm_kernel_sizes: tuple[int, ...] = (3, 3)
    cnn_lstm_hidden_dim: int = 128
    cnn_lstm_layers: int = 1
    cnn_lstm_use_batch_norm: bool = True
    cnn_lstm_use_input_layer_norm: bool = True

    batch_size: int = 16
    pretrain_epochs: int = 60
    base_lr: float = 1e-3
    epsilon: float = 1e-8
    weight_decay: float = 1e-2
    train_ratio: float = 0.8
    mask_strategy: str = "station"
    mask_ratio: float = 0.15
    clean_window_ratio: float = 0.7
    local_loss_weight: float = 1.0
    cross_all_loss_weight: float = 0.5
    cross_masked_loss_weight: float = 1.0
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-5
    lr_decay_ratio: float = 0.5

    min_available_stations: int = 2
    max_train_windows: int | None = None
    max_val_windows: int | None = None

    base_export_seq_len: int = 8
    base_export_prompt_num: int = 1

    save_root: str = str(REPO_ROOT / "results" / "cross_station" / "pretrain" / "v2_runs")
    mirror_compatible_runs: bool = False
    ptl_compat_root: str = str(REPO_ROOT / "results" / "ptl" / "pretrain" / "runs")
    base_compat_root: str = str(REPO_ROOT / "results" / "base" / "pretrain" / "runs")

    max_train_batches: int | None = None
    max_val_batches: int | None = None
    device_name: str | None = None

    @property
    def basin_dir(self) -> Path:
        return Path(self.basin_data_root) / self.basin_key

    @property
    def run_prefix(self) -> str:
        backbone_name = normalize_backbone_name(self.backbone_name)
        backbone_tag = "" if backbone_name == "transformer" else f"_{backbone_name}"
        return f"pretrain_cross_station_v2{backbone_tag}_{self.basin_key}_seed"


class CrossStationMaskedDataset(Dataset):
    def __init__(
        self,
        series_list,
        station_names,
        raw_seq_len,
        model_seq_len,
        split="train",
        train_ratio=0.8,
        min_available_stations=2,
        max_windows=None,
        resize_mode="linear",
    ):
        self.series_list = []
        self.station_names = list(station_names)
        self.raw_seq_len = int(raw_seq_len)
        self.model_seq_len = int(model_seq_len)
        self.resize_mode = resize_mode
        self.samples = []
        self.available_station_counts = []

        for series in series_list:
            train_end = max(1, int(len(series) * train_ratio))
            if split == "train":
                sliced = series[:train_end]
            elif split == "val":
                start = max(0, train_end - raw_seq_len + 1)
                sliced = series[start:]
            else:
                raise ValueError(f"Unsupported split: {split}")
            self.series_list.append(sliced.astype(np.float32, copy=False))

        candidate_window_count = max(
            0,
            max((len(series) - raw_seq_len + 1) for series in self.series_list),
        ) if self.series_list else 0
        starts = build_start_indices(candidate_window_count, max_windows=max_windows)
        for start in starts:
            available = [
                series_idx
                for series_idx, series in enumerate(self.series_list)
                if len(series) >= int(start) + self.raw_seq_len
            ]
            if len(available) < int(min_available_stations):
                continue
            self.samples.append((int(start), tuple(available)))
            self.available_station_counts.append(len(available))

        self.candidate_window_count = int(candidate_window_count)
        self.filtered_window_count = int(candidate_window_count - len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        start, available_series = self.samples[index]
        target = np.zeros(
            (len(self.series_list), self.model_seq_len, len(FEATURE_COLUMNS)),
            dtype=np.float32,
        )
        available_mask = np.zeros(len(self.series_list), dtype=bool)

        for series_idx in available_series:
            raw_window = self.series_list[series_idx][start:start + self.raw_seq_len]
            resized = resize_sequence_length(
                raw_window,
                self.model_seq_len,
                mode=self.resize_mode,
            )
            target[series_idx] = resized
            available_mask[series_idx] = True

        return (
            torch.as_tensor(target, dtype=torch.float32),
            torch.as_tensor(available_mask, dtype=torch.bool),
            torch.tensor(index, dtype=torch.long),
        )

    def summary(self):
        if not self.available_station_counts:
            return {
                "windows": 0,
                "candidate_windows": self.candidate_window_count,
                "filtered_windows": self.filtered_window_count,
                "mean_available_stations": 0.0,
                "median_available_stations": 0.0,
                "min_available_stations": 0,
                "max_available_stations": 0,
                "effective_station_windows": 0,
            }

        counts = np.asarray(self.available_station_counts, dtype=np.int64)
        return {
            "windows": len(self.samples),
            "candidate_windows": self.candidate_window_count,
            "filtered_windows": self.filtered_window_count,
            "mean_available_stations": float(np.mean(counts)),
            "median_available_stations": float(np.median(counts)),
            "min_available_stations": int(np.min(counts)),
            "max_available_stations": int(np.max(counts)),
            "effective_station_windows": int(np.sum(counts)),
        }


class CrossStationInteractionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, key_padding_mask=None):
        attended, _ = self.attention(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ffn(x))
        return x


class CrossStationBackbonePretrainer(nn.Module):
    def __init__(
        self,
        backbone,
        num_stations,
        cross_station_heads,
        cross_station_layers,
        dropout,
        station_identity_enabled=True,
        station_embedding_init_std=0.02,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_stations = int(num_stations)
        self.uses_generic_pretrain_interface = all(
            hasattr(backbone, attribute)
            for attribute in ("encode_pretrain_tokens", "decode_pretrain_tokens")
        )
        self.cross_station_layers = nn.ModuleList(
            [
                CrossStationInteractionBlock(
                    hidden_size=backbone.d_model,
                    num_heads=cross_station_heads,
                    dropout=dropout,
                )
                for _ in range(cross_station_layers)
            ]
        )
        self.fusion_scale = nn.Parameter(torch.zeros(1))
        self.station_identity_enabled = bool(station_identity_enabled)
        if self.station_identity_enabled:
            self.station_embedding = nn.Embedding(self.num_stations, backbone.d_model)
            nn.init.normal_(self.station_embedding.weight, mean=0.0, std=station_embedding_init_std)
            self.station_identity_scale = nn.Parameter(torch.ones(1))
        else:
            self.station_embedding = None
            self.station_identity_scale = None

    def encode_station(self, masked_station_window, target_station_window, masked_positions):
        if self.uses_generic_pretrain_interface:
            encoded = self.backbone.encode_pretrain_tokens(
                masked_station_window.masked_fill(masked_positions.bool(), 0.0)
            )
            means = torch.zeros_like(target_station_window[:, :1])
            stdev = torch.ones_like(target_station_window[:, :1])
            return encoded, means, stdev

        means = target_station_window.mean(dim=1, keepdim=True).detach()
        centered = target_station_window - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (masked_station_window - means) / stdev
        normalized = normalized.masked_fill(masked_positions.bool(), 0.0)
        normalized = normalized.permute(0, 2, 1)
        if self.backbone.temporal_adapter is not None:
            normalized = self.backbone.temporal_adapter(normalized)
        encoded = self.backbone.embedding(normalized)
        encoded = self.backbone.encoder(encoded)
        return encoded, means, stdev

    def decode_station(self, encoded, means, stdev):
        if self.uses_generic_pretrain_interface:
            return self.backbone.decode_pretrain_tokens(encoded)

        decoded = self.backbone.head(encoded).permute(0, 2, 1)
        decoded = decoded[:, :, : self.backbone.target_dim]
        target_stdev = stdev[:, 0, : self.backbone.target_dim].unsqueeze(1).repeat(
            1,
            self.backbone.pred_len,
            1,
        )
        target_means = means[:, 0, : self.backbone.target_dim].unsqueeze(1).repeat(
            1,
            self.backbone.pred_len,
            1,
        )
        return decoded * target_stdev + target_means

    def forward(self, masked_windows, target_windows, masked_positions, station_available):
        batch_size, num_stations, _, _ = masked_windows.shape
        encoded_stations = []
        means_list = []
        stdev_list = []

        for station_idx in range(num_stations):
            encoded, means, stdev = self.encode_station(
                masked_windows[:, station_idx],
                target_windows[:, station_idx],
                masked_positions[:, station_idx],
            )
            encoded_stations.append(encoded)
            means_list.append(means)
            stdev_list.append(stdev)

        encoded_stations = torch.stack(encoded_stations, dim=1)
        feature_token_count = encoded_stations.shape[2]
        local_predictions = []
        for station_idx in range(num_stations):
            local_predictions.append(
                self.decode_station(
                    encoded_stations[:, station_idx],
                    means_list[station_idx],
                    stdev_list[station_idx],
                )
            )
        local_predictions = torch.stack(local_predictions, dim=1)

        station_enriched = encoded_stations
        if self.station_embedding is not None:
            station_ids = torch.arange(num_stations, device=encoded_stations.device)
            station_bias = self.station_embedding(station_ids).view(1, num_stations, 1, -1)
            station_enriched = station_enriched + (self.station_identity_scale * station_bias)

        station_tokens = station_enriched.permute(0, 2, 1, 3).reshape(
            batch_size * feature_token_count,
            num_stations,
            self.backbone.d_model,
        )
        key_padding_mask = (~station_available.bool()).unsqueeze(1).expand(
            -1,
            feature_token_count,
            -1,
        ).reshape(batch_size * feature_token_count, num_stations)

        station_context = station_tokens
        for layer in self.cross_station_layers:
            station_context = layer(station_context, key_padding_mask=key_padding_mask)

        station_context = station_context.reshape(
            batch_size,
            feature_token_count,
            num_stations,
            self.backbone.d_model,
        ).permute(0, 2, 1, 3)

        fused = encoded_stations + torch.tanh(self.fusion_scale) * station_context

        predictions = []
        for station_idx in range(num_stations):
            predictions.append(
                self.decode_station(
                    fused[:, station_idx],
                    means_list[station_idx],
                    stdev_list[station_idx],
                )
            )
        cross_predictions = torch.stack(predictions, dim=1)
        return local_predictions, cross_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-station pretraining with compatibility exports for PTL and Base finetune flows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone",
        type=str,
        default="transformer",
        choices=("transformer", *SUPPORTED_MODEL_AGNOSTIC_BACKBONES),
        help="Forecasting backbone used for cross-station pretraining.",
    )
    parser.add_argument(
        "--model-agnostic-interface",
        choices=("legacy", "feature_token_v1", "feature_token_residual_v2"),
        default="legacy",
        help="Representation interface shared by CNN/LSTM pretraining and forecasting.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--e-layer", type=int, default=None)
    parser.add_argument("--cross-station-heads", type=int, default=None)
    parser.add_argument("--cross-station-layers", type=int, default=None)
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--raw-seq-len", type=int, default=None)
    parser.add_argument("--model-seq-len", type=int, default=None)
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--mask-strategy", type=str, default=None, choices=sorted(MASK_BUILDERS))
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--clean-window-ratio", type=float, default=None)
    parser.add_argument("--local-loss-weight", type=float, default=None)
    parser.add_argument("--cross-all-loss-weight", type=float, default=None)
    parser.add_argument("--cross-masked-loss-weight", type=float, default=None)
    parser.add_argument("--mirror-compatible-runs", action="store_true")
    parser.add_argument("--save-root", type=str, default=None)
    return parser.parse_args()


def load_station_series(config):
    basin_dir = config.basin_dir
    if not basin_dir.is_dir():
        raise FileNotFoundError(f"Missing basin directory: {basin_dir}")

    station_files = sorted(
        filename
        for filename in os.listdir(basin_dir)
        if filename.endswith(".csv") and not filename.startswith(".")
    )
    if config.selected_stations:
        selected = set(config.selected_stations)
        station_files = [
            filename
            for filename in station_files
            if filename in selected or os.path.splitext(filename)[0] in selected
        ]

    loaded = []
    for filename in station_files:
        frame = pd.read_csv(basin_dir / filename)
        if not set(config.feature_columns).issubset(frame.columns):
            continue

        values = frame[list(config.feature_columns)].apply(pd.to_numeric, errors="coerce")
        values = values.interpolate(limit_direction="both").dropna()
        if len(values) < max(config.min_series_length, config.raw_seq_len + 1):
            continue
        loaded.append((os.path.splitext(filename)[0], values.to_numpy(dtype=np.float32, copy=True)))

    loaded.sort(key=lambda item: len(item[1]), reverse=True)
    if config.max_stations is not None:
        loaded = loaded[: config.max_stations]

    if not loaded:
        raise ValueError("No station series are available for cross-station pretraining.")

    scaler = fit_scaler_on_train_slices([values for _, values in loaded], config.train_ratio)
    scaled = [scaler.transform(values).astype(np.float32, copy=False) for _, values in loaded]
    station_names = [name for name, _ in loaded]
    raw_lengths = {name: int(len(values)) for name, values in loaded}
    return scaled, station_names, raw_lengths, scaler


def _random_like(shape, device, generator=None):
    if generator is None:
        return torch.rand(shape, device=device)
    return torch.rand(shape, generator=generator, device="cpu").to(device)


def _ensure_visible_positions(masked_positions, station_available):
    masked_positions = masked_positions.clone()
    visible = station_available.unsqueeze(-1).unsqueeze(-1) & (~masked_positions)
    collapsed = visible.view(visible.shape[0], -1).any(dim=1)
    for batch_idx in torch.nonzero(~collapsed, as_tuple=False).flatten().tolist():
        available_station_indices = torch.nonzero(
            station_available[batch_idx],
            as_tuple=False,
        ).flatten()
        if len(available_station_indices) == 0:
            continue
        station_idx = int(available_station_indices[0].item())
        masked_positions[batch_idx, station_idx] = False
    return masked_positions


def random_masking(targets, station_available, mask_ratio, generator=None):
    valid = station_available.unsqueeze(-1).unsqueeze(-1).expand_as(targets)
    masked_positions = valid & (_random_like(targets.shape, targets.device, generator=generator) < mask_ratio)
    return _ensure_visible_positions(masked_positions, station_available)


def feature_masking(targets, station_available, mask_ratio, generator=None):
    batch_size, num_stations, _, num_features = targets.shape
    valid = station_available.unsqueeze(-1).expand(batch_size, num_stations, num_features)
    mask_by_feature = valid & (
        _random_like((batch_size, num_stations, num_features), targets.device, generator=generator)
        < mask_ratio
    )
    masked_positions = mask_by_feature.unsqueeze(2).expand_as(targets)
    return _ensure_visible_positions(masked_positions, station_available)


def station_masking(targets, station_available, mask_ratio, generator=None):
    batch_size, num_stations = station_available.shape
    mask_by_station = station_available & (
        _random_like((batch_size, num_stations), targets.device, generator=generator) < mask_ratio
    )
    masked_positions = mask_by_station.unsqueeze(-1).unsqueeze(-1).expand_as(targets)
    return _ensure_visible_positions(masked_positions, station_available)


def temporal_masking(targets, station_available, mask_ratio, generator=None):
    batch_size, _, seq_len, _ = targets.shape
    mask_by_time = _random_like((batch_size, seq_len), targets.device, generator=generator) < mask_ratio
    masked_positions = mask_by_time.unsqueeze(1).unsqueeze(-1).expand_as(targets)
    masked_positions = masked_positions & station_available.unsqueeze(-1).unsqueeze(-1)
    return _ensure_visible_positions(masked_positions, station_available)


MASK_BUILDERS = {
    "random": random_masking,
    "feature": feature_masking,
    "station": station_masking,
    "temporal": temporal_masking,
}


def build_masked_inputs(
    targets,
    station_available,
    mask_ratio,
    mask_strategy,
    clean_window_ratio=0.0,
    sample_indices=None,
    base_seed=None,
):
    if mask_strategy not in MASK_BUILDERS:
        raise ValueError(f"Unsupported mask_strategy: {mask_strategy}")
    masked_inputs = targets.clone()
    masked_positions = torch.zeros_like(targets, dtype=torch.bool)
    for batch_idx in range(targets.shape[0]):
        generator = None
        if sample_indices is not None and base_seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(base_seed + int(sample_indices[batch_idx].item())))
        keep_clean = False
        if clean_window_ratio > 0.0:
            clean_draw = _random_like((1,), targets.device, generator=generator).flatten()[0]
            keep_clean = bool(clean_draw.item() < clean_window_ratio)
        if keep_clean:
            continue
        local_mask = MASK_BUILDERS[mask_strategy](
            targets[batch_idx:batch_idx + 1],
            station_available[batch_idx:batch_idx + 1],
            mask_ratio=mask_ratio,
            generator=generator,
        )
        masked_inputs[batch_idx][local_mask[0]] = 0.0
        masked_positions[batch_idx] = local_mask[0]
    return masked_inputs, masked_positions


def compute_pretraining_loss(
    local_preds,
    cross_preds,
    targets,
    station_available,
    masked_positions,
    local_loss_weight,
    cross_all_loss_weight,
    cross_masked_loss_weight,
):
    valid_positions = station_available.unsqueeze(-1).unsqueeze(-1).expand_as(targets)
    local_squared_error = (local_preds - targets) ** 2
    cross_squared_error = (cross_preds - targets) ** 2

    local_valid_error = local_squared_error[valid_positions]
    local_mse = (
        torch.mean(local_valid_error)
        if local_valid_error.numel() > 0
        else torch.zeros((), device=targets.device, dtype=targets.dtype)
    )
    cross_valid_error = cross_squared_error[valid_positions]
    cross_all_mse = (
        torch.mean(cross_valid_error)
        if cross_valid_error.numel() > 0
        else torch.zeros((), device=targets.device, dtype=targets.dtype)
    )
    masked_valid = masked_positions & valid_positions
    masked_error = cross_squared_error[masked_valid]
    cross_masked_mse = (
        torch.mean(masked_error)
        if masked_error.numel() > 0
        else cross_all_mse
    )
    valid_count = valid_positions.float().sum()
    masked_fraction = (
        masked_valid.float().sum() / valid_count
        if valid_count.item() > 0
        else torch.zeros((), device=targets.device, dtype=targets.dtype)
    )
    loss = (
        (local_loss_weight * local_mse)
        + (cross_all_loss_weight * cross_all_mse)
        + (cross_masked_loss_weight * cross_masked_mse)
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "local_mse": float(local_mse.detach().cpu()),
        "cross_all_mse": float(cross_all_mse.detach().cpu()),
        "cross_masked_mse": float(cross_masked_mse.detach().cpu()),
        "masked_fraction": float(masked_fraction.detach().cpu()),
    }


def compute_numpy_metrics(preds, targets, valid_mask):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != preds.shape:
        raise ValueError("valid_mask shape must match preds/targets.")
    if not np.any(valid_mask):
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}

    valid_preds = preds[valid_mask]
    valid_targets = targets[valid_mask]
    diff = valid_preds - valid_targets
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    denom = np.where(np.abs(valid_targets) < 1e-8, np.nan, np.abs(valid_targets))
    mape = float(np.nanmean(np.abs(diff) / denom) * 100.0)
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}


@torch.no_grad()
def summarize_checkpoint(model, loader, device, config, seed, scaler, station_names):
    model.eval()
    local_preds_list = []
    cross_preds_list = []
    targets_list = []
    valid_masks = []

    for batch_idx, (targets, station_available, sample_indices) in enumerate(loader):
        if config.max_val_batches is not None and batch_idx >= config.max_val_batches:
            break
        targets = targets.to(device)
        station_available = station_available.to(device)
        sample_indices = sample_indices.to(device)

        masked_inputs, masked_positions = build_masked_inputs(
            targets,
            station_available,
            mask_ratio=config.mask_ratio,
            mask_strategy=config.mask_strategy,
            clean_window_ratio=config.clean_window_ratio,
            sample_indices=sample_indices,
            base_seed=seed * 100_000,
        )
        local_preds, cross_preds = model(masked_inputs, targets, masked_positions, station_available)

        local_preds_list.append(local_preds.detach().cpu().numpy())
        cross_preds_list.append(cross_preds.detach().cpu().numpy())
        targets_list.append(targets.detach().cpu().numpy())
        valid_masks.append(
            station_available.unsqueeze(-1).unsqueeze(-1).expand_as(targets).detach().cpu().numpy()
        )

    local_preds = np.concatenate(local_preds_list, axis=0)
    cross_preds = np.concatenate(cross_preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    valid_mask = np.concatenate(valid_masks, axis=0).astype(bool)

    local_raw = scaler.inverse_transform(local_preds)
    cross_raw = scaler.inverse_transform(cross_preds)
    targets_raw = scaler.inverse_transform(targets)

    per_station_cross = {}
    per_station_local = {}
    for station_idx, station_name in enumerate(station_names):
        station_valid = valid_mask[:, station_idx]
        per_station_cross[station_name] = compute_numpy_metrics(
            cross_raw[:, station_idx],
            targets_raw[:, station_idx],
            station_valid,
        )
        per_station_local[station_name] = compute_numpy_metrics(
            local_raw[:, station_idx],
            targets_raw[:, station_idx],
            station_valid,
        )

    return {
        "cross_scaled_metrics": compute_numpy_metrics(cross_preds, targets, valid_mask),
        "cross_raw_metrics": compute_numpy_metrics(cross_raw, targets_raw, valid_mask),
        "local_scaled_metrics": compute_numpy_metrics(local_preds, targets, valid_mask),
        "local_raw_metrics": compute_numpy_metrics(local_raw, targets_raw, valid_mask),
        "per_station_cross_raw_metrics": per_station_cross,
        "per_station_local_raw_metrics": per_station_local,
    }


def train_one_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    total_local_mse = 0.0
    total_cross_all_mse = 0.0
    total_cross_masked_mse = 0.0
    total_masked_fraction = 0.0

    for batch_idx, (targets, station_available, sample_indices) in enumerate(loader):
        if config.max_train_batches is not None and batch_idx >= config.max_train_batches:
            break
        targets = targets.to(device)
        station_available = station_available.to(device)
        masked_inputs, masked_positions = build_masked_inputs(
            targets,
            station_available,
            mask_ratio=config.mask_ratio,
            mask_strategy=config.mask_strategy,
            clean_window_ratio=config.clean_window_ratio,
        )

        optimizer.zero_grad()
        local_preds, cross_preds = model(masked_inputs, targets, masked_positions, station_available)
        loss, metrics = compute_pretraining_loss(
            local_preds,
            cross_preds,
            targets,
            station_available,
            masked_positions,
            local_loss_weight=config.local_loss_weight,
            cross_all_loss_weight=config.cross_all_loss_weight,
            cross_masked_loss_weight=config.cross_masked_loss_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        total_loss += metrics["loss"]
        total_local_mse += metrics["local_mse"]
        total_cross_all_mse += metrics["cross_all_mse"]
        total_cross_masked_mse += metrics["cross_masked_mse"]
        total_masked_fraction += metrics["masked_fraction"]

    num_batches = max(
        1,
        min(len(loader), config.max_train_batches or len(loader)),
    )
    return {
        "loss": total_loss / num_batches,
        "local_mse": total_local_mse / num_batches,
        "cross_all_mse": total_cross_all_mse / num_batches,
        "cross_masked_mse": total_cross_masked_mse / num_batches,
        "masked_fraction": total_masked_fraction / num_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device, config, seed):
    model.eval()
    total_loss = 0.0
    total_local_mse = 0.0
    total_cross_all_mse = 0.0
    total_cross_masked_mse = 0.0
    total_masked_fraction = 0.0

    for batch_idx, (targets, station_available, sample_indices) in enumerate(loader):
        if config.max_val_batches is not None and batch_idx >= config.max_val_batches:
            break
        targets = targets.to(device)
        station_available = station_available.to(device)
        sample_indices = sample_indices.to(device)

        masked_inputs, masked_positions = build_masked_inputs(
            targets,
            station_available,
            mask_ratio=config.mask_ratio,
            mask_strategy=config.mask_strategy,
            clean_window_ratio=config.clean_window_ratio,
            sample_indices=sample_indices,
            base_seed=seed * 100_000,
        )

        local_preds, cross_preds = model(masked_inputs, targets, masked_positions, station_available)
        _, metrics = compute_pretraining_loss(
            local_preds,
            cross_preds,
            targets,
            station_available,
            masked_positions,
            local_loss_weight=config.local_loss_weight,
            cross_all_loss_weight=config.cross_all_loss_weight,
            cross_masked_loss_weight=config.cross_masked_loss_weight,
        )
        total_loss += metrics["loss"]
        total_local_mse += metrics["local_mse"]
        total_cross_all_mse += metrics["cross_all_mse"]
        total_cross_masked_mse += metrics["cross_masked_mse"]
        total_masked_fraction += metrics["masked_fraction"]

    num_batches = max(
        1,
        min(len(loader), config.max_val_batches or len(loader)),
    )
    return {
        "loss": total_loss / num_batches,
        "local_mse": total_local_mse / num_batches,
        "cross_all_mse": total_cross_all_mse / num_batches,
        "cross_masked_mse": total_cross_masked_mse / num_batches,
        "masked_fraction": total_masked_fraction / num_batches,
    }


def export_merged_compatible_state_dict(backbone_state, num_stations, base_seq_len, base_prompt_num):
    merged = {key: value.detach().cpu().clone() for key, value in backbone_state.items()}

    for station_idx in range(num_stations):
        for key, value in backbone_state.items():
            mapped_value = value.detach().cpu().clone()
            if key == "embedding.embed.weight" and mapped_value.shape[1] != base_seq_len:
                resized = F.interpolate(
                    mapped_value.float().unsqueeze(1),
                    size=base_seq_len,
                    mode="linear",
                    align_corners=False,
                )
                mapped_value = resized.squeeze(1).to(dtype=value.dtype)
            merged[f"transformers.{station_idx}.{key}"] = mapped_value

    merged["mlp.weight"] = torch.full(
        (base_prompt_num, num_stations),
        1.0 / float(num_stations),
        dtype=torch.float32,
    )
    merged["mlp.bias"] = torch.zeros(base_prompt_num, dtype=torch.float32)
    return merged


def maybe_mirror_run(source_dir, target_root):
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    target_dir = target_root / source_dir.name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    return target_dir


def save_run_artifacts(
    run_dir,
    merged_state,
    backbone_state,
    pretrainer_state,
    scaler,
    history,
    config,
    station_names,
    raw_lengths,
    train_summary,
    val_summary,
    best_stats,
    evaluation_summary,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(merged_state, run_dir / "model.pth")
    torch.save(backbone_state, run_dir / "backbone_only.pth")
    torch.save(pretrainer_state, run_dir / "pretrainer_full_state.pth")
    with open(run_dir / "scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    backbone_name = normalize_backbone_name(config.backbone_name)
    is_transformer = backbone_name == "transformer"
    config_payload = {
        **asdict(config),
        "feature_columns": list(config.feature_columns),
        "device_name": str(config.device_name),
        "station_names": station_names,
        "num_stations": len(station_names),
        "raw_lengths": raw_lengths,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "best_epoch": best_stats["epoch"],
        "best_val_loss": best_stats["val_loss"],
        "best_val_local_mse": best_stats["val_local_mse"],
        "best_val_cross_all_mse": best_stats["val_cross_all_mse"],
        "best_val_cross_masked_mse": best_stats["val_cross_masked_mse"],
        "checkpoint_format": (
            "merged_ptl_base"
            if is_transformer
            else (
                "unified_feature_token_encoder"
                if normalize_model_agnostic_interface(config.model_agnostic_interface)
                != "legacy"
                else "forecast_backbone_encoder"
            )
        ),
        "ptl_compatible": is_transformer,
        "base_compatible": is_transformer,
        "progressive_finetune_compatible": True,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config_payload, file, ensure_ascii=False, indent=2)
    with open(run_dir / "evaluation.json", "w", encoding="utf-8") as file:
        json.dump(evaluation_summary, file, ensure_ascii=False, indent=2)


def build_pretraining_backbone(config):
    backbone_name = normalize_backbone_name(config.backbone_name)
    if backbone_name == "transformer":
        return WaterQualityTransformer(
            num_heads=config.num_heads,
            e_layer=config.e_layer,
            hidden_size=config.hidden_size,
            input_dim=config.input_dim,
            target_dim=config.input_dim,
            target_feature_names=list(config.feature_columns),
            seq_len=config.model_seq_len,
            pred_len=config.model_seq_len,
            use_temporal_adapter=True,
            temporal_adapter_kernel_size=5,
        )

    interface = normalize_model_agnostic_interface(config.model_agnostic_interface)
    if interface in {"feature_token_v1", "feature_token_residual_v2"}:
        return build_model_agnostic_forecaster(
            backbone_name=backbone_name,
            input_dim=config.input_dim,
            seq_len=config.model_seq_len,
            pred_len=1,
            target_dim=config.input_dim,
            config=config,
            interface=interface,
            reconstruction_len=config.model_seq_len,
        )
    forecast_model = build_temporal_forecasting_backbone(
        backbone_name=backbone_name,
        input_dim=config.input_dim,
        seq_len=config.model_seq_len,
        pred_len=1,
        target_dim=config.input_dim,
        config=config,
    )
    return TemporalFeatureTokenPretrainAdapter(
        forecast_model=forecast_model,
        backbone_name=backbone_name,
        target_dim=config.input_dim,
        reconstruction_len=config.model_seq_len,
    )


def export_progressive_finetune_state(backbone_state, config, num_stations):
    backbone_name = normalize_backbone_name(config.backbone_name)
    if backbone_name == "transformer":
        return export_merged_compatible_state_dict(
            backbone_state,
            num_stations=num_stations,
            base_seq_len=config.base_export_seq_len,
            base_prompt_num=config.base_export_prompt_num,
        )
    if normalize_model_agnostic_interface(config.model_agnostic_interface) != "legacy":
        return {
            key: value.detach().cpu().clone()
            for key, value in backbone_state.items()
            if not key.startswith("head.")
            and not key.startswith("reconstruction_head.")
            and not key.startswith("forecast_model.head.")
        }
    prefix = "forecast_model."
    return {
        key[len(prefix):]: value.detach().cpu().clone()
        for key, value in backbone_state.items()
        if key.startswith(prefix) and not key[len(prefix):].startswith("head.")
    }


def main(seed=42, cli_args=None):
    args = cli_args or parse_args()
    set_seed(seed)

    config = CrossStationPretrainConfig()
    config.backbone_name = normalize_backbone_name(getattr(args, "backbone", "transformer"))
    config.model_agnostic_interface = normalize_model_agnostic_interface(
        getattr(args, "model_agnostic_interface", "legacy")
    )
    if args.epochs is not None:
        config.pretrain_epochs = int(args.epochs)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.hidden_size is not None:
        config.hidden_size = int(args.hidden_size)
    if args.num_heads is not None:
        config.num_heads = int(args.num_heads)
    if args.e_layer is not None:
        config.e_layer = int(args.e_layer)
    if args.cross_station_heads is not None:
        config.cross_station_heads = int(args.cross_station_heads)
    if args.cross_station_layers is not None:
        config.cross_station_layers = int(args.cross_station_layers)
    if args.max_stations is not None:
        config.max_stations = int(args.max_stations)
    if args.max_train_windows is not None:
        config.max_train_windows = int(args.max_train_windows)
    if args.max_val_windows is not None:
        config.max_val_windows = int(args.max_val_windows)
    if args.max_train_batches is not None:
        config.max_train_batches = int(args.max_train_batches)
    if args.max_val_batches is not None:
        config.max_val_batches = int(args.max_val_batches)
    if args.raw_seq_len is not None:
        config.raw_seq_len = int(args.raw_seq_len)
    if args.model_seq_len is not None:
        config.model_seq_len = int(args.model_seq_len)
    if args.base_lr is not None:
        config.base_lr = float(args.base_lr)
    if args.mask_strategy is not None:
        config.mask_strategy = str(args.mask_strategy)
    if args.mask_ratio is not None:
        config.mask_ratio = float(args.mask_ratio)
    if args.clean_window_ratio is not None:
        config.clean_window_ratio = float(args.clean_window_ratio)
    if args.local_loss_weight is not None:
        config.local_loss_weight = float(args.local_loss_weight)
    if args.cross_all_loss_weight is not None:
        config.cross_all_loss_weight = float(args.cross_all_loss_weight)
    if args.cross_masked_loss_weight is not None:
        config.cross_masked_loss_weight = float(args.cross_masked_loss_weight)
    if args.save_root is not None:
        config.save_root = args.save_root
    if args.mirror_compatible_runs:
        config.mirror_compatible_runs = True
    config.device_name = str(infer_device())
    device = torch.device(config.device_name)

    print("=" * 80)
    print("Cross-station compatible pretraining")
    print("=" * 80)
    print(
        f"seed={seed} backbone={config.backbone_name} basin={config.basin_key} "
        f"interface={config.model_agnostic_interface} "
        f"raw_seq_len={config.raw_seq_len} model_seq_len={config.model_seq_len}"
    )

    scaled_series, station_names, raw_lengths, scaler = load_station_series(config)
    train_dataset = CrossStationMaskedDataset(
        scaled_series,
        station_names=station_names,
        raw_seq_len=config.raw_seq_len,
        model_seq_len=config.model_seq_len,
        split="train",
        train_ratio=config.train_ratio,
        min_available_stations=config.min_available_stations,
        max_windows=config.max_train_windows,
    )
    val_dataset = CrossStationMaskedDataset(
        scaled_series,
        station_names=station_names,
        raw_seq_len=config.raw_seq_len,
        model_seq_len=config.model_seq_len,
        split="val",
        train_ratio=config.train_ratio,
        min_available_stations=config.min_available_stations,
        max_windows=config.max_val_windows,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Cross-station dataset is empty; relax window length or availability thresholds.")

    train_summary = train_dataset.summary()
    val_summary = val_dataset.summary()
    print(f"stations={len(station_names)} train_windows={len(train_dataset)} val_windows={len(val_dataset)}")
    print(f"train_summary={train_summary}")
    print(f"val_summary={val_summary}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    backbone = build_pretraining_backbone(config)
    model = CrossStationBackbonePretrainer(
        backbone=backbone,
        num_stations=len(station_names),
        cross_station_heads=config.cross_station_heads,
        cross_station_layers=config.cross_station_layers,
        dropout=config.cross_station_dropout,
        station_identity_enabled=config.station_identity_enabled,
        station_embedding_init_std=config.station_embedding_init_std,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.base_lr,
        eps=config.epsilon,
        weight_decay=config.weight_decay,
        amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_decay_ratio,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    history = []
    best_stats = {
        "epoch": 0,
        "val_loss": float("inf"),
        "val_local_mse": float("inf"),
        "val_cross_all_mse": float("inf"),
        "val_cross_masked_mse": float("inf"),
    }
    best_state = snapshot_state_dict(model.backbone)
    best_pretrainer_state = snapshot_state_dict(model)
    patience_counter = 0
    train_start = time.time()

    for epoch in range(config.pretrain_epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            config=config,
        )
        val_stats = evaluate(
            model,
            val_loader,
            device=device,
            config=config,
            seed=seed,
        )
        scheduler.step(val_stats["loss"])

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "train_local_mse": train_stats["local_mse"],
            "train_cross_all_mse": train_stats["cross_all_mse"],
            "train_cross_masked_mse": train_stats["cross_masked_mse"],
            "train_masked_fraction": train_stats["masked_fraction"],
            "val_loss": val_stats["loss"],
            "val_local_mse": val_stats["local_mse"],
            "val_cross_all_mse": val_stats["cross_all_mse"],
            "val_cross_masked_mse": val_stats["cross_masked_mse"],
            "val_masked_fraction": val_stats["masked_fraction"],
            "clean_window_ratio": config.clean_window_ratio,
            "mask_ratio": config.mask_ratio,
            "mask_strategy": config.mask_strategy,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(history_entry)

        print(
            f"[Epoch {epoch + 1:03d}] "
            f"train_loss={train_stats['loss']:.6f} "
            f"val_loss={val_stats['loss']:.6f} "
            f"local={val_stats['local_mse']:.6f} "
            f"cross_all={val_stats['cross_all_mse']:.6f} "
            f"cross_masked={val_stats['cross_masked_mse']:.6f} "
            f"mask={config.mask_strategy}@{config.mask_ratio:.2f} "
            f"clean={config.clean_window_ratio:.2f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        improved = val_stats["loss"] < (best_stats["val_loss"] - config.early_stopping_min_delta)
        if improved:
            best_stats = {
                "epoch": epoch + 1,
                "val_loss": val_stats["loss"],
                "val_local_mse": val_stats["local_mse"],
                "val_cross_all_mse": val_stats["cross_all_mse"],
                "val_cross_masked_mse": val_stats["cross_masked_mse"],
            }
            best_state = snapshot_state_dict(model.backbone)
            best_pretrainer_state = snapshot_state_dict(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    train_seconds = time.time() - train_start
    merged_state = export_progressive_finetune_state(
        best_state,
        config,
        num_stations=len(station_names),
    )
    saved_backbone_state = (
        best_state
        if config.backbone_name == "transformer"
        else merged_state
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.run_prefix}{seed}_{timestamp}"
    run_dir = Path(config.save_root) / run_name
    best_stats["train_seconds"] = train_seconds
    model.load_state_dict(best_pretrainer_state, strict=True)
    evaluation_summary = summarize_checkpoint(
        model,
        val_loader,
        device=device,
        config=config,
        seed=seed,
        scaler=scaler,
        station_names=station_names,
    )
    save_run_artifacts(
        run_dir=run_dir,
        merged_state=merged_state,
        backbone_state=saved_backbone_state,
        pretrainer_state=best_pretrainer_state,
        scaler=scaler,
        history=history,
        config=config,
        station_names=station_names,
        raw_lengths=raw_lengths,
        train_summary=train_summary,
        val_summary=val_summary,
        best_stats=best_stats,
        evaluation_summary=evaluation_summary,
    )

    mirrored_dirs = []
    if config.mirror_compatible_runs:
        mirrored_dirs.append(str(maybe_mirror_run(run_dir, config.ptl_compat_root)))
        mirrored_dirs.append(str(maybe_mirror_run(run_dir, config.base_compat_root)))

    print("=" * 80)
    print(
        f"done best_epoch={best_stats['epoch']} "
        f"best_val_loss={best_stats['val_loss']:.6f} "
        f"best_raw_rmse={evaluation_summary['cross_raw_metrics']['rmse']:.6f} "
        f"run_dir={run_dir}"
    )
    if mirrored_dirs:
        print(f"mirrored_dirs={mirrored_dirs}")
    return best_stats["val_loss"], str(run_dir)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(seed=parsed_args.seed, cli_args=parsed_args)
