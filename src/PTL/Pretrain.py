import datetime
import json
import os
import pickle
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from progressive_core import (
    MaskedReconstructionDataset,
    WaterQualityTransformer,
    fit_model,
    fit_scaler_on_train_slices,
    infer_device,
    set_seed,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "results", "ptl")
PRETRAIN_RUNS_DIR = os.path.join(ARTIFACTS_ROOT, "pretrain", "runs")
PRETRAIN_FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]


class PretrainConfig:
    def __init__(self):
        self.basin_data_root = os.path.join(REPO_ROOT, "data", "data_cleaned")
        self.basin_key = "yangzte"
        self.river_name = "长江"
        self.save_dir = PRETRAIN_RUNS_DIR

        self.selected_stations = None
        self.max_stations = None
        self.min_series_length = 200
        self.max_train_windows_per_station = None
        self.max_val_windows_per_station = None

        self.model_seq_len = 168
        self.raw_seq_len = 168
        self.input_dim = len(PRETRAIN_FEATURE_COLUMNS)
        self.feature_columns = list(PRETRAIN_FEATURE_COLUMNS)
        self.hidden_size = 256
        self.num_heads = 8
        self.e_layer = 3

        self.batch_size = 64
        self.pretrain_epochs = 100
        self.base_lr = 1e-3
        self.epsilon = 1e-8
        self.weight_decay = 0.0
        self.train_ratio = 0.8
        self.mask_ratio = 0.7
        self.lr_milestones = [30, 60, 90]
        self.lr_decay_ratio = 0.5
        self.max_grad_norm = 1.0

        self.device = infer_device()
        os.makedirs(self.save_dir, exist_ok=True)


def load_pretrain_series(config):
    basin_dir = os.path.join(
        config.basin_data_root,
        config.basin_key,
    )
    if not os.path.isdir(basin_dir):
        raise ValueError(f"流域目录不存在: {basin_dir}")

    station_files = sorted(
        filename
        for filename in os.listdir(basin_dir)
        if filename.endswith(".csv") and not filename.startswith(".")
    )

    if config.selected_stations:
        selected_names = set(config.selected_stations)
        station_files = [
            filename
            for filename in station_files
            if filename in selected_names or os.path.splitext(filename)[0] in selected_names
        ]

    loaded = []
    for filename in station_files:
        frame = pd.read_csv(os.path.join(basin_dir, filename))
        if not set(PRETRAIN_FEATURE_COLUMNS).issubset(frame.columns):
            continue

        values = frame[PRETRAIN_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        values = values.interpolate(limit_direction="both").dropna().to_numpy(dtype="float32", copy=True)
        if len(values) < max(config.min_series_length, config.raw_seq_len + 1):
            continue

        loaded.append((os.path.splitext(filename)[0], values))

    loaded.sort(key=lambda item: len(item[1]), reverse=True)
    if config.max_stations is not None:
        loaded = loaded[:config.max_stations]

    if not loaded:
        raise ValueError("没有可用于预训练的站点序列。")

    scaler = fit_scaler_on_train_slices([values for _, values in loaded], config.train_ratio)
    scaled_series = [scaler.transform(values).astype("float32") for _, values in loaded]
    station_names = [name for name, _ in loaded]

    train_dataset = MaskedReconstructionDataset(
        scaled_series,
        raw_seq_len=config.raw_seq_len,
        model_seq_len=config.model_seq_len,
        mask_ratio=config.mask_ratio,
        split="train",
        train_ratio=config.train_ratio,
        max_windows_per_series=config.max_train_windows_per_station,
    )
    val_dataset = MaskedReconstructionDataset(
        scaled_series,
        raw_seq_len=config.raw_seq_len,
        model_seq_len=config.model_seq_len,
        mask_ratio=config.mask_ratio,
        split="val",
        train_ratio=config.train_ratio,
        max_windows_per_series=config.max_val_windows_per_station,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("预训练窗口为空，请降低窗口长度或检查数据长度。")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, scaler, station_names


def main(seed=42):
    set_seed(seed)
    config = PretrainConfig()

    print(f"\n{'=' * 70}")
    print(
        f"Stage 0 预训练开始 | seed={seed} | basin={config.river_name} "
        f"({config.basin_key}) | weekly data"
    )
    print(f"{'=' * 70}")

    train_loader, val_loader, scaler, station_names = load_pretrain_series(config)
    print(
        f"载入站点数: {len(station_names)} | "
        f"train windows: {len(train_loader.dataset)} | val windows: {len(val_loader.dataset)}"
    )

    model = WaterQualityTransformer(
        num_heads=config.num_heads,
        e_layer=config.e_layer,
        hidden_size=config.hidden_size,
        input_dim=config.input_dim,
        seq_len=config.model_seq_len,
        pred_len=config.model_seq_len,
    )

    train_start = time.time()
    model, history, best_stats = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.device,
        epochs=config.pretrain_epochs,
        base_lr=config.base_lr,
        epsilon=config.epsilon,
        weight_decay=config.weight_decay,
        lr_milestones=config.lr_milestones,
        lr_decay_ratio=config.lr_decay_ratio,
        max_grad_norm=config.max_grad_norm,
        log_prefix="Pretrain",
    )
    train_seconds = time.time() - train_start
    best_val_loss = best_stats["val_loss"]
    best_epoch = best_stats["epoch"]
    best_val_nse = best_stats["val_nse"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        config.save_dir,
        f"pretrain_{config.basin_key}_weekly_seed{seed}_{timestamp}",
    )
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

    config_dict = {
        **vars(config),
        "device": str(config.device),
        "station_names": station_names,
        "num_stations": len(station_names),
        "best_val_loss": best_val_loss,
        "best_val_nse": best_val_nse,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
    }
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config_dict, file, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as file:
        pickle.dump(scaler, file)

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "history.csv"), index=False)

    print(f"\n{'=' * 70}")
    print(
        f"Stage 0 完成 | best_epoch={best_epoch} | best_val_loss={best_val_loss:.6f} | "
        f"save_dir={save_dir}"
    )
    return best_val_loss, save_dir


if __name__ == "__main__":
    main(seed=42)
