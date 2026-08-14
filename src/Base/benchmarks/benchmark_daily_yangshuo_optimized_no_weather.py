from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
TRAINING_DIR = BASE_DIR / "training"
REPO_ROOT = SCRIPT_DIR.parents[2]


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    spec.loader.exec_module(module)
    return module


BASE_OPT = load_module("base_finetune_optimized_module", TRAINING_DIR / "finetune_optimized.py")
PTL_CORE = load_module("ptl_progressive_core_baseopt_module", REPO_ROOT / "src" / "PTL" / "progressive_core.py")


FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "water_quality_processed_2023_2025" / "daily" / "阳朔.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "base" / "finetune" / "optimized_daily_benchmarks"
DEFAULT_PTL_REFERENCE_DIR = (
    REPO_ROOT
    / "results"
    / "ptl"
    / "finetune"
    / "runs"
    / "progressive_阳朔_seed42_20260408_185725"
    / "stage3_daily"
)


class DailyBenchmarkDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        raw_seq_len: int,
        model_seq_len: int,
        raw_pred_len: int,
        model_pred_len: int,
        split: str,
        train_ratio: float,
        val_ratio: float,
        resize_mode: str = "linear",
    ):
        train_end, val_end = PTL_CORE.compute_split_points(len(values), train_ratio, val_ratio)

        if split == "train":
            sliced_values = values[:train_end]
            sliced_timestamps = timestamps[:train_end]
        elif split == "val":
            start = max(0, train_end - raw_seq_len)
            sliced_values = values[start:val_end]
            sliced_timestamps = timestamps[start:val_end]
        elif split == "test":
            start = max(0, val_end - raw_seq_len)
            sliced_values = values[start:]
            sliced_timestamps = timestamps[start:]
        else:
            raise ValueError(f"未知 split: {split}")

        self.values = sliced_values.astype(np.float32, copy=False)
        self.timestamps = sliced_timestamps.astype("datetime64[ns]")
        self.raw_seq_len = int(raw_seq_len)
        self.model_seq_len = int(model_seq_len)
        self.raw_pred_len = int(raw_pred_len)
        self.model_pred_len = int(model_pred_len)
        self.resize_mode = resize_mode
        self.window_count = max(0, len(self.values) - self.raw_seq_len - self.raw_pred_len + 1)

    def __len__(self):
        return self.window_count

    def __getitem__(self, index):
        start = int(index)
        mid = start + self.raw_seq_len
        end = mid + self.raw_pred_len

        water_x = PTL_CORE.resize_sequence_length(
            self.values[start:mid],
            target_len=self.model_seq_len,
            mode=self.resize_mode,
        )
        weather_x = np.empty((self.model_seq_len, 0), dtype=np.float32)
        water_y = PTL_CORE.resize_sequence_length(
            self.values[mid:end],
            target_len=self.model_pred_len,
            mode="linear",
        )
        target_times = self.timestamps[mid:end].astype(np.int64)

        return (
            torch.as_tensor(water_x, dtype=torch.float32),
            torch.as_tensor(weather_x, dtype=torch.float32),
            torch.as_tensor(water_y, dtype=torch.float32),
            torch.as_tensor(target_times, dtype=torch.int64),
        )


def read_station_frame(data_path: Path):
    frame = pd.read_csv(data_path)
    required_columns = {"timestamp", *FEATURE_COLUMNS}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"数据缺少必要列: {sorted(required_columns - set(frame.columns))}")
    frame = frame[["timestamp", *FEATURE_COLUMNS]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("阳朔日级数据存在 NaN，当前脚本要求完整序列。")
    return frame.reset_index(drop=True)


def build_scaled_series(frame: pd.DataFrame, train_ratio: float, val_ratio: float):
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    timestamps = frame["timestamp"].to_numpy(dtype="datetime64[ns]")
    train_end, val_end = PTL_CORE.compute_split_points(len(values), train_ratio, val_ratio)

    scaler = PTL_CORE.StandardScaler()
    scaler.fit(values[:train_end])
    scaled_values = scaler.transform(values).astype(np.float32)

    return {
        "timestamps": timestamps,
        "raw_values": values,
        "scaled_values": scaled_values,
        "scaler": scaler,
        "train_end": int(train_end),
        "val_end": int(val_end),
    }


def build_loaders(
    scaled_values: np.ndarray,
    timestamps: np.ndarray,
    raw_seq_len: int,
    model_seq_len: int,
    raw_pred_len: int,
    model_pred_len: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    resize_mode: str = "linear",
):
    datasets = {}
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = DailyBenchmarkDataset(
            values=scaled_values,
            timestamps=timestamps,
            raw_seq_len=raw_seq_len,
            model_seq_len=model_seq_len,
            raw_pred_len=raw_pred_len,
            model_pred_len=model_pred_len,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            resize_mode=resize_mode,
        )
        datasets[split] = dataset
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
    return datasets, loaders


def snapshot_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_pretrain_metadata(pretrain_dir: Path):
    config_path = pretrain_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"缺少预训练配置: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_model(pretrain_config, device):
    config = BASE_OPT.FinetuneConfig()
    config.device = device
    config.input_dim = len(FEATURE_COLUMNS)
    config.feature_dim = 0
    config.n_in = int(pretrain_config["n_in"])
    config.n_out = 1
    config.hidden_size = int(pretrain_config["hidden_size"])
    config.num_heads = int(pretrain_config["num_heads"])
    config.e_layer = int(pretrain_config["e_layer"])
    config.prompt_num = 1
    return config, BASE_OPT.Prompt_MultiTransformer(
        config,
        num_pretrain_stations=int(pretrain_config["num_stations"]),
    )


def load_matching_pretrain_weights(model, pretrain_dir: Path, device):
    model_path = pretrain_dir / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"缺少预训练权重: {model_path}")

    pretrained_state = torch.load(model_path, map_location=device)
    current_state = model.state_dict()
    matched_state = {
        key: value
        for key, value in pretrained_state.items()
        if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
    }
    current_state.update(matched_state)
    model.load_state_dict(current_state, strict=False)
    return matched_state


def apply_optimized_freeze_strategy(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_keywords = ["prompt", "memory", "mlp", "fusion_layer", "combined_attention"]
    for name, parameter in model.named_parameters():
        if any(keyword in name for keyword in trainable_keywords):
            parameter.requires_grad = True
        if name.startswith("transformers"):
            if "head" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_ratio": float(trainable_params / max(1, total_params)),
    }


def evaluate_model(model, loader, device, scaler=None):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    timestamps = []

    with torch.no_grad():
        for water_x, weather_x, water_y, target_times in loader:
            water_x = water_x.to(device)
            weather_x = weather_x.to(device)
            water_y = water_y.to(device)

            output = model(water_x, weather_x)
            if output.shape != water_y.shape:
                water_y = water_y.view_as(output)

            loss = criterion(output, water_y)
            total_loss += float(loss.item())

            pred_np = output.detach().cpu().numpy()
            target_np = water_y.detach().cpu().numpy()
            if scaler is not None:
                pred_np = scaler.inverse_transform(pred_np)
                target_np = scaler.inverse_transform(target_np)

            preds.append(pred_np)
            targets.append(target_np)
            timestamps.append(target_times.detach().cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    if not preds:
        return avg_loss, np.empty((0, 0, 0)), np.empty((0, 0, 0)), np.empty((0, 0), dtype=np.int64)

    return (
        avg_loss,
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(timestamps, axis=0),
    )


def train_model(model, train_loader, val_loader, device, args):
    model = model.to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.base_lr,
        eps=args.epsilon,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_decay_ratio,
    )
    criterion = nn.MSELoss()

    history = []
    best_state = snapshot_state_dict(model)
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_nse = float("-inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0

        for water_x, weather_x, water_y, _ in train_loader:
            water_x = water_x.to(device)
            weather_x = weather_x.to(device)
            water_y = water_y.to(device)

            optimizer.zero_grad()
            output = model(water_x, weather_x)
            if output.shape != water_y.shape:
                water_y = water_y.view_as(output)

            loss = criterion(output, water_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss_sum += float(loss.item())

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, len(train_loader))
        val_loss, val_preds, val_targets, _ = evaluate_model(model, val_loader, device, scaler=None)
        val_metrics = PTL_CORE.compute_per_feature_metrics(
            val_preds,
            val_targets,
            feature_names=FEATURE_COLUMNS,
        )
        val_nse = float(val_metrics["__overall__"]["NSE"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_nse": val_nse,
                "lr": current_lr,
            }
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[BaseOptimizedNoWeather] Epoch {epoch + 1}/{args.epochs} "
                f"| train_loss={avg_train_loss:.6f} "
                f"| val_loss={val_loss:.6f} "
                f"| val_nse={val_nse:.6f}"
            )

        improved = val_loss < (best_val_loss - args.early_stopping_min_delta)
        if improved:
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_val_nse = val_nse
            best_state = snapshot_state_dict(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if (
            args.early_stopping_patience is not None
            and args.early_stopping_patience > 0
            and patience_counter >= args.early_stopping_patience
        ):
            print(
                f"[BaseOptimizedNoWeather] Early stopping at epoch {epoch + 1} "
                f"| best_epoch={best_epoch} "
                f"| best_val_loss={best_val_loss:.6f} "
                f"| best_val_nse={best_val_nse:.6f}"
            )
            break

    model.load_state_dict(best_state)
    return model, history, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_nse": float(best_val_nse),
    }


def save_predictions(path: Path, preds, targets, timestamps):
    prediction_frame = pd.DataFrame({"timestamp": pd.to_datetime(timestamps.reshape(-1))})
    preds_flat = preds.reshape(-1, preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])
    for index, feature_name in enumerate(FEATURE_COLUMNS):
        prediction_frame[f"True_{feature_name}"] = targets_flat[:, index]
        prediction_frame[f"Pred_{feature_name}"] = preds_flat[:, index]
    prediction_frame.to_csv(path, index=False)


def build_focus_summary(metrics, focus_features: list[str]):
    summary = {}
    nse_values = [float(metrics[name]["NSE"]) for name in focus_features if name in metrics]
    rmse_values = [float(metrics[name]["RMSE"]) for name in focus_features if name in metrics]
    mae_values = [float(metrics[name]["MAE"]) for name in focus_features if name in metrics]
    if nse_values:
        summary["mean_nse"] = float(np.mean(nse_values))
    if rmse_values:
        summary["mean_rmse"] = float(np.mean(rmse_values))
    if mae_values:
        summary["mean_mae"] = float(np.mean(mae_values))
    summary["features"] = list(focus_features)
    return summary


def load_reference_run(reference_dir: Path):
    metric_candidates = [
        reference_dir / "metrics.csv",
        reference_dir / "评估指标_metrics.csv",
    ]
    meta_candidates = [
        reference_dir / "meta.json",
        reference_dir / "运行元信息_meta.json",
    ]

    metrics = None
    for metric_path in metric_candidates:
        if metric_path.exists():
            metrics = pd.read_csv(metric_path, index_col=0).replace({np.nan: None}).to_dict(orient="index")
            break

    meta = None
    for meta_path in meta_candidates:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            break

    return {
        "reference_dir": str(reference_dir),
        "metrics": metrics,
        "meta": meta,
    }


def build_comparison_summary(base_metrics, reference_metrics):
    summary = {
        "baseline_overall": base_metrics.get("__overall__"),
        "reference_overall": reference_metrics.get("__overall__") if reference_metrics else None,
        "baseline_focus": build_focus_summary(base_metrics, FOCUS_FEATURES),
        "reference_focus": build_focus_summary(reference_metrics, FOCUS_FEATURES) if reference_metrics else None,
        "per_feature_delta_baseline_minus_reference": {},
    }
    if reference_metrics:
        for feature_name in FEATURE_COLUMNS:
            if feature_name not in base_metrics or feature_name not in reference_metrics:
                continue
            summary["per_feature_delta_baseline_minus_reference"][feature_name] = {
                key: float(base_metrics[feature_name][key]) - float(reference_metrics[feature_name][key])
                for key in ("MAE", "RMSE", "NSE", "MAPE")
                if base_metrics[feature_name].get(key) is not None and reference_metrics[feature_name].get(key) is not None
            }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="finetune_optimized 无气象版阳朔日预测 benchmark")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pretrain-dir", type=Path, default=None)
    parser.add_argument("--ptl-reference-dir", type=Path, default=DEFAULT_PTL_REFERENCE_DIR)
    parser.add_argument("--station-name", type=str, default="阳朔")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--raw-seq-len", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--base-lr", type=float, default=1e-2)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=[40, 60, 80])
    parser.add_argument("--lr-decay-ratio", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--resize-mode", type=str, default="linear")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    pretrain_dir = args.pretrain_dir
    if pretrain_dir is None:
        latest_pretrain_dir = BASE_OPT.find_latest_pretrain_run(BASE_OPT.PRETRAIN_RUNS_DIR)
        if latest_pretrain_dir is None:
            raise FileNotFoundError("未找到可用的 Base 预训练目录。")
        pretrain_dir = Path(latest_pretrain_dir)

    PTL_CORE.set_seed(args.seed)
    device = PTL_CORE.infer_device()

    frame = read_station_frame(args.data_path)
    data_bundle = build_scaled_series(frame, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    pretrain_config = load_pretrain_metadata(pretrain_dir)
    model_seq_len = int(pretrain_config["n_in"])
    raw_seq_len = int(args.raw_seq_len) if args.raw_seq_len is not None else model_seq_len

    datasets, loaders = build_loaders(
        scaled_values=data_bundle["scaled_values"],
        timestamps=data_bundle["timestamps"],
        raw_seq_len=raw_seq_len,
        model_seq_len=model_seq_len,
        raw_pred_len=args.pred_len,
        model_pred_len=args.pred_len,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        resize_mode=args.resize_mode,
    )

    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("train/val/test 至少有一个切分没有可用窗口。")

    config, model = build_model(pretrain_config, device=device)
    matched_state = load_matching_pretrain_weights(model, pretrain_dir=pretrain_dir, device=device)
    freeze_stats = apply_optimized_freeze_strategy(model)

    print("=" * 68)
    print("Base finetune_optimized No-Weather Yangshuo Benchmark")
    print("=" * 68)
    print(f"数据文件: {args.data_path}")
    print(f"PTL 对照目录: {args.ptl_reference_dir}")
    print(f"预训练目录: {pretrain_dir}")
    print(f"站点: {args.station_name}")
    print(f"特征: {FEATURE_COLUMNS}")
    print(f"focus_features: {FOCUS_FEATURES}")
    print(
        f"切分: train/val/test = {args.train_ratio:.2f}/{args.val_ratio:.2f}/{1.0 - args.train_ratio - args.val_ratio:.2f}"
    )
    print(f"窗口: raw_seq_len={raw_seq_len}, model_seq_len={model_seq_len}, pred_len={args.pred_len}")
    print(
        f"样本数: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}"
    )
    print(f"迁移权重数: {len(matched_state)}")
    print(
        f"可训练参数: {freeze_stats['trainable_params']:,} / {freeze_stats['total_params']:,} "
        f"({freeze_stats['trainable_ratio']:.3%})"
    )

    start_time = pd.Timestamp.now()
    model, history, best_info = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        args=args,
    )
    train_seconds = float((pd.Timestamp.now() - start_time).total_seconds())

    test_loss, preds, targets, timestamps = evaluate_model(
        model,
        loaders["test"],
        device=device,
        scaler=data_bundle["scaler"],
    )
    metrics = PTL_CORE.compute_per_feature_metrics(preds, targets, feature_names=FEATURE_COLUMNS)
    reference_run = load_reference_run(args.ptl_reference_dir)
    comparison_summary = build_comparison_summary(metrics, reference_run.get("metrics"))

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"base_optimized_noweather_{args.station_name}_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    model_path = run_dir / "model.pth"
    model_weights_saved = BASE_OPT.should_save_finetune_model_weights()
    if model_weights_saved:
        torch.save(model.state_dict(), model_path)
    else:
        print(f"跳过保存模型权重: {model_path}")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(run_dir / "metrics.csv")
    save_predictions(run_dir / "predictions.csv", preds, targets, timestamps)

    meta = {
        "station_name": args.station_name,
        "feature_columns": FEATURE_COLUMNS,
        "focus_features": FOCUS_FEATURES,
        "data_path": str(args.data_path),
        "pretrain_dir": str(pretrain_dir),
        "ptl_reference_dir": str(args.ptl_reference_dir),
        "device": str(device),
        "seed": int(args.seed),
        "records": int(len(frame)),
        "train_end": data_bundle["train_end"],
        "val_end": data_bundle["val_end"],
        "raw_seq_len": int(raw_seq_len),
        "model_seq_len": int(model_seq_len),
        "pred_len": int(args.pred_len),
        "resize_mode": args.resize_mode,
        "train_windows": int(len(datasets["train"])),
        "val_windows": int(len(datasets["val"])),
        "test_windows": int(len(datasets["test"])),
        "loaded_pretrain_keys": int(len(matched_state)),
        "freeze_stats": freeze_stats,
        "model_weights_saved": model_weights_saved,
        "model_weights_path": str(model_path) if model_weights_saved else None,
        "best_epoch": best_info["best_epoch"],
        "best_val_loss": best_info["best_val_loss"],
        "best_val_nse": best_info["best_val_nse"],
        "test_loss": float(test_loss),
        "train_seconds": train_seconds,
        "focus_summary": build_focus_summary(metrics, FOCUS_FEATURES),
        "metrics": metrics["__overall__"],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "reference_run.json").write_text(
        json.dumps(reference_run, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 68)
    print(f"finetune_optimized 无气象 benchmark 完成: {run_dir}")
    print(
        f"best_epoch={best_info['best_epoch']} "
        f"| best_val_loss={best_info['best_val_loss']:.6f} "
        f"| best_val_nse={best_info['best_val_nse']:.6f}"
    )
    print(
        f"test_nse={metrics['__overall__']['NSE']:.6f} "
        f"| focus_mean_nse={build_focus_summary(metrics, FOCUS_FEATURES).get('mean_nse', float('nan')):.6f}"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
