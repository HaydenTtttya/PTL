from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MODELS_DIR = BASE_DIR / "models"
REPO_ROOT = SCRIPT_DIR.parents[2]


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    spec.loader.exec_module(module)
    return module


PTL_CORE = load_module("ptl_progressive_core_plain_baseline", REPO_ROOT / "src" / "PTL" / "progressive_core.py")
PLAIN_MODEL = load_module("plain_transformer_baseline_module", MODELS_DIR / "plain_transformer_baseline.py")


FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "water_quality_processed_2023_2025" / "daily" / "阳朔.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "base" / "fair_compare" / "plain_transformer"
DEFAULT_PTL_REFERENCE_DIR = (
    REPO_ROOT
    / "results"
    / "ptl"
    / "finetune"
    / "runs"
    / "progressive_阳朔_seed42_20260408_185725"
    / "stage3_daily"
)
DEFAULT_LOSS_FEATURE_WEIGHTS = {
    "CODMn": 1.8,
    "DO": 1.4,
    "NH4N": 0.05,
    "pH": 1.45,
}
DEFAULT_MONITOR_FEATURE_WEIGHTS = {
    "CODMn": 1.85,
    "DO": 1.4,
    "NH4N": 0.05,
    "pH": 1.55,
}


class DailyForecastDataset(Dataset):
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

        x = PTL_CORE.resize_sequence_length(
            self.values[start:mid],
            target_len=self.model_seq_len,
            mode=self.resize_mode,
        )
        y = PTL_CORE.resize_sequence_length(
            self.values[mid:end],
            target_len=self.model_pred_len,
            mode="linear",
        )
        target_times = self.timestamps[mid:end].astype(np.int64)

        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(target_times, dtype=torch.int64),
        )


def read_station_frame(
    data_path: Path,
    feature_columns: list[str],
    time_start: str | None = None,
    time_end: str | None = None,
):
    frame = PTL_CORE.load_water_frame(
        str(data_path),
        time_start=time_start,
        time_end=time_end,
        expected_freq=PTL_CORE.RESOLUTION_TO_FREQ["daily"],
        feature_columns=feature_columns,
    )
    if frame is None:
        raise FileNotFoundError(f"无法读取有效日级数据: {data_path}")
    return frame.reset_index(drop=True)


def build_scaled_series(
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_ratio: float,
    val_ratio: float,
    soft_gap_max_steps: int | None,
):
    values = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    timestamps = frame["timestamp"].to_numpy(dtype="datetime64[ns]")
    raw_invalid_mask = (
        frame["__gap__"].to_numpy(dtype=bool, copy=True)
        if "__gap__" in frame.columns
        else np.zeros(len(frame), dtype=bool)
    )
    input_invalid_mask, target_invalid_mask, gap_stats = PTL_CORE.build_gap_aware_invalid_masks(
        raw_invalid_mask,
        soft_gap_max_steps=soft_gap_max_steps,
    )
    train_end, _ = PTL_CORE.compute_split_points(len(values), train_ratio, val_ratio)

    scaler = PTL_CORE.StandardScaler()
    scaler.fit(values[:train_end])
    scaled_values = scaler.transform(values).astype(np.float32)

    return {
        "timestamps": timestamps,
        "raw_values": values,
        "scaled_values": scaled_values,
        "scaler": scaler,
        "raw_invalid_mask": raw_invalid_mask,
        "input_invalid_mask": input_invalid_mask,
        "target_invalid_mask": target_invalid_mask,
        "gap_stats": gap_stats,
    }


def build_loaders(
    scaled_values: np.ndarray,
    timestamps: np.ndarray,
    input_invalid_mask: np.ndarray,
    target_invalid_mask: np.ndarray,
    raw_seq_len: int,
    model_seq_len: int,
    raw_pred_len: int,
    model_pred_len: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    resize_mode: str = "linear",
    invalid_window_policy: str = "all",
):
    datasets = {}
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = PTL_CORE.ForecastWindowDataset(
            timestamps=timestamps,
            raw_seq_len=raw_seq_len,
            raw_pred_len=raw_pred_len,
            model_seq_len=model_seq_len,
            model_pred_len=model_pred_len,
            input_values=scaled_values,
            target_values=scaled_values,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            filter_invalid_windows=True,
            invalid_window_policy=invalid_window_policy,
            resize_mode=resize_mode,
            input_invalid_mask=input_invalid_mask,
            target_invalid_mask=target_invalid_mask,
        )
        datasets[split] = dataset
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
    return datasets, loaders


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


def save_predictions(path: Path, preds, targets, timestamps, feature_columns: list[str]):
    prediction_frame = pd.DataFrame({"timestamp": pd.to_datetime(timestamps.reshape(-1))})
    preds_flat = preds.reshape(-1, preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])
    for index, feature_name in enumerate(feature_columns):
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


def build_comparison_summary(baseline_metrics, reference_metrics, focus_features: list[str]):
    summary = {
        "baseline_overall": baseline_metrics.get("__overall__"),
        "reference_overall": reference_metrics.get("__overall__") if reference_metrics else None,
        "baseline_focus": build_focus_summary(baseline_metrics, focus_features),
        "reference_focus": build_focus_summary(reference_metrics, focus_features) if reference_metrics else None,
        "per_feature_delta_baseline_minus_reference": {},
    }
    if reference_metrics:
        for feature_name in focus_features:
            if feature_name not in baseline_metrics or feature_name not in reference_metrics:
                continue
            summary["per_feature_delta_baseline_minus_reference"][feature_name] = {
                key: float(baseline_metrics[feature_name][key]) - float(reference_metrics[feature_name][key])
                for key in ("MAE", "RMSE", "NSE", "MAPE")
                if baseline_metrics[feature_name].get(key) is not None and reference_metrics[feature_name].get(key) is not None
            }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="基础传统 Transformer 日级 baseline")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ptl-reference-dir", type=Path, default=DEFAULT_PTL_REFERENCE_DIR)
    parser.add_argument("--station-name", type=str, default="阳朔")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--raw-seq-len", type=int, default=12)
    parser.add_argument("--model-seq-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--resize-mode", type=str, default="linear")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=[30, 60, 90])
    parser.add_argument("--lr-decay-ratio", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--nse-weight", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=6)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--time-start", type=str, default=None)
    parser.add_argument("--time-end", type=str, default=None)
    parser.add_argument("--soft-gap-max-steps", type=int, default=6)
    parser.add_argument(
        "--invalid-window-policy",
        choices=("all", "input_only", "target_only"),
        default="all",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    PTL_CORE.set_seed(args.seed)
    device = PTL_CORE.infer_device()
    feature_columns = list(FEATURE_COLUMNS)

    frame = read_station_frame(
        args.data_path,
        feature_columns,
        time_start=args.time_start,
        time_end=args.time_end,
    )
    data_bundle = build_scaled_series(
        frame=frame,
        feature_columns=feature_columns,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        soft_gap_max_steps=args.soft_gap_max_steps,
    )
    datasets, loaders = build_loaders(
        scaled_values=data_bundle["scaled_values"],
        timestamps=data_bundle["timestamps"],
        input_invalid_mask=data_bundle["input_invalid_mask"],
        target_invalid_mask=data_bundle["target_invalid_mask"],
        raw_seq_len=args.raw_seq_len,
        model_seq_len=args.model_seq_len,
        raw_pred_len=args.pred_len,
        model_pred_len=args.pred_len,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        resize_mode=args.resize_mode,
        invalid_window_policy=args.invalid_window_policy,
    )

    model = PLAIN_MODEL.PlainTransformerBaseline(
        input_dim=len(feature_columns),
        seq_len=args.model_seq_len,
        pred_len=args.pred_len,
        target_dim=len(feature_columns),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )

    train_start = time.time()
    model, history, best_stats = PTL_CORE.fit_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        epochs=args.epochs,
        base_lr=args.base_lr,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        lr_milestones=args.lr_milestones,
        lr_decay_ratio=args.lr_decay_ratio,
        max_grad_norm=args.max_grad_norm,
        log_prefix="PlainTransformerBaseline",
        loss_name="mse_nse",
        nse_weight=args.nse_weight,
        loss_feature_weights=DEFAULT_LOSS_FEATURE_WEIGHTS,
        monitor_metric="nse",
        monitor_feature_weights=DEFAULT_MONITOR_FEATURE_WEIGHTS,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        scheduler_name="plateau",
        scheduler_patience=args.scheduler_patience,
        scheduler_min_lr=args.scheduler_min_lr,
        freeze_backbone_epochs=0,
        feature_names=feature_columns,
    )
    train_seconds = time.time() - train_start

    test_loss, preds, targets, timestamps = PTL_CORE.evaluate_model(
        model,
        loaders["test"],
        device=device,
        scaler=data_bundle["scaler"],
    )
    metrics = PTL_CORE.compute_per_feature_metrics(preds, targets, feature_names=feature_columns)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"plain_transformer_{args.station_name}_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "model.pth")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(run_dir / "metrics.csv")
    save_predictions(run_dir / "predictions.csv", preds, targets, timestamps, feature_columns)

    reference_run = load_reference_run(args.ptl_reference_dir) if args.ptl_reference_dir else None
    comparison_summary = build_comparison_summary(
        baseline_metrics=metrics,
        reference_metrics=(reference_run or {}).get("metrics"),
        focus_features=FOCUS_FEATURES,
    )

    meta = {
        "station_name": args.station_name,
        "data_path": str(args.data_path),
        "feature_columns": feature_columns,
        "focus_features": list(FOCUS_FEATURES),
        "ptl_reference_dir": str(args.ptl_reference_dir) if args.ptl_reference_dir else None,
        "device": str(device),
        "records": int(len(frame)),
        "invalid_records": int(data_bundle["raw_invalid_mask"].sum()),
        "input_invalid_records": int(data_bundle["input_invalid_mask"].sum()),
        "target_invalid_records": int(data_bundle["target_invalid_mask"].sum()),
        "gap_stats": data_bundle["gap_stats"],
        "train_windows": int(len(datasets["train"])),
        "train_candidate_windows": int(getattr(datasets["train"], "candidate_window_count", len(datasets["train"]))),
        "train_filtered_windows": int(getattr(datasets["train"], "filtered_window_count", 0)),
        "val_windows": int(len(datasets["val"])),
        "val_candidate_windows": int(getattr(datasets["val"], "candidate_window_count", len(datasets["val"]))),
        "val_filtered_windows": int(getattr(datasets["val"], "filtered_window_count", 0)),
        "test_windows": int(len(datasets["test"])),
        "test_candidate_windows": int(getattr(datasets["test"], "candidate_window_count", len(datasets["test"]))),
        "test_filtered_windows": int(getattr(datasets["test"], "filtered_window_count", 0)),
        "time_start": args.time_start,
        "time_end": args.time_end,
        "soft_gap_max_steps": args.soft_gap_max_steps,
        "invalid_window_policy": args.invalid_window_policy,
        "raw_seq_len": int(args.raw_seq_len),
        "model_seq_len": int(args.model_seq_len),
        "pred_len": int(args.pred_len),
        "d_model": int(args.d_model),
        "nhead": int(args.nhead),
        "num_layers": int(args.num_layers),
        "dim_feedforward": int(args.dim_feedforward),
        "dropout": float(args.dropout),
        "loss_feature_weights": dict(DEFAULT_LOSS_FEATURE_WEIGHTS),
        "monitor_feature_weights": dict(DEFAULT_MONITOR_FEATURE_WEIGHTS),
        "best_epoch": int(best_stats["epoch"]),
        "best_val_loss": float(best_stats["val_loss"]),
        "best_val_nse": float(best_stats["val_nse"]),
        "best_val_monitor_nse": float(best_stats["val_monitor_nse"]),
        "test_loss": float(test_loss),
        "test_nse": float(metrics["__overall__"]["NSE"]),
        "train_seconds": float(train_seconds),
        "focus_summary": build_focus_summary(metrics, FOCUS_FEATURES),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"Plain transformer baseline 完成: {run_dir}")
    print(
        f"best_epoch={best_stats['epoch']} "
        f"| best_val_nse={best_stats['val_nse']:.6f} "
        f"| test_nse={metrics['__overall__']['NSE']:.6f}"
    )
    focus_summary = build_focus_summary(metrics, FOCUS_FEATURES)
    print(
        f"focus_mean_nse={focus_summary.get('mean_nse', float('nan')):.6f} "
        f"| focus_mean_rmse={focus_summary.get('mean_rmse', float('nan')):.6f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
