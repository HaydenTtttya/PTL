from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader


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


PTL_CORE = load_module(
    "ptl_progressive_core_traditional_transfer",
    REPO_ROOT / "src" / "PTL" / "progressive_core.py",
)
BASELINE_MODEL = load_module(
    "base_direct_compare_baseline_traditional_transfer",
    MODELS_DIR / "direct_compare_baseline.py",
)


FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
DEFAULT_SOURCE_DATA_DIR = REPO_ROOT / "data" / "data_cleaned" / "yangzte"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "water_quality_processed_2021_2024"
DEFAULT_STATION_META = DEFAULT_DATA_ROOT / "station_meta.csv"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "base"
    / "fair_compare"
    / "traditional_transfer_transformer"
)
DEFAULT_TARGET_STATIONS = ["老口", "上中", "白马", "阳朔"]
DEFAULT_TARGET_LABELS = {
    "老口": "C1",
    "上中": "C2",
    "白马": "C3",
    "阳朔": "C4",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traditional Transfer Learning Transformer baseline: source pre-train 30 epochs, target fine-tune 30 epochs.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--station-meta", type=Path, default=DEFAULT_STATION_META)
    parser.add_argument("--source-data-dir", type=Path, default=DEFAULT_SOURCE_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-station", action="append", default=[])
    parser.add_argument("--target-station", action="append", default=[])
    parser.add_argument("--max-source-stations", type=int, default=None)
    parser.add_argument("--time-start", type=str, default="2023-01-01 00:00:00")
    parser.add_argument("--time-end", type=str, default="2024-12-31 23:59:59")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-epochs", type=int, default=30)
    parser.add_argument("--target-epochs", type=int, default=30)
    parser.add_argument("--source-batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=32)
    parser.add_argument("--raw-seq-len", type=int, default=12)
    parser.add_argument("--model-seq-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--resize-mode", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--soft-gap-max-steps", type=int, default=6)
    parser.add_argument(
        "--invalid-window-policy",
        choices=("all", "input_only", "target_only"),
        default="all",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--e-layer", type=int, default=3)
    parser.add_argument("--temporal-adapter-kernel-size", type=int, default=5)
    parser.add_argument("--use-temporal-adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_model(args):
    return BASELINE_MODEL.DirectDailyTransformerBaseline(
        num_heads=args.num_heads,
        e_layer=args.e_layer,
        hidden_size=args.hidden_size,
        input_dim=len(FEATURE_COLUMNS),
        seq_len=args.model_seq_len,
        pred_len=args.pred_len,
        target_dim=len(FEATURE_COLUMNS),
        target_feature_names=list(FEATURE_COLUMNS),
        use_temporal_adapter=args.use_temporal_adapter,
        temporal_adapter_kernel_size=args.temporal_adapter_kernel_size,
    )


def read_station_frame(data_path: Path, args):
    frame = PTL_CORE.load_water_frame(
        str(data_path),
        time_start=args.time_start,
        time_end=args.time_end,
        expected_freq=PTL_CORE.RESOLUTION_TO_FREQ["daily"],
        feature_columns=FEATURE_COLUMNS,
    )
    if frame is None:
        return None
    return frame.reset_index(drop=True)


def read_source_frame(data_path: Path):
    frame = pd.read_csv(data_path)
    if not set(FEATURE_COLUMNS).issubset(frame.columns):
        return None
    values = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    raw_invalid_mask = values.isna().any(axis=1).to_numpy(dtype=bool)
    values = values.interpolate(limit_direction="both").dropna()
    if values.empty:
        return None
    cleaned = values.reset_index(drop=True).copy()
    cleaned.insert(
        0,
        "timestamp",
        pd.date_range("2000-01-02", periods=len(cleaned), freq="W-SUN"),
    )
    cleaned["__gap__"] = raw_invalid_mask[: len(cleaned)]
    return cleaned


def build_invalid_masks(frame: pd.DataFrame, soft_gap_max_steps: int | None):
    raw_invalid_mask = (
        frame["__gap__"].to_numpy(dtype=bool, copy=True)
        if "__gap__" in frame.columns
        else np.zeros(len(frame), dtype=bool)
    )
    input_invalid_mask, target_invalid_mask, gap_stats = PTL_CORE.build_gap_aware_invalid_masks(
        raw_invalid_mask,
        soft_gap_max_steps=soft_gap_max_steps,
    )
    return raw_invalid_mask, input_invalid_mask, target_invalid_mask, gap_stats


def build_window_dataset(
    frame: pd.DataFrame,
    scaler,
    split: str,
    args,
):
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    scaled_values = scaler.transform(values).astype(np.float32)
    timestamps = frame["timestamp"].to_numpy(dtype="datetime64[ns]")
    _, input_invalid_mask, target_invalid_mask, _ = build_invalid_masks(
        frame,
        soft_gap_max_steps=args.soft_gap_max_steps,
    )
    return PTL_CORE.ForecastWindowDataset(
        timestamps=timestamps,
        raw_seq_len=args.raw_seq_len,
        raw_pred_len=args.pred_len,
        model_seq_len=args.model_seq_len,
        model_pred_len=args.pred_len,
        input_values=scaled_values,
        target_values=scaled_values,
        split=split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        filter_invalid_windows=True,
        invalid_window_policy=args.invalid_window_policy,
        resize_mode=args.resize_mode,
        input_invalid_mask=input_invalid_mask,
        target_invalid_mask=target_invalid_mask,
    )


def select_source_files(args):
    if not args.source_data_dir.exists():
        raise FileNotFoundError(f"source data dir not found: {args.source_data_dir}")
    all_files = sorted(
        path
        for path in args.source_data_dir.glob("*.csv")
        if not path.name.startswith(".")
    )
    if args.source_station:
        wanted = list(dict.fromkeys(args.source_station))
        indexed = {path.stem: path for path in all_files}
        indexed.update({path.name: path for path in all_files})
        missing = [name for name in wanted if name not in indexed]
        if missing:
            raise ValueError(f"source station not found in source data dir: {missing}")
        source_files = [indexed[name] for name in wanted]
    else:
        source_files = all_files
        if args.max_source_stations is not None:
            source_files = source_files[: int(args.max_source_stations)]
    if not source_files:
        raise ValueError("No source station files selected.")
    return source_files


def prepare_source_loaders(args, target_stations):
    source_files = select_source_files(args)
    loaded_frames = []
    source_preview = []
    for source_path in source_files:
        station_name = source_path.stem
        frame = read_source_frame(source_path)
        if frame is None:
            continue
        if len(frame) < args.raw_seq_len + args.pred_len + 8:
            continue
        loaded_frames.append((station_name, frame))

    if not loaded_frames:
        raise ValueError("No source frames could be loaded.")

    scaler = PTL_CORE.StandardScaler()
    scaler.fit(
        np.concatenate(
            [
                frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
                for _, frame in loaded_frames
            ],
            axis=0,
        )
    )

    train_datasets = []
    val_datasets = []
    for station_name, frame in loaded_frames:
        train_dataset = build_window_dataset(frame, scaler, "train", args)
        val_dataset = build_window_dataset(frame, scaler, "val", args)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            continue
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        raw_invalid_mask, input_invalid_mask, target_invalid_mask, gap_stats = build_invalid_masks(
            frame,
            soft_gap_max_steps=args.soft_gap_max_steps,
        )
        source_preview.append(
            {
                "station_name": station_name,
                "records": int(len(frame)),
                "invalid_records": int(raw_invalid_mask.sum()),
                "input_invalid_records": int(input_invalid_mask.sum()),
                "target_invalid_records": int(target_invalid_mask.sum()),
                "train_windows": int(len(train_dataset)),
                "train_candidate_windows": int(train_dataset.candidate_window_count),
                "train_filtered_windows": int(train_dataset.filtered_window_count),
                "val_windows": int(len(val_dataset)),
                "val_candidate_windows": int(val_dataset.candidate_window_count),
                "val_filtered_windows": int(val_dataset.filtered_window_count),
                **{f"gap_{key}": value for key, value in gap_stats.items()},
            }
        )

    if not train_datasets or not val_datasets:
        raise ValueError("No valid source windows after filtering.")

    train_concat = ConcatDataset(train_datasets)
    val_concat = ConcatDataset(val_datasets)
    loaders = {
        "train": DataLoader(train_concat, batch_size=args.source_batch_size, shuffle=True),
        "val": DataLoader(val_concat, batch_size=args.source_batch_size, shuffle=False),
    }
    return loaders, scaler, pd.DataFrame(source_preview)


def prepare_target_loaders(args, station_name):
    data_path = args.data_root / "daily" / f"{station_name}.csv"
    frame = read_station_frame(data_path, args)
    if frame is None:
        raise FileNotFoundError(f"cannot load target daily data: {data_path}")

    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    train_end, _ = PTL_CORE.compute_split_points(len(values), args.train_ratio, args.val_ratio)
    scaler = PTL_CORE.StandardScaler()
    scaler.fit(values[:train_end])

    datasets = {
        split: build_window_dataset(frame, scaler, split, args)
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.target_batch_size,
            shuffle=(split == "train"),
        )
        for split, dataset in datasets.items()
    }
    raw_invalid_mask, input_invalid_mask, target_invalid_mask, gap_stats = build_invalid_masks(
        frame,
        soft_gap_max_steps=args.soft_gap_max_steps,
    )
    meta = {
        "station_name": station_name,
        "data_path": str(data_path),
        "records": int(len(frame)),
        "invalid_records": int(raw_invalid_mask.sum()),
        "input_invalid_records": int(input_invalid_mask.sum()),
        "target_invalid_records": int(target_invalid_mask.sum()),
        "gap_stats": gap_stats,
        "train_windows": int(len(datasets["train"])),
        "train_candidate_windows": int(datasets["train"].candidate_window_count),
        "train_filtered_windows": int(datasets["train"].filtered_window_count),
        "val_windows": int(len(datasets["val"])),
        "val_candidate_windows": int(datasets["val"].candidate_window_count),
        "val_filtered_windows": int(datasets["val"].filtered_window_count),
        "test_windows": int(len(datasets["test"])),
        "test_candidate_windows": int(datasets["test"].candidate_window_count),
        "test_filtered_windows": int(datasets["test"].filtered_window_count),
    }
    return loaders, scaler, meta


def evaluate_on_loader(model, loader, device, scaler=None):
    test_loss, preds, targets, timestamps = PTL_CORE.evaluate_model(
        model,
        loader,
        device=device,
        scaler=scaler,
    )
    metrics = PTL_CORE.compute_per_feature_metrics(preds, targets, FEATURE_COLUMNS)
    return test_loss, preds, targets, timestamps, metrics


def train_fixed_epochs(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    epsilon,
    weight_decay,
    max_grad_norm,
    log_prefix,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        eps=epsilon,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()
    history = []

    for epoch in range(int(epochs)):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            if output.shape != y.shape:
                y = y.view_as(output)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_loss += float(loss.item())

        train_loss = train_loss / max(1, len(train_loader))
        val_loss, val_preds, val_targets, _ = PTL_CORE.evaluate_model(
            model,
            val_loader,
            device=device,
            scaler=None,
        )
        val_metrics = PTL_CORE.compute_per_feature_metrics(val_preds, val_targets, FEATURE_COLUMNS)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": float(val_loss),
                "val_nse": float(val_metrics["__overall__"]["NSE"]),
            }
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == int(epochs):
            print(
                f"[{log_prefix}] Epoch {epoch + 1}/{epochs} "
                f"| TrainMSE: {train_loss:.6f} "
                f"| ValMSE: {val_loss:.6f} "
                f"| ValNSE: {val_metrics['__overall__']['NSE']:.6f}"
            )

    return model, history


def save_predictions(path: Path, preds, targets, timestamps):
    prediction_frame = pd.DataFrame({"timestamp": pd.to_datetime(timestamps.reshape(-1))})
    preds_flat = preds.reshape(-1, preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])
    for index, feature_name in enumerate(FEATURE_COLUMNS):
        prediction_frame[f"True_{feature_name}"] = targets_flat[:, index]
        prediction_frame[f"Pred_{feature_name}"] = preds_flat[:, index]
    prediction_frame.to_csv(path, index=False)


def build_focus_summary(metrics):
    return {
        "features": list(FOCUS_FEATURES),
        "mean_nse": float(np.mean([metrics[name]["NSE"] for name in FOCUS_FEATURES])),
        "mean_rmse": float(np.mean([metrics[name]["RMSE"] for name in FOCUS_FEATURES])),
        "mean_mae": float(np.mean([metrics[name]["MAE"] for name in FOCUS_FEATURES])),
    }


def main():
    args = parse_args()
    PTL_CORE.set_seed(args.seed)
    device = PTL_CORE.infer_device()
    target_stations = list(dict.fromkeys(args.target_station or DEFAULT_TARGET_STATIONS))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"traditional_transfer_transformer_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Traditional Transfer Learning Transformer")
    print(f"source: {args.source_data_dir} | targets: {target_stations}")
    print(f"source_epochs={args.source_epochs} | target_epochs={args.target_epochs}")
    print("=" * 70)

    source_loaders, _, source_preview = prepare_source_loaders(args, target_stations)
    source_preview_path = run_dir / "source_stations.csv"
    source_preview.to_csv(source_preview_path, index=False, encoding="utf-8-sig")
    print(
        f"source stations={len(source_preview)} "
        f"| train_windows={sum(source_preview['train_windows'])} "
        f"| val_windows={sum(source_preview['val_windows'])}"
    )

    setup = {
        "method": "Traditional Transfer Learning",
        "paper_reference": (
            "source model is pre-trained for n=30 epochs, last epoch parameters are saved, "
            "then target training data fine-tunes the saved model for n=30 epochs."
        ),
        "source_data_dir": str(args.source_data_dir),
        "data_root": str(args.data_root),
        "station_meta": str(args.station_meta),
        "target_stations": target_stations,
        "feature_columns": list(FEATURE_COLUMNS),
        "focus_features": list(FOCUS_FEATURES),
        "time_start": args.time_start,
        "time_end": args.time_end,
        "source_epochs": int(args.source_epochs),
        "target_epochs": int(args.target_epochs),
        "checkpoint_policy": "last_epoch",
        "optimizer": "Adam",
        "loss": "MSE",
        "lr": float(args.lr),
        "epsilon": float(args.epsilon),
        "weight_decay": float(args.weight_decay),
        "raw_seq_len": int(args.raw_seq_len),
        "model_seq_len": int(args.model_seq_len),
        "pred_len": int(args.pred_len),
        "hidden_size": int(args.hidden_size),
        "num_heads": int(args.num_heads),
        "e_layer": int(args.e_layer),
        "use_temporal_adapter": bool(args.use_temporal_adapter),
        "temporal_adapter_kernel_size": int(args.temporal_adapter_kernel_size),
        "soft_gap_max_steps": args.soft_gap_max_steps,
        "invalid_window_policy": args.invalid_window_policy,
        "device": str(device),
        "source_stations_path": str(source_preview_path),
    }
    (run_dir / "run_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"dry run complete: {run_dir}")
        return

    source_model = build_model(args)
    source_start = time.time()
    source_model, source_history = train_fixed_epochs(
        source_model,
        train_loader=source_loaders["train"],
        val_loader=source_loaders["val"],
        device=device,
        epochs=args.source_epochs,
        lr=args.lr,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        log_prefix="SourcePretrain",
    )
    source_seconds = time.time() - source_start
    source_dir = run_dir / "source_pretrain"
    source_dir.mkdir(parents=True, exist_ok=True)
    torch.save(source_model.state_dict(), source_dir / "model_last.pth")
    pd.DataFrame(source_history).to_csv(source_dir / "history.csv", index=False)
    (source_dir / "meta.json").write_text(
        json.dumps(
            {
                "source_epochs": int(args.source_epochs),
                "checkpoint_policy": "last_epoch",
                "train_seconds": float(source_seconds),
                "final_epoch": int(source_history[-1]["epoch"]),
                "final_val_loss": float(source_history[-1]["val_loss"]),
                "final_val_nse": float(source_history[-1]["val_nse"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_rows = []
    long_rows = []
    for station_name in target_stations:
        print("\n" + "=" * 70)
        print(f"Target fine-tune: {station_name}")
        target_loaders, target_scaler, target_meta = prepare_target_loaders(args, station_name)
        target_model = build_model(args)
        target_model.load_state_dict(source_model.state_dict())
        target_start = time.time()
        target_model, target_history = train_fixed_epochs(
            target_model,
            train_loader=target_loaders["train"],
            val_loader=target_loaders["val"],
            device=device,
            epochs=args.target_epochs,
            lr=args.lr,
            epsilon=args.epsilon,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            log_prefix=f"TargetFineTune:{station_name}",
        )
        target_seconds = time.time() - target_start
        test_loss, preds, targets, timestamps, metrics = evaluate_on_loader(
            target_model,
            target_loaders["test"],
            device=device,
            scaler=target_scaler,
        )
        station_dir = run_dir / f"target_{station_name}"
        station_dir.mkdir(parents=True, exist_ok=True)
        torch.save(target_model.state_dict(), station_dir / "model_last.pth")
        pd.DataFrame(target_history).to_csv(station_dir / "history.csv", index=False)
        pd.DataFrame.from_dict(metrics, orient="index").to_csv(station_dir / "metrics.csv")
        save_predictions(station_dir / "predictions.csv", preds, targets, timestamps)
        focus_summary = build_focus_summary(metrics)
        station_meta = {
            **target_meta,
            "method": "Traditional Transfer Learning",
            "source_model_path": str(source_dir / "model_last.pth"),
            "source_epochs": int(args.source_epochs),
            "target_epochs": int(args.target_epochs),
            "checkpoint_policy": "last_epoch",
            "optimizer": "Adam",
            "loss": "MSE",
            "target_train_seconds": float(target_seconds),
            "test_loss": float(test_loss),
            "test_nse": float(metrics["__overall__"]["NSE"]),
            "focus_summary": focus_summary,
            "final_epoch": int(target_history[-1]["epoch"]),
            "final_val_loss": float(target_history[-1]["val_loss"]),
            "final_val_nse": float(target_history[-1]["val_nse"]),
        }
        (station_dir / "meta.json").write_text(
            json.dumps(station_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_rows.append(
            {
                "station_label": DEFAULT_TARGET_LABELS.get(station_name, ""),
                "station_name": station_name,
                "model": "TraditionalTransferLearningTransformer",
                "overall_nse": float(metrics["__overall__"]["NSE"]),
                "overall_rmse": float(metrics["__overall__"]["RMSE"]),
                "focus_mean_nse": focus_summary["mean_nse"],
                "focus_mean_rmse": focus_summary["mean_rmse"],
                "source_epochs": int(args.source_epochs),
                "target_epochs": int(args.target_epochs),
                "train_windows": target_meta["train_windows"],
                "val_windows": target_meta["val_windows"],
                "test_windows": target_meta["test_windows"],
                "metrics_path": str(station_dir / "metrics.csv"),
                "meta_path": str(station_dir / "meta.json"),
            }
        )
        for feature_name, row in metrics.items():
            long_rows.append(
                {
                    "station_label": DEFAULT_TARGET_LABELS.get(station_name, ""),
                    "station_name": station_name,
                    "model": "TraditionalTransferLearningTransformer",
                    "feature": feature_name,
                    "MAE": row.get("MAE"),
                    "RMSE": row.get("RMSE"),
                    "NSE": row.get("NSE"),
                    "MAPE": row.get("MAPE"),
                    "metrics_path": str(station_dir / "metrics.csv"),
                }
            )
        print(
            f"{station_name} done | overall_nse={metrics['__overall__']['NSE']:.6f} "
            f"| focus_mean_nse={focus_summary['mean_nse']:.6f}"
        )

    summary = pd.DataFrame(summary_rows).sort_values("station_label")
    long = pd.DataFrame(long_rows).sort_values(["station_label", "feature"])
    summary.to_csv(run_dir / "traditional_transfer_summary.csv", index=False, encoding="utf-8-sig")
    long.to_csv(run_dir / "traditional_transfer_metrics_long.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 70)
    print(f"Traditional Transfer Learning complete: {run_dir}")
    print(summary[["station_label", "station_name", "overall_nse", "focus_mean_nse"]].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
