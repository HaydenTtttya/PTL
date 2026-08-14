from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MODELS_DIR = BASE_DIR / "models"
REPO_ROOT = SCRIPT_DIR.parents[2]


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Unable to load module: {file_path}")
    spec.loader.exec_module(module)
    return module


BASE_BENCH = load_module("daily_mlp_benchmark_for_cla", SCRIPT_DIR / "benchmark_daily_mlp.py")
CLA_MODEL = load_module("cla_baseline_module", MODELS_DIR / "cla_baseline.py")
PTL_CORE = BASE_BENCH.PTL_CORE


FEATURE_COLUMNS = list(BASE_BENCH.FEATURE_COLUMNS)
FOCUS_FEATURES = list(BASE_BENCH.FOCUS_FEATURES)
DEFAULT_DATA_PATH = BASE_BENCH.DEFAULT_DATA_PATH
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "base" / "fair_compare" / "cla"
DEFAULT_PTL_REFERENCE_DIR = BASE_BENCH.DEFAULT_PTL_REFERENCE_DIR
DEFAULT_LOSS_FEATURE_WEIGHTS = dict(BASE_BENCH.DEFAULT_LOSS_FEATURE_WEIGHTS)
DEFAULT_MONITOR_FEATURE_WEIGHTS = dict(BASE_BENCH.DEFAULT_MONITOR_FEATURE_WEIGHTS)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def parse_args():
    parser = argparse.ArgumentParser(description="Daily CNN-LSTM-Attention baseline")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ptl-reference-dir", type=Path, default=DEFAULT_PTL_REFERENCE_DIR)
    parser.add_argument("--station-name", type=str, default="Yangshuo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--raw-seq-len", type=int, default=12)
    parser.add_argument("--model-seq-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--resize-mode", type=str, default="linear")
    parser.add_argument("--conv-channels", type=int, nargs="*", default=[64, 128])
    parser.add_argument("--kernel-sizes", type=int, nargs="*", default=[3, 3])
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-num-layers", type=int, default=1)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--use-batch-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-input-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-lr", type=float, default=2.5e-4)
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

    frame = BASE_BENCH.read_station_frame(
        args.data_path,
        feature_columns,
        time_start=args.time_start,
        time_end=args.time_end,
    )
    data_bundle = BASE_BENCH.build_scaled_series(
        frame=frame,
        feature_columns=feature_columns,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        soft_gap_max_steps=args.soft_gap_max_steps,
    )
    datasets, loaders = BASE_BENCH.build_loaders(
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

    model = CLA_MODEL.DailyCLABaseline(
        input_dim=len(feature_columns),
        seq_len=args.model_seq_len,
        pred_len=args.pred_len,
        target_dim=len(feature_columns),
        conv_channels=tuple(args.conv_channels),
        kernel_sizes=tuple(args.kernel_sizes),
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        attention_dim=args.attention_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        activation=args.activation,
        use_batch_norm=args.use_batch_norm,
        use_input_layer_norm=args.use_input_layer_norm,
    )
    parameter_count = count_parameters(model)

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
        log_prefix="CLABaseline",
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
    run_dir = args.output_root / f"cla_{args.station_name}_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "model.pth")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(run_dir / "metrics.csv")
    BASE_BENCH.save_predictions(run_dir / "predictions.csv", preds, targets, timestamps, feature_columns)

    reference_run = BASE_BENCH.load_reference_run(args.ptl_reference_dir) if args.ptl_reference_dir else None
    comparison_summary = BASE_BENCH.build_comparison_summary(
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
        "model_family": "cla",
        "model_note": "CNN-LSTM-Attention baseline trained directly on target daily data.",
        "parameter_count": parameter_count,
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
        "conv_channels": [int(value) for value in args.conv_channels],
        "kernel_sizes": [int(value) for value in args.kernel_sizes],
        "lstm_hidden_dim": int(args.lstm_hidden_dim),
        "lstm_num_layers": int(args.lstm_num_layers),
        "attention_dim": int(args.attention_dim),
        "head_hidden_dim": int(args.head_hidden_dim),
        "dropout": float(args.dropout),
        "activation": str(args.activation),
        "use_batch_norm": bool(args.use_batch_norm),
        "use_input_layer_norm": bool(args.use_input_layer_norm),
        "loss_feature_weights": dict(DEFAULT_LOSS_FEATURE_WEIGHTS),
        "monitor_feature_weights": dict(DEFAULT_MONITOR_FEATURE_WEIGHTS),
        "best_epoch": int(best_stats["epoch"]),
        "best_val_loss": float(best_stats["val_loss"]),
        "best_val_nse": float(best_stats["val_nse"]),
        "best_val_monitor_nse": float(best_stats["val_monitor_nse"]),
        "test_loss": float(test_loss),
        "test_nse": float(metrics["__overall__"]["NSE"]),
        "train_seconds": float(train_seconds),
        "focus_summary": BASE_BENCH.build_focus_summary(metrics, FOCUS_FEATURES),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"CLA baseline complete: {run_dir}")
    print(
        f"params={parameter_count} "
        f"| best_epoch={best_stats['epoch']} "
        f"| best_val_nse={best_stats['val_nse']:.6f} "
        f"| test_nse={metrics['__overall__']['NSE']:.6f}"
    )
    focus_summary = BASE_BENCH.build_focus_summary(metrics, FOCUS_FEATURES)
    print(
        f"focus_mean_nse={focus_summary.get('mean_nse', float('nan')):.6f} "
        f"| focus_mean_rmse={focus_summary.get('mean_rmse', float('nan')):.6f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
