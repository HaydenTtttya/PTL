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
        raise ImportError(f"无法加载模块: {file_path}")
    spec.loader.exec_module(module)
    return module


SHARED = load_module("daily_mlp_baseline_shared_for_basic_transformer", SCRIPT_DIR / "benchmark_daily_mlp.py")
BASIC_MODEL = load_module("basic_transformer_baseline_module", MODELS_DIR / "basic_transformer_baseline.py")
PTL_CORE = SHARED.PTL_CORE


FEATURE_COLUMNS = list(SHARED.FEATURE_COLUMNS)
FOCUS_FEATURES = list(SHARED.FOCUS_FEATURES)
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "water_quality_processed_2021_2024" / "daily" / "老口.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "base" / "fair_compare" / "basic_transformer"
DEFAULT_PTL_REFERENCE_DIR = (
    REPO_ROOT
    / "results"
    / "ptl"
    / "finetune"
    / "runs"
    / "batch_pearl_other_core3_progressive_v2pretrain_v2_2021_2024_20260409_145922"
    / "progressive_老口_seed42_20260409_145922"
    / "stage3_daily"
)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def parse_args():
    parser = argparse.ArgumentParser(description="基础 nn.TransformerEncoder 日级 baseline")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ptl-reference-dir", type=Path, default=DEFAULT_PTL_REFERENCE_DIR)
    parser.add_argument("--station-name", type=str, default="老口")
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

    frame = SHARED.read_station_frame(
        args.data_path,
        feature_columns,
        time_start=args.time_start,
        time_end=args.time_end,
    )
    data_bundle = SHARED.build_scaled_series(
        frame=frame,
        feature_columns=feature_columns,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        soft_gap_max_steps=args.soft_gap_max_steps,
    )
    datasets, loaders = SHARED.build_loaders(
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

    model = BASIC_MODEL.BasicTransformerBaseline(
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
        log_prefix="BasicTransformerBaseline",
        loss_name="mse_nse",
        nse_weight=args.nse_weight,
        loss_feature_weights=SHARED.DEFAULT_LOSS_FEATURE_WEIGHTS,
        monitor_metric="nse",
        monitor_feature_weights=SHARED.DEFAULT_MONITOR_FEATURE_WEIGHTS,
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
    run_dir = args.output_root / f"basic_transformer_{args.station_name}_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "model.pth")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(run_dir / "metrics.csv")
    SHARED.save_predictions(run_dir / "predictions.csv", preds, targets, timestamps, feature_columns)

    reference_run = SHARED.load_reference_run(args.ptl_reference_dir) if args.ptl_reference_dir else None
    comparison_summary = SHARED.build_comparison_summary(
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
        "model_family": "basic_transformer_encoder",
        "model_note": (
            "Plain nn.TransformerEncoder baseline: input projection, sinusoidal positional encoding, "
            "TransformerEncoder, and last-step output head only."
        ),
        "uses_ptl_model_architecture": False,
        "uses_pretraining": False,
        "uses_progressive_transfer": False,
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
        "d_model": int(args.d_model),
        "nhead": int(args.nhead),
        "num_layers": int(args.num_layers),
        "dim_feedforward": int(args.dim_feedforward),
        "dropout": float(args.dropout),
        "loss_feature_weights": dict(SHARED.DEFAULT_LOSS_FEATURE_WEIGHTS),
        "monitor_feature_weights": dict(SHARED.DEFAULT_MONITOR_FEATURE_WEIGHTS),
        "best_epoch": int(best_stats["epoch"]),
        "best_val_loss": float(best_stats["val_loss"]),
        "best_val_nse": float(best_stats["val_nse"]),
        "best_val_monitor_nse": float(best_stats["val_monitor_nse"]),
        "test_loss": float(test_loss),
        "test_nse": float(metrics["__overall__"]["NSE"]),
        "train_seconds": float(train_seconds),
        "focus_summary": SHARED.build_focus_summary(metrics, FOCUS_FEATURES),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"Basic transformer baseline 完成: {run_dir}")
    print(
        f"params={parameter_count} "
        f"| best_epoch={best_stats['epoch']} "
        f"| best_val_nse={best_stats['val_nse']:.6f} "
        f"| test_nse={metrics['__overall__']['NSE']:.6f}"
    )
    focus_summary = SHARED.build_focus_summary(metrics, FOCUS_FEATURES)
    print(
        f"focus_mean_nse={focus_summary.get('mean_nse', float('nan')):.6f} "
        f"| focus_mean_rmse={focus_summary.get('mean_rmse', float('nan')):.6f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
