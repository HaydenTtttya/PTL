import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    spec.loader.exec_module(module)
    return module


BASE_BENCH = load_module("base_daily_benchmark_module", SCRIPT_DIR / "benchmark_daily_yangshuo_nh4n.py")
PTL_CORE = load_module("ptl_progressive_core_processed_base", REPO_ROOT / "src" / "PTL" / "progressive_core.py")


FEATURE_COLUMNS = list(BASE_BENCH.FEATURE_COLUMNS)
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "water_quality_processed_2021_2024" / "daily" / "深圳河口.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "base" / "finetune" / "processed_daily_benchmarks"
DEFAULT_PTL_REFERENCE_DIR = (
    REPO_ROOT
    / "results"
    / "ptl"
    / "finetune"
    / "runs"
    / "batch_pearl_other_core3_progressive_v2pretrain_v2_2021_2024_20260409_150246"
    / "progressive_深圳河口_seed42_20260409_150246"
    / "stage3_daily"
)
DEFAULT_TIME_START = "2023-01-01 00:00:00"
DEFAULT_TIME_END = "2024-12-31 23:59:59"


class WrappedForecastDataset(Dataset):
    def __init__(self, forecast_dataset):
        self.forecast_dataset = forecast_dataset

    def __len__(self):
        return len(self.forecast_dataset)

    def __getitem__(self, index):
        water_x, water_y, target_times = self.forecast_dataset[index]
        weather_x = torch.empty((water_x.shape[0], 0), dtype=torch.float32)
        return water_x, weather_x, water_y, target_times


def load_reference_run(reference_dir: Path | None):
    if reference_dir is None:
        return {"reference_dir": None, "metrics": None, "meta": None, "prediction_count": None}

    metric_candidates = [
        reference_dir / "metrics.csv",
        reference_dir / "评估指标_metrics.csv",
    ]
    meta_candidates = [
        reference_dir / "meta.json",
        reference_dir / "运行元信息_meta.json",
    ]
    prediction_candidates = [
        reference_dir / "predictions.csv",
        reference_dir / "预测明细_predictions.csv",
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

    prediction_count = None
    for prediction_path in prediction_candidates:
        if prediction_path.exists():
            prediction_count = int(len(pd.read_csv(prediction_path)))
            break

    return {
        "reference_dir": str(reference_dir),
        "metrics": metrics,
        "meta": meta,
        "prediction_count": prediction_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Base 模型在 processed daily 数据上的日级 benchmark")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pretrain-dir", type=Path, default=None)
    parser.add_argument("--ptl-reference-dir", type=Path, default=DEFAULT_PTL_REFERENCE_DIR)
    parser.add_argument("--station-name", type=str, default="深圳河口")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--raw-seq-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--base-lr", type=float, default=4e-4)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=[40, 60, 80])
    parser.add_argument("--lr-decay-ratio", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--freeze-ratio", type=float, default=0.38)
    parser.add_argument("--resize-mode", type=str, default="linear")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--time-start", type=str, default=DEFAULT_TIME_START)
    parser.add_argument("--time-end", type=str, default=DEFAULT_TIME_END)
    parser.add_argument("--soft-gap-max-steps", type=int, default=6)
    parser.add_argument(
        "--invalid-window-policy",
        choices=("all", "input_only", "target_only"),
        default="all",
    )
    return parser.parse_args()


def read_station_frame(data_path: Path, time_start: str, time_end: str):
    frame = PTL_CORE.load_water_frame(
        str(data_path),
        time_start=time_start,
        time_end=time_end,
        expected_freq=PTL_CORE.RESOLUTION_TO_FREQ["daily"],
        feature_columns=FEATURE_COLUMNS,
    )
    if frame is None:
        raise FileNotFoundError(f"无法读取有效日级数据: {data_path}")
    return frame.reset_index(drop=True)


def build_scaled_series(frame: pd.DataFrame, train_ratio: float, val_ratio: float, soft_gap_max_steps: int | None):
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
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
        "raw_invalid_mask": raw_invalid_mask,
        "input_invalid_mask": input_invalid_mask,
        "target_invalid_mask": target_invalid_mask,
        "gap_stats": gap_stats,
    }


def build_loaders(
    scaled_values,
    timestamps,
    input_invalid_mask,
    target_invalid_mask,
    raw_seq_len,
    model_seq_len,
    raw_pred_len,
    model_pred_len,
    batch_size,
    train_ratio,
    val_ratio,
    resize_mode,
    invalid_window_policy,
):
    datasets = {}
    loaders = {}
    for split in ("train", "val", "test"):
        forecast_dataset = PTL_CORE.ForecastWindowDataset(
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
        dataset = WrappedForecastDataset(forecast_dataset)
        datasets[split] = dataset
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
    return datasets, loaders


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    pretrain_dir = args.pretrain_dir
    if pretrain_dir is None:
        latest_pretrain_dir = BASE_BENCH.BASE_FINETUNE.find_latest_pretrain_run(
            BASE_BENCH.BASE_FINETUNE.PRETRAIN_RUNS_DIR
        )
        if latest_pretrain_dir is None:
            raise FileNotFoundError("未找到可用的 Base 预训练目录。")
        pretrain_dir = Path(latest_pretrain_dir)

    PTL_CORE.set_seed(args.seed)
    device = PTL_CORE.infer_device()

    frame = read_station_frame(
        data_path=args.data_path,
        time_start=args.time_start,
        time_end=args.time_end,
    )
    data_bundle = build_scaled_series(
        frame=frame,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        soft_gap_max_steps=args.soft_gap_max_steps,
    )
    pretrain_config = BASE_BENCH.load_pretrain_metadata(pretrain_dir)
    model_seq_len = int(pretrain_config["n_in"])

    datasets, loaders = build_loaders(
        scaled_values=data_bundle["scaled_values"],
        timestamps=data_bundle["timestamps"],
        input_invalid_mask=data_bundle["input_invalid_mask"],
        target_invalid_mask=data_bundle["target_invalid_mask"],
        raw_seq_len=args.raw_seq_len,
        model_seq_len=model_seq_len,
        raw_pred_len=args.pred_len,
        model_pred_len=args.pred_len,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        resize_mode=args.resize_mode,
        invalid_window_policy=args.invalid_window_policy,
    )

    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("train/val/test 至少有一个切分没有可用窗口。")

    config, model = BASE_BENCH.build_model(pretrain_config, device=device)
    matched_state = BASE_BENCH.load_matching_pretrain_weights(model, pretrain_dir=pretrain_dir, device=device)
    freeze_stats = BASE_BENCH.freeze_backbone(model, freeze_ratio=args.freeze_ratio)

    print("=" * 68)
    print("Base Processed Daily Benchmark")
    print("=" * 68)
    print(f"数据文件: {args.data_path}")
    print(f"PTL 对照目录: {args.ptl_reference_dir}")
    print(f"预训练目录: {pretrain_dir}")
    print(f"站点: {args.station_name}")
    print(f"特征: {FEATURE_COLUMNS}")
    print(
        f"时间范围: {args.time_start} -> {args.time_end}"
    )
    print(
        f"切分: train/val/test = {args.train_ratio:.2f}/{args.val_ratio:.2f}/{1.0 - args.train_ratio - args.val_ratio:.2f}"
    )
    print(
        f"窗口: raw_seq_len={args.raw_seq_len}, model_seq_len={model_seq_len}, pred_len={args.pred_len}"
    )
    print(
        f"缺口: raw={int(data_bundle['raw_invalid_mask'].sum())}, "
        f"input_invalid={int(data_bundle['input_invalid_mask'].sum())}, "
        f"target_invalid={int(data_bundle['target_invalid_mask'].sum())}, "
        f"soft_gap<={data_bundle['gap_stats']['soft_gap_max_steps']}"
    )
    print(
        f"样本数: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}"
    )
    print(f"迁移权重数: {len(matched_state)}")
    print(
        f"冻结参数: {freeze_stats['frozen_param_count']}/{freeze_stats['transformer_param_count']}"
    )

    start_time = pd.Timestamp.now()
    model, history, best_info = BASE_BENCH.train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        args=args,
    )
    train_seconds = float((pd.Timestamp.now() - start_time).total_seconds())

    test_loss, preds, targets, timestamps = BASE_BENCH.evaluate_model(
        model,
        loaders["test"],
        device=device,
        scaler=data_bundle["scaler"],
    )
    metrics = PTL_CORE.compute_per_feature_metrics(preds, targets, feature_names=FEATURE_COLUMNS)
    ptl_reference = load_reference_run(args.ptl_reference_dir)
    comparison_summary = BASE_BENCH.build_comparison_summary(metrics, ptl_reference)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"base_processed_daily_{args.station_name}_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    model_path = run_dir / "model.pth"
    model_weights_saved = BASE_BENCH.BASE_FINETUNE.should_save_finetune_model_weights()
    if model_weights_saved:
        torch.save(model.state_dict(), model_path)
    else:
        print(f"跳过保存模型权重: {model_path}")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(run_dir / "metrics.csv")
    BASE_BENCH.save_predictions(run_dir / "predictions.csv", preds, targets, timestamps)

    meta = {
        "station_name": args.station_name,
        "feature_columns": FEATURE_COLUMNS,
        "data_path": str(args.data_path),
        "pretrain_dir": str(pretrain_dir),
        "ptl_reference_dir": str(args.ptl_reference_dir) if args.ptl_reference_dir else None,
        "device": str(device),
        "seed": int(args.seed),
        "records": int(len(frame)),
        "time_start": args.time_start,
        "time_end": args.time_end,
        "train_end": data_bundle["train_end"],
        "val_end": data_bundle["val_end"],
        "raw_seq_len": int(args.raw_seq_len),
        "model_seq_len": int(model_seq_len),
        "pred_len": int(args.pred_len),
        "resize_mode": args.resize_mode,
        "train_windows": int(len(datasets["train"])),
        "val_windows": int(len(datasets["val"])),
        "test_windows": int(len(datasets["test"])),
        "raw_gap_records": int(data_bundle["raw_invalid_mask"].sum()),
        "input_invalid_records": int(data_bundle["input_invalid_mask"].sum()),
        "target_invalid_records": int(data_bundle["target_invalid_mask"].sum()),
        "gap_stats": data_bundle["gap_stats"],
        "loaded_pretrain_keys": int(len(matched_state)),
        "freeze_stats": freeze_stats,
        "model_weights_saved": model_weights_saved,
        "model_weights_path": str(model_path) if model_weights_saved else None,
        "best_epoch": best_info["best_epoch"],
        "best_val_loss": best_info["best_val_loss"],
        "best_val_nse": best_info["best_val_nse"],
        "test_loss": float(test_loss),
        "train_seconds": train_seconds,
        "metrics": metrics["__overall__"],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "ptl_reference.json").write_text(
        json.dumps(ptl_reference, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("Benchmark 完成")
    print("=" * 68)
    print(f"最佳验证损失: {best_info['best_val_loss']:.6f}")
    print(f"最佳验证 NSE: {best_info['best_val_nse']:.6f}")
    print(f"测试损失: {test_loss:.6f}")
    print(f"测试整体 NSE: {metrics['__overall__']['NSE']:.6f}")
    print(f"结果目录: {run_dir}")
    if comparison_summary["ptl_overall"] is not None:
        print("\n与 PTL 对照:")
        print(f"  Base NSE: {comparison_summary['base_overall']['NSE']:.6f}")
        print(f"  PTL  NSE: {comparison_summary['ptl_overall']['NSE']:.6f}")
        delta_nse = comparison_summary["overall_delta_base_minus_ptl"]["NSE"]
        print(f"  Delta   : {delta_nse:.6f}")


if __name__ == "__main__":
    main()
