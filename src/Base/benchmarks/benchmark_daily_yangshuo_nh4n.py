import argparse
import importlib.util
import json
import os
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


BASE_FINETUNE = load_module("base_finetune_module", TRAINING_DIR / "finetune.py")
PTL_CORE = load_module("ptl_progressive_core_module", REPO_ROOT / "src" / "PTL" / "progressive_core.py")


FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "water_quality_processed_2023_2025" / "daily" / "阳朔.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "base" / "finetune" / "daily_benchmarks"
DEFAULT_PTL_REFERENCE_DIR = (
    REPO_ROOT
    / "results"
    / "ptl"
    / "finetune"
    / "comparisons"
    / "20260320_094921_预设与默认全面对比_seed42"
    / "04_目标75_v2_软缺口6步"
    / "阶段3_日级训练"
)


class DailyBenchmarkDataset(Dataset):
    def __init__(
        self,
        values,
        timestamps,
        raw_seq_len,
        model_seq_len,
        raw_pred_len,
        model_pred_len,
        split,
        train_ratio,
        val_ratio,
        resize_mode="linear",
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
        raise ValueError("日级 benchmark 数据中存在 NaN，当前脚本要求输入为完整序列。")
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
    scaled_values,
    timestamps,
    raw_seq_len,
    model_seq_len,
    raw_pred_len,
    model_pred_len,
    batch_size,
    train_ratio,
    val_ratio,
    resize_mode="linear",
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
    config = BASE_FINETUNE.FinetuneConfig()
    config.device = device
    config.input_dim = len(FEATURE_COLUMNS)
    config.feature_dim = 0
    config.n_in = int(pretrain_config["n_in"])
    config.n_out = 1
    config.hidden_size = int(pretrain_config["hidden_size"])
    config.num_heads = int(pretrain_config["num_heads"])
    config.e_layer = int(pretrain_config["e_layer"])
    config.prompt_num = 1
    return config, BASE_FINETUNE.Prompt_MultiTransformer(
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


def freeze_backbone(model, freeze_ratio: float):
    transformer_params = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("transformers")
    ]
    freeze_count = int(len(transformer_params) * float(freeze_ratio))
    for index, (_, parameter) in enumerate(transformer_params):
        if index < freeze_count:
            parameter.requires_grad = False
    return {
        "freeze_ratio": float(freeze_ratio),
        "frozen_param_count": int(freeze_count),
        "transformer_param_count": int(len(transformer_params)),
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
                f"[BaseDailyBenchmark] Epoch {epoch + 1}/{args.epochs} "
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
                f"[BaseDailyBenchmark] Early stopping at epoch {epoch + 1} "
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


def load_reference_run(reference_dir: Path):
    metrics_path = reference_dir / "评估指标_metrics.csv"
    meta_path = reference_dir / "运行元信息_meta.json"
    predictions_path = reference_dir / "预测明细_predictions.csv"

    reference_metrics = None
    if metrics_path.exists():
        reference_metrics = (
            pd.read_csv(metrics_path, index_col=0)
            .replace({np.nan: None})
            .to_dict(orient="index")
        )

    reference_meta = None
    if meta_path.exists():
        reference_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    reference_prediction_count = None
    if predictions_path.exists():
        reference_prediction_count = int(len(pd.read_csv(predictions_path)))

    return {
        "reference_dir": str(reference_dir),
        "metrics": reference_metrics,
        "meta": reference_meta,
        "prediction_count": reference_prediction_count,
    }


def build_comparison_summary(base_metrics, ptl_reference):
    summary = {
        "base_overall": base_metrics.get("__overall__"),
        "ptl_overall": None,
        "overall_delta_base_minus_ptl": None,
        "per_feature_delta_base_minus_ptl": {},
    }

    ptl_metrics = ptl_reference.get("metrics") or {}
    if "__overall__" in ptl_metrics:
        summary["ptl_overall"] = ptl_metrics["__overall__"]
        summary["overall_delta_base_minus_ptl"] = {
            key: (
                float(base_metrics["__overall__"][key]) - float(ptl_metrics["__overall__"][key])
                if key in base_metrics["__overall__"] and key in ptl_metrics["__overall__"]
                and base_metrics["__overall__"][key] is not None
                and ptl_metrics["__overall__"][key] is not None
                else None
            )
            for key in ("MSE", "MAE", "RMSE", "NSE", "MAPE")
        }

    for feature_name in FEATURE_COLUMNS:
        base_feature = base_metrics.get(feature_name)
        ptl_feature = ptl_metrics.get(feature_name)
        if base_feature is None or ptl_feature is None:
            continue
        summary["per_feature_delta_base_minus_ptl"][feature_name] = {
            key: (
                float(base_feature[key]) - float(ptl_feature[key])
                if key in base_feature and key in ptl_feature
                and base_feature[key] is not None
                and ptl_feature[key] is not None
                else None
            )
            for key in ("MAE", "RMSE", "NSE", "MAPE")
        }

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Base 模型阳朔近三年 NH4N 日预测 benchmark")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pretrain-dir", type=Path, default=None)
    parser.add_argument("--ptl-reference-dir", type=Path, default=DEFAULT_PTL_REFERENCE_DIR)
    parser.add_argument("--station-name", type=str, default="阳朔")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--raw-seq-len", type=int, default=28)
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
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    pretrain_dir = args.pretrain_dir
    if pretrain_dir is None:
        latest_pretrain_dir = BASE_FINETUNE.find_latest_pretrain_run(BASE_FINETUNE.PRETRAIN_RUNS_DIR)
        if latest_pretrain_dir is None:
            raise FileNotFoundError("未找到可用的 Base 预训练目录。")
        pretrain_dir = Path(latest_pretrain_dir)

    PTL_CORE.set_seed(args.seed)
    device = PTL_CORE.infer_device()

    frame = read_station_frame(args.data_path)
    data_bundle = build_scaled_series(frame, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    pretrain_config = load_pretrain_metadata(pretrain_dir)
    model_seq_len = int(pretrain_config["n_in"])

    datasets, loaders = build_loaders(
        scaled_values=data_bundle["scaled_values"],
        timestamps=data_bundle["timestamps"],
        raw_seq_len=args.raw_seq_len,
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
    freeze_stats = freeze_backbone(model, freeze_ratio=args.freeze_ratio)

    print("=" * 60)
    print("Base Daily NH4N Benchmark")
    print("=" * 60)
    print(f"数据文件: {args.data_path}")
    print(f"PTL 对照目录: {args.ptl_reference_dir}")
    print(f"预训练目录: {pretrain_dir}")
    print(f"站点: {args.station_name}")
    print(f"特征: {FEATURE_COLUMNS}")
    print(
        f"切分: train/val/test = {args.train_ratio:.2f}/{args.val_ratio:.2f}/{1.0 - args.train_ratio - args.val_ratio:.2f}"
    )
    print(
        f"窗口: raw_seq_len={args.raw_seq_len}, model_seq_len={model_seq_len}, pred_len={args.pred_len}"
    )
    print(
        f"样本数: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}"
    )
    print(f"迁移权重数: {len(matched_state)}")
    print(
        f"冻结参数: {freeze_stats['frozen_param_count']}/{freeze_stats['transformer_param_count']}"
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
    ptl_reference = load_reference_run(args.ptl_reference_dir)
    comparison_summary = build_comparison_summary(metrics, ptl_reference)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"base_daily_{args.station_name}_nh4n_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    model_path = run_dir / "model.pth"
    model_weights_saved = BASE_FINETUNE.should_save_finetune_model_weights()
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
        "data_path": str(args.data_path),
        "pretrain_dir": str(pretrain_dir),
        "ptl_reference_dir": str(args.ptl_reference_dir),
        "device": str(device),
        "seed": int(args.seed),
        "records": int(len(frame)),
        "train_end": data_bundle["train_end"],
        "val_end": data_bundle["val_end"],
        "raw_seq_len": int(args.raw_seq_len),
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

    print("\n" + "=" * 60)
    print("Benchmark 完成")
    print("=" * 60)
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
