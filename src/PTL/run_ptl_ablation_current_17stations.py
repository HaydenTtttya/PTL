from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finetune import (
    FINETUNE_RUNS_DIR,
    FinetuneConfig,
    build_finetune_preset,
    build_shared_stage_time_ranges,
    build_stage_nh4n_two_stage_config,
    build_stage_specs,
    build_model,
    find_latest_pretrain_run,
    fit_stage_postprocess,
    load_stage_data,
    save_stage_outputs,
    apply_stage_postprocess,
)
from progressive_core import (
    compute_per_feature_metrics,
    evaluate_model,
    fit_model,
    load_matching_weights,
    snapshot_state_dict,
    set_seed,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PRESET_NAME = "core3_progressive_v2pretrain_v2"
FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "water_quality_processed_2021_2024"
DEFAULT_SUMMARY_ROOT = (
    REPO_ROOT
    / "results"
    / "summary"
    / "current_all_tested_stations_overall_nse"
)
DEFAULT_STATION_CLASS_PATH = DEFAULT_SUMMARY_ROOT / "均衡十五站方案_新增两站" / "站点分类.csv"
DEFAULT_FULL_PTL_REFERENCE = (
    DEFAULT_SUMMARY_ROOT
    / "均衡十五站方案_新增两站"
    / "PTL_GPU补跑_带权重_17站对比.csv"
)
DEFAULT_PRETRAIN_ROOTS = (
    REPO_ROOT / "results" / "cross_station" / "pretrain" / "v2_full_runs",
    REPO_ROOT / "results" / "cross_station" / "pretrain" / "v2_competition_runs",
    REPO_ROOT / "results" / "cross_station" / "pretrain" / "v2_runs",
    REPO_ROOT / "results" / "cross_station" / "pretrain" / "v2_arch_runs",
)


ABLATION_SPECS = [
    {
        "key": "ptl_full",
        "label": "PTL full: cross-station pretrain + progressive transfer",
        "source": "reference",
        "uses_cross_station_pretrain": True,
        "uses_progressive_transfer": True,
        "trains_weekly_stage": True,
        "trains_4d_stage": True,
        "final_stage_receives_stage_transfer": True,
        "planned_epochs": 180,
    },
    {
        "key": "scratch_direct_daily",
        "label": "No pretrain: direct daily fine-tune from random init",
        "source": "train",
        "stage_mode": "daily_only",
        "init_mode": "random",
        "uses_cross_station_pretrain": False,
        "uses_progressive_transfer": False,
        "trains_weekly_stage": False,
        "trains_4d_stage": False,
        "final_stage_receives_stage_transfer": False,
        "planned_epochs": 80,
    },
    {
        "key": "pretrain_direct_daily",
        "label": "Cross-station pretrain + direct daily fine-tune",
        "source": "train",
        "stage_mode": "daily_only",
        "init_mode": "pretrain",
        "uses_cross_station_pretrain": True,
        "uses_progressive_transfer": False,
        "trains_weekly_stage": False,
        "trains_4d_stage": False,
        "final_stage_receives_stage_transfer": False,
        "planned_epochs": 80,
    },
    {
        "key": "no_progressive_handoff",
        "label": "No progressive transfer: weekly/4d/daily trained but no stage handoff",
        "source": "train",
        "stage_mode": "all_stages_without_handoff",
        "init_mode": "pretrain_each_stage",
        "uses_cross_station_pretrain": True,
        "uses_progressive_transfer": False,
        "trains_weekly_stage": True,
        "trains_4d_stage": True,
        "final_stage_receives_stage_transfer": False,
        "planned_epochs": 180,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PTL ablations for the current 17-station 2023-2024 summary set.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--station-class-path", type=Path, default=DEFAULT_STATION_CLASS_PATH)
    parser.add_argument("--full-ptl-reference", type=Path, default=DEFAULT_FULL_PTL_REFERENCE)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="PTL_ablation",
        help="Subdirectory under --summary-root for ablation summary tables.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(FINETUNE_RUNS_DIR) / "ablation_current_17stations_2023_2024",
    )
    parser.add_argument("--pretrain-dir", type=Path, default=None)
    parser.add_argument("--station", action="append", default=[])
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--time-start", default="2023-01-01 00:00:00")
    parser.add_argument("--time-end", default="2024-12-31 23:59:59")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_pretrain_dir(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        explicit_path = explicit_path.resolve()
        if not (explicit_path / "config.json").exists() or not (explicit_path / "model.pth").exists():
            raise FileNotFoundError(f"Invalid pretrain dir: {explicit_path}")
        return explicit_path

    for root in DEFAULT_PRETRAIN_ROOTS:
        latest = find_latest_pretrain_run(str(root))
        if latest is not None:
            return Path(latest).resolve()

    searched = ", ".join(str(path) for path in DEFAULT_PRETRAIN_ROOTS)
    raise FileNotFoundError(f"No valid V2 pretrain run found under: {searched}")


def load_station_class(path: Path, requested_stations: list[str], max_stations: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "站点顺序" in frame.columns:
        frame = frame.sort_values("站点顺序")
    if requested_stations:
        requested = list(dict.fromkeys(requested_stations))
        indexed = frame.set_index("station", drop=False)
        missing = [name for name in requested if name not in indexed.index]
        if missing:
            raise ValueError(f"Requested station(s) missing from station class file: {missing}")
        frame = pd.DataFrame([indexed.loc[name] for name in requested]).reset_index(drop=True)
    if max_stations is not None:
        frame = frame.head(int(max_stations))
    return frame.reset_index(drop=True)


def build_custom_config(args, station_names: list[str], save_dir: Path) -> dict:
    custom_config = copy.deepcopy(build_finetune_preset(PRESET_NAME))
    if custom_config is None:
        raise ValueError(f"Unable to build preset: {PRESET_NAME}")

    custom_config.update(
        {
            "data_root": str(args.data_root.resolve()),
            "save_dir": str(save_dir.resolve()),
            "feature_columns": list(FEATURE_COLUMNS),
            "target_station_names": list(station_names),
            "time_start": args.time_start,
            "time_end": args.time_end,
            "stage_time_ranges": build_shared_stage_time_ranges(args.time_start, args.time_end),
            "save_model_weights": bool(args.save_model_weights),
        }
    )

    custom_config["invalid_window_policy"] = "all"
    for stage in custom_config["progressive_stages"]:
        stage["invalid_window_policy"] = "all"
        stage["soft_gap_max_steps"] = None
    return custom_config


def make_runtime_config(custom_config: dict, pretrain_config: dict, pretrain_dir: Path) -> FinetuneConfig:
    config = FinetuneConfig()
    config.pretrain_model_dir = str(pretrain_dir)
    for key, value in custom_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.feature_columns = list(config.feature_columns)
    config.input_dim = len(config.feature_columns)
    config.hidden_size = int(pretrain_config["hidden_size"])
    config.num_heads = int(pretrain_config["num_heads"])
    config.e_layer = int(pretrain_config["e_layer"])
    config.model_seq_len = int(pretrain_config.get("model_seq_len", config.model_seq_len))
    return config


def get_stage_specs_for_variant(config: FinetuneConfig, variant: dict) -> list[dict]:
    stage_specs = build_stage_specs(config)
    if variant.get("stage_mode") == "daily_only":
        return [stage for stage in stage_specs if stage["name"] == "stage3_daily"]
    return stage_specs


def read_metrics_csv(path: Path) -> dict:
    frame = pd.read_csv(path, index_col=0)
    return frame.replace({np.nan: None}).to_dict(orient="index")


def focus_summary(metrics: dict) -> dict:
    nse_values = [float(metrics[name]["NSE"]) for name in FOCUS_FEATURES if name in metrics]
    rmse_values = [float(metrics[name]["RMSE"]) for name in FOCUS_FEATURES if name in metrics]
    mae_values = [float(metrics[name]["MAE"]) for name in FOCUS_FEATURES if name in metrics]
    return {
        "focus_mean_nse": float(np.mean(nse_values)) if nse_values else np.nan,
        "focus_mean_rmse": float(np.mean(rmse_values)) if rmse_values else np.nan,
        "focus_mean_mae": float(np.mean(mae_values)) if mae_values else np.nan,
    }


def flatten_metrics(metrics: dict) -> dict:
    row = {}
    overall = metrics.get("__overall__", {})
    row.update(
        {
            "overall_nse": overall.get("NSE"),
            "overall_rmse": overall.get("RMSE"),
            "overall_mae": overall.get("MAE"),
        }
    )
    row.update(focus_summary(metrics))
    for feature_name in FEATURE_COLUMNS:
        feature_metrics = metrics.get(feature_name, {})
        row[f"{feature_name}_nse"] = feature_metrics.get("NSE")
        row[f"{feature_name}_rmse"] = feature_metrics.get("RMSE")
        row[f"{feature_name}_mae"] = feature_metrics.get("MAE")
    return row


def build_common_row(station_row: dict, variant: dict) -> dict:
    return {
        "station": station_row["station"],
        "river_reach": station_row.get("river_reach", ""),
        "river_type": station_row.get("river_type", ""),
        "comparison_group": station_row.get("comparison_group", ""),
        "verified_waterbody": station_row.get("verified_waterbody", ""),
        "ablation_key": variant["key"],
        "ablation_label": variant["label"],
        "uses_cross_station_pretrain": variant["uses_cross_station_pretrain"],
        "uses_progressive_transfer": variant["uses_progressive_transfer"],
        "trains_weekly_stage": variant["trains_weekly_stage"],
        "trains_4d_stage": variant["trains_4d_stage"],
        "final_stage_receives_stage_transfer": variant["final_stage_receives_stage_transfer"],
        "planned_epochs": variant["planned_epochs"],
        "same_ptl_architecture": True,
    }


def load_full_reference_rows(reference_path: Path, station_class: pd.DataFrame) -> list[dict]:
    reference = pd.read_csv(reference_path, encoding="utf-8-sig")
    stage_meta_by_station = dict(zip(reference["station"], reference["stage3_meta"]))
    rows = []
    variant = ABLATION_SPECS[0]
    for station_row in station_class.to_dict(orient="records"):
        station = station_row["station"]
        meta_path_text = stage_meta_by_station.get(station)
        if not meta_path_text:
            raise FileNotFoundError(f"Full PTL reference missing station: {station}")
        meta_path = Path(meta_path_text)
        metrics_path = meta_path.parent / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Full PTL metrics missing: {metrics_path}")
        metrics = read_metrics_csv(metrics_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        row = build_common_row(station_row, variant)
        row.update(flatten_metrics(metrics))
        row.update(
            {
                "status": "completed",
                "run_dir": str(meta_path.parents[1]),
                "stage3_dir": str(meta_path.parent),
                "metrics_path": str(metrics_path),
                "meta_path": str(meta_path),
                "train_windows": meta.get("train_windows"),
                "val_windows": meta.get("val_windows"),
                "test_windows": meta.get("test_windows"),
                "soft_gap_max_steps": meta.get("soft_gap_max_steps"),
                "invalid_window_policy": meta.get("invalid_window_policy"),
                "best_epoch": meta.get("best_epoch"),
                "train_seconds": meta.get("train_seconds"),
            }
        )
        rows.append(row)
    return rows


def stage_init_state(
    variant: dict,
    stage_index: int,
    pretrain_state: dict,
    previous_stage_state: dict | None,
) -> dict | None:
    init_mode = variant.get("init_mode")
    if init_mode == "random":
        return None
    if init_mode == "pretrain":
        return pretrain_state
    if init_mode == "pretrain_each_stage":
        return pretrain_state
    if init_mode == "progressive":
        return pretrain_state if stage_index == 0 else previous_stage_state
    raise ValueError(f"Unsupported init_mode: {init_mode}")


def run_one_stage(
    config: FinetuneConfig,
    station_name: str,
    stage: dict,
    init_state: dict | None,
    stage_dir: Path,
    variant: dict,
) -> tuple[dict, dict]:
    stage_data = load_stage_data(config, station_name, stage)
    if stage_data is None:
        raise RuntimeError(f"{station_name} has no runnable data for {stage['name']}")

    nh4n_two_stage_config = build_stage_nh4n_two_stage_config(
        stage,
        stage_data["scaler"],
        config.feature_columns,
    )
    stage_model = build_model(
        config,
        {**stage, "input_dim": stage_data["input_dim"]},
        nh4n_two_stage_config=nh4n_two_stage_config,
    )
    matched_keys = {}
    if init_state is not None:
        matched_keys = load_matching_weights(
            stage_model,
            init_state,
            skip_prefixes=stage.get("skip_weight_prefixes"),
        )

    train_start = time.time()
    stage_model, history, best_stats = fit_model(
        model=stage_model,
        train_loader=stage_data["train_loader"],
        val_loader=stage_data["val_loader"],
        device=config.device,
        epochs=stage.get("epochs", config.epochs),
        base_lr=stage.get("base_lr", config.base_lr),
        epsilon=config.epsilon,
        weight_decay=stage.get("weight_decay", config.weight_decay),
        lr_milestones=stage.get("lr_milestones", config.lr_milestones),
        lr_decay_ratio=stage.get("lr_decay_ratio", config.lr_decay_ratio),
        max_grad_norm=config.max_grad_norm,
        log_prefix=f"{variant['key']}:{station_name}:{stage['name']}",
        loss_name=stage.get("loss_name", config.loss_name),
        nse_weight=stage.get("nse_weight", config.nse_weight),
        loss_feature_weights=stage.get("loss_feature_weights"),
        multitask_config=nh4n_two_stage_config,
        monitor_metric=stage.get("monitor_metric", config.monitor_metric),
        monitor_feature=stage.get("monitor_feature"),
        monitor_feature_weights=stage.get("monitor_feature_weights"),
        early_stopping_patience=stage.get(
            "early_stopping_patience",
            config.early_stopping_patience,
        ),
        early_stopping_min_delta=stage.get(
            "early_stopping_min_delta",
            config.early_stopping_min_delta,
        ),
        scheduler_name=stage.get("scheduler_name", config.scheduler_name),
        scheduler_patience=stage.get("scheduler_patience", config.scheduler_patience),
        scheduler_min_lr=stage.get("scheduler_min_lr", config.scheduler_min_lr),
        freeze_backbone_epochs=stage.get(
            "freeze_backbone_epochs",
            config.freeze_backbone_epochs,
        ),
        feature_names=config.feature_columns,
    )
    train_seconds = time.time() - train_start

    postprocess_params = None
    postprocess_val_metrics = None
    if stage.get("postprocess_mode"):
        _, val_preds, val_targets, val_timestamps = evaluate_model(
            stage_model,
            stage_data["val_loader"],
            config.device,
            scaler=stage_data["scaler"],
        )
        postprocess_params = fit_stage_postprocess(
            stage,
            stage_data["frame"],
            val_preds,
            val_targets,
            val_timestamps,
            config.feature_columns,
        )
        if postprocess_params is not None:
            val_preds = apply_stage_postprocess(
                postprocess_params,
                stage_data["frame"],
                val_preds,
                val_timestamps,
                config.feature_columns,
            )
            postprocess_val_metrics = compute_per_feature_metrics(
                val_preds,
                val_targets,
                config.feature_columns,
            )

    test_loss, preds, targets, timestamps = evaluate_model(
        stage_model,
        stage_data["test_loader"],
        config.device,
        scaler=stage_data["scaler"],
    )
    if postprocess_params is not None:
        preds = apply_stage_postprocess(
            postprocess_params,
            stage_data["frame"],
            preds,
            timestamps,
            config.feature_columns,
        )
    metrics = compute_per_feature_metrics(preds, targets, config.feature_columns)
    stage_dir.mkdir(parents=True, exist_ok=True)
    model_weights_path = stage_dir / "model.pth"
    model_weights_saved = bool(config.save_model_weights)
    if model_weights_saved:
        torch.save(stage_model.state_dict(), model_weights_path)
    with (stage_dir / "scaler.pkl").open("wb") as file:
        pickle.dump(stage_data["scaler"], file)

    meta = {
        "station_name": station_name,
        "ablation_key": variant["key"],
        "ablation_label": variant["label"],
        "stage_name": stage["name"],
        "resolution": stage["resolution"],
        "window_days": stage["window_days"],
        "raw_seq_len": stage["raw_seq_len"],
        "model_seq_len": stage["model_seq_len"],
        "pred_steps": stage["pred_steps"],
        "resize_mode": stage.get("resize_mode", config.resize_mode),
        "records": stage_data["records"],
        "invalid_records": stage_data["invalid_records"],
        "valid_records": stage_data["valid_records"],
        "input_dim": stage_data["input_dim"],
        "input_feature_columns": stage_data["input_feature_columns"],
        "feature_columns": list(config.feature_columns),
        "model_weights_saved": model_weights_saved,
        "model_weights_path": str(model_weights_path) if model_weights_saved else None,
        "train_windows": stage_data["train_windows"],
        "train_candidate_windows": stage_data["train_candidate_windows"],
        "train_filtered_windows": stage_data["train_filtered_windows"],
        "val_windows": stage_data["val_windows"],
        "val_candidate_windows": stage_data["val_candidate_windows"],
        "val_filtered_windows": stage_data["val_filtered_windows"],
        "test_windows": stage_data["test_windows"],
        "test_candidate_windows": stage_data["test_candidate_windows"],
        "test_filtered_windows": stage_data["test_filtered_windows"],
        "time_start": stage_data["time_start"],
        "time_end": stage_data["time_end"],
        "stage_time_ranges": config.stage_time_ranges,
        "filter_invalid_windows": stage_data["filter_invalid_windows"],
        "invalid_window_policy": stage_data["invalid_window_policy"],
        "soft_gap_max_steps": stage_data["soft_gap_max_steps"],
        "input_invalid_records": stage_data["input_invalid_records"],
        "target_invalid_records": stage_data["target_invalid_records"],
        "loss_feature_weights": stage.get("loss_feature_weights"),
        "monitor_feature_weights": stage.get("monitor_feature_weights"),
        "nh4n_two_stage": nh4n_two_stage_config,
        "best_epoch": int(best_stats["epoch"]),
        "best_val_loss": float(best_stats["val_loss"]),
        "best_val_nse": float(best_stats["val_nse"]),
        "best_val_monitor_nse": float(best_stats["val_monitor_nse"]),
        "test_loss": float(test_loss),
        "test_nse": float(metrics["__overall__"]["NSE"]),
        "train_seconds": float(train_seconds),
        "matched_init_keys": int(len(matched_keys)),
        "monitor_metric": best_stats["monitor_metric"],
        "monitor_feature": best_stats["monitor_feature"],
        "best_monitor_feature_weights": best_stats["monitor_feature_weights"],
        "monitor_value": best_stats["monitor_value"],
        "train_sampler_stats": stage_data["train_sampler_stats"],
        "postprocess_params": postprocess_params,
        "postprocess_val_metrics": (
            postprocess_val_metrics["__overall__"]
            if postprocess_val_metrics is not None
            else None
        ),
        "use_temporal_adapter": config.use_temporal_adapter,
        "temporal_adapter_kernel_size": config.temporal_adapter_kernel_size,
        "metrics": metrics["__overall__"],
    }
    save_stage_outputs(
        stage_dir,
        history,
        preds,
        targets,
        timestamps,
        metrics,
        meta,
        config.feature_columns,
    )
    stage_result = {
        "stage_name": stage["name"],
        "best_epoch": int(best_stats["epoch"]),
        "best_val_loss": float(best_stats["val_loss"]),
        "best_val_nse": float(best_stats["val_nse"]),
        "test_loss": float(test_loss),
        "test_nse": float(metrics["__overall__"]["NSE"]),
        "save_dir": str(stage_dir),
        "metrics": metrics,
        "meta": meta,
    }
    return stage_result, snapshot_state_dict(stage_model)


def run_train_variant(
    args,
    variant: dict,
    station_class: pd.DataFrame,
    pretrain_dir: Path,
    pretrain_config: dict,
    pretrain_state: dict,
    batch_dir: Path,
) -> list[dict]:
    variant_dir = batch_dir / variant["key"]
    custom_config = build_custom_config(
        args,
        station_class["station"].tolist(),
        variant_dir,
    )
    config = make_runtime_config(custom_config, pretrain_config, pretrain_dir)
    stage_specs = get_stage_specs_for_variant(config, variant)

    rows = []
    for station_index, station_row in enumerate(station_class.to_dict(orient="records"), start=1):
        station_name = station_row["station"]
        print("\n" + "#" * 70)
        print(f"{variant['key']} [{station_index}/{len(station_class)}] {station_name}")
        print("#" * 70)
        set_seed(args.seed)
        station_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        station_dir = variant_dir / f"{variant['key']}_{station_name}_seed{args.seed}_{station_timestamp}"
        station_dir.mkdir(parents=True, exist_ok=True)

        previous_state = None
        stage_results = []
        station_failed = False
        error_message = ""
        for stage_index, stage in enumerate(stage_specs):
            try:
                init_state = stage_init_state(variant, stage_index, pretrain_state, previous_state)
                stage_dir = station_dir / stage["name"]
                stage_result, trained_state = run_one_stage(
                    config=config,
                    station_name=station_name,
                    stage=stage,
                    init_state=init_state,
                    stage_dir=stage_dir,
                    variant=variant,
                )
                stage_results.append(stage_result)
                if variant.get("init_mode") == "progressive":
                    previous_state = trained_state
                else:
                    previous_state = None
            except Exception as exc:  # noqa: BLE001
                station_failed = True
                error_message = repr(exc)
                print(f"[failed] {station_name} {stage['name']}: {error_message}")
                break

        summary = {
            "station_name": station_name,
            "ablation_key": variant["key"],
            "status": "failed" if station_failed else "completed",
            "error_message": error_message,
            "stages": [
                {
                    key: value
                    for key, value in stage_result.items()
                    if key not in {"metrics", "meta"}
                }
                for stage_result in stage_results
            ],
            "save_dir": str(station_dir),
        }
        (station_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        row = build_common_row(station_row, variant)
        row.update(
            {
                "status": summary["status"],
                "error_message": error_message,
                "run_dir": str(station_dir),
            }
        )
        final_result = stage_results[-1] if stage_results else None
        if final_result is not None:
            metrics = final_result["metrics"]
            meta = final_result["meta"]
            row.update(flatten_metrics(metrics))
            row.update(
                {
                    "stage3_dir": final_result["save_dir"],
                    "metrics_path": str(Path(final_result["save_dir"]) / "metrics.csv"),
                    "meta_path": str(Path(final_result["save_dir"]) / "meta.json"),
                    "train_windows": meta.get("train_windows"),
                    "val_windows": meta.get("val_windows"),
                    "test_windows": meta.get("test_windows"),
                    "soft_gap_max_steps": meta.get("soft_gap_max_steps"),
                    "invalid_window_policy": meta.get("invalid_window_policy"),
                    "best_epoch": meta.get("best_epoch"),
                    "train_seconds": sum(float(item["meta"].get("train_seconds", 0.0)) for item in stage_results),
                }
            )
        rows.append(row)
    return rows


def add_delta_columns(frame: pd.DataFrame) -> pd.DataFrame:
    full = frame[frame["ablation_key"] == "ptl_full"][
        ["station", "overall_nse", "focus_mean_nse"]
    ].rename(
        columns={
            "overall_nse": "full_overall_nse",
            "focus_mean_nse": "full_focus_mean_nse",
        }
    )
    merged = frame.merge(full, on="station", how="left")
    merged["delta_overall_nse_vs_full"] = merged["overall_nse"] - merged["full_overall_nse"]
    merged["delta_focus_mean_nse_vs_full"] = (
        merged["focus_mean_nse"] - merged["full_focus_mean_nse"]
    )
    return merged


def markdown_table(frame: pd.DataFrame) -> str:
    text_frame = frame.copy()
    for column in text_frame.columns:
        if pd.api.types.is_float_dtype(text_frame[column]):
            text_frame[column] = text_frame[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
            )
        else:
            text_frame[column] = text_frame[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )

    headers = list(text_frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in text_frame.to_dict(orient="records"):
        lines.append("| " + " | ".join(row[column] for column in headers) + " |")
    return "\n".join(lines)


def write_summary_tables(rows: list[dict], output_dir: Path, batch_dir: Path, pretrain_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame = add_delta_columns(frame)
    frame.to_csv(output_dir / "ptl_ablation_metrics_long.csv", index=False, encoding="utf-8-sig")

    wide = frame.pivot_table(
        index=["station", "river_reach", "river_type"],
        columns="ablation_key",
        values="overall_nse",
        aggfunc="first",
    ).reset_index()
    wide.to_csv(output_dir / "ptl_ablation_overall_nse_wide.csv", index=False, encoding="utf-8-sig")

    focus_wide = frame.pivot_table(
        index=["station", "river_reach", "river_type"],
        columns="ablation_key",
        values="focus_mean_nse",
        aggfunc="first",
    ).reset_index()
    focus_wide.to_csv(output_dir / "ptl_ablation_focus_nse_wide.csv", index=False, encoding="utf-8-sig")

    summary = (
        frame.groupby(["ablation_key", "ablation_label"], as_index=False)
        .agg(
            station_count=("station", "count"),
            completed_count=("status", lambda values: int((values == "completed").sum())),
            mean_overall_nse=("overall_nse", "mean"),
            median_overall_nse=("overall_nse", "median"),
            mean_focus_nse=("focus_mean_nse", "mean"),
            median_focus_nse=("focus_mean_nse", "median"),
            mean_delta_overall_vs_full=("delta_overall_nse_vs_full", "mean"),
            mean_delta_focus_vs_full=("delta_focus_mean_nse_vs_full", "mean"),
        )
        .sort_values("mean_overall_nse", ascending=False)
    )
    summary.to_csv(output_dir / "ptl_ablation_model_average_summary.csv", index=False, encoding="utf-8-sig")

    delta = frame[frame["ablation_key"] != "ptl_full"].copy()
    delta.to_csv(output_dir / "ptl_ablation_delta_vs_full.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "batch_dir": str(batch_dir),
        "pretrain_dir": str(pretrain_dir),
        "output_dir": str(output_dir),
        "ablation_specs": ABLATION_SPECS,
        "focus_features": FOCUS_FEATURES,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# PTL ablation summary",
        "",
        f"- Raw run directory: `{batch_dir}`",
        f"- Cross-station pretrain: `{pretrain_dir}`",
        f"- Stations: {int(frame['station'].nunique())}",
        "- Focus NSE averages CODMn, DO, and pH.",
        "",
        "## Model averages",
        "",
        summary[
            [
                "ablation_key",
                "completed_count",
                "mean_overall_nse",
                "mean_focus_nse",
                "mean_delta_overall_vs_full",
                "mean_delta_focus_vs_full",
            ]
        ].pipe(markdown_table),
        "",
        "## Overall NSE by station",
        "",
        markdown_table(wide),
        "",
    ]
    (output_dir / "ptl_ablation_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    pretrain_dir = resolve_pretrain_dir(args.pretrain_dir)
    pretrain_config = json.loads((pretrain_dir / "config.json").read_text(encoding="utf-8"))
    pretrain_state = torch.load(pretrain_dir / "model.pth", map_location="cpu")
    station_class = load_station_class(
        args.station_class_path,
        requested_stations=args.station,
        max_stations=args.max_stations,
    )
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = args.run_root / f"batch_ptl_ablation_{timestamp}"
    output_dir = args.summary_root / args.output_subdir
    batch_dir.mkdir(parents=True, exist_ok=True)

    setup = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "preset_name": PRESET_NAME,
        "data_root": str(args.data_root.resolve()),
        "station_class_path": str(args.station_class_path.resolve()),
        "full_ptl_reference": str(args.full_ptl_reference.resolve()),
        "summary_output_dir": str(output_dir.resolve()),
        "pretrain_dir": str(pretrain_dir),
        "stations": station_class["station"].tolist(),
        "time_start": args.time_start,
        "time_end": args.time_end,
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "save_model_weights": bool(args.save_model_weights),
    }
    (batch_dir / "run_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("PTL ablation runner")
    print(f"stations={len(station_class)} | seed={args.seed}")
    print(f"pretrain_dir={pretrain_dir}")
    print(f"batch_dir={batch_dir}")
    print(f"summary_dir={output_dir}")
    print("=" * 70)

    rows = load_full_reference_rows(args.full_ptl_reference, station_class)
    if args.dry_run:
        write_summary_tables(rows, output_dir, batch_dir, pretrain_dir)
        print(f"Dry run summary written: {output_dir}")
        return

    for variant in ABLATION_SPECS[1:]:
        variant_rows = run_train_variant(
            args=args,
            variant=variant,
            station_class=station_class,
            pretrain_dir=pretrain_dir,
            pretrain_config=pretrain_config,
            pretrain_state=pretrain_state,
            batch_dir=batch_dir,
        )
        rows.extend(variant_rows)
        write_summary_tables(rows, output_dir, batch_dir, pretrain_dir)

    write_summary_tables(rows, output_dir, batch_dir, pretrain_dir)
    print("=" * 70)
    print(f"PTL ablation complete. Summary written to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
