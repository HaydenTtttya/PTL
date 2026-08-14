from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from finetune import (
    FinetuneConfig,
    build_finetune_preset,
    build_shared_stage_time_ranges,
    main as finetune_main,
)
from model_agnostic_backbones import (
    normalize_backbone_name,
    normalize_model_agnostic_interface,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
BACKBONE_LABELS = {
    "mlp": "MLP",
    "cnn": "CNN",
    "lstm": "LSTM",
    "bilstm": "Bi-LSTM",
    "cnn_lstm": "CNN-LSTM",
}
PRESET_NAME = "core3_progressive_v2pretrain_v2"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "water_quality_processed_2021_2024"
DEFAULT_CURRENT_SUMMARY = (
    REPO_ROOT
    / "results"
    / "summary"
    / "current_all_tested_stations_overall_nse"
    / "均衡十五站方案_新增两站"
)
DEFAULT_STATION_CLASS = DEFAULT_CURRENT_SUMMARY / "站点分类.csv"
DEFAULT_DIRECT_BASELINE_LONG = DEFAULT_CURRENT_SUMMARY / "模型对比长表.csv"
DEFAULT_PRETRAIN_ROOT = (
    REPO_ROOT / "results" / "cross_station" / "pretrain" / "v2_model_agnostic"
)
DEFAULT_FINETUNE_ROOT = (
    REPO_ROOT / "results" / "ptl" / "finetune" / "runs" / "model_agnostic_17stations"
)
DEFAULT_DIRECT_ROOT = (
    REPO_ROOT / "results" / "base" / "fair_compare" / "model_agnostic_feature_token"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_CURRENT_SUMMARY / "模型无关实验"
OPTIMIZATION_PROFILES = (
    "default",
    "lstm_direct_daily_v1",
    "lstm_head_warmup_v2",
    "lstm_blend50_v3",
    "lstm_blend25_v4",
    "joint_backbone_adaptive_v1",
    "unified_feature_token_v1",
    "unified_feature_token_residual_v2",
    "unified_feature_token_residual_adaptive_v3",
    "unified_feature_token_safe_transfer_v4",
)


def profile_interface(profile: str) -> str:
    if profile == "unified_feature_token_v1":
        return "feature_token_v1"
    if profile in {
        "unified_feature_token_residual_v2",
        "unified_feature_token_residual_adaptive_v3",
        "unified_feature_token_safe_transfer_v4",
    }:
        return "feature_token_residual_v2"
    return "legacy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run seed-42 CNN/LSTM cross-station pretraining and the same "
            "weekly-to-4d-to-daily PTL flow used by the current 17-station experiment."
        )
    )
    parser.add_argument(
        "--step",
        choices=("all", "pretrain", "direct", "finetune", "summary"),
        default="all",
    )
    parser.add_argument(
        "--backbone",
        action="append",
        choices=tuple(BACKBONE_LABELS),
        default=[],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--station-class", type=Path, default=DEFAULT_STATION_CLASS)
    parser.add_argument("--direct-baseline-long", type=Path, default=DEFAULT_DIRECT_BASELINE_LONG)
    parser.add_argument("--pretrain-root", type=Path, default=DEFAULT_PRETRAIN_ROOT)
    parser.add_argument("--finetune-root", type=Path, default=DEFAULT_FINETUNE_ROOT)
    parser.add_argument("--direct-root", type=Path, default=DEFAULT_DIRECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--time-start", default="2023-01-01 00:00:00")
    parser.add_argument("--time-end", default="2024-12-31 23:59:59")
    parser.add_argument("--station", action="append", default=[])
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--save-model-weights", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--optimization-profile",
        choices=OPTIMIZATION_PROFILES,
        default="default",
    )

    parser.add_argument("--worker-finetune", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-direct", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-pretrain-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-station", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def selected_backbones(args: argparse.Namespace) -> list[str]:
    return list(dict.fromkeys(args.backbone or ["cnn", "lstm"]))


def load_stations(args: argparse.Namespace) -> list[str]:
    frame = pd.read_csv(args.station_class, encoding="utf-8-sig")
    if "站点顺序" in frame.columns:
        frame = frame.sort_values("站点顺序")
    stations = frame["station"].astype(str).tolist()
    if args.station:
        requested = list(dict.fromkeys(args.station))
        missing = [station for station in requested if station not in stations]
        if missing:
            raise ValueError(f"Station(s) missing from classification table: {missing}")
        stations = requested
    if args.max_stations is not None:
        stations = stations[: int(args.max_stations)]
    if not stations:
        raise ValueError("No target stations were selected.")
    return stations


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_completed_pretrain(args: argparse.Namespace, backbone: str) -> Path | None:
    root = args.pretrain_root / backbone
    candidates = sorted(
        root.glob(f"pretrain_cross_station_v2_{backbone}_*_seed{args.seed}_*/config.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for config_path in candidates:
        model_path = config_path.parent / "model.pth"
        if not model_path.exists():
            continue
        config = read_json(config_path)
        if (
            normalize_backbone_name(config.get("backbone_name")) == backbone
            and normalize_model_agnostic_interface(
                config.get("model_agnostic_interface", "legacy")
            )
            == profile_interface(args.optimization_profile)
        ):
            return config_path.parent
    return None


def find_completed_finetune(
    args: argparse.Namespace,
    backbone: str,
    station: str,
) -> Path | None:
    root = finetune_run_root(args, backbone)
    candidates = sorted(
        root.glob(f"progressive_{station}_seed{args.seed}_*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in candidates:
        summary = read_json(summary_path)
        stage3_dir = summary_path.parent / "stage3_daily"
        meta_path = stage3_dir / "meta.json"
        metrics_path = stage3_dir / "metrics.csv"
        if summary.get("status") != "completed" or not meta_path.exists() or not metrics_path.exists():
            continue
        meta = read_json(meta_path)
        if (
            normalize_backbone_name(meta.get("backbone_name")) == backbone
            and meta.get("optimization_profile", "default")
            == args.optimization_profile
            and normalize_model_agnostic_interface(
                meta.get("model_agnostic_interface", "legacy")
            )
            == profile_interface(args.optimization_profile)
        ):
            return summary_path.parent
    return None


def direct_run_root(args: argparse.Namespace, backbone: str) -> Path:
    return args.direct_root / args.optimization_profile / backbone


def find_completed_direct(
    args: argparse.Namespace,
    backbone: str,
    station: str,
) -> Path | None:
    root = direct_run_root(args, backbone)
    candidates = sorted(
        root.glob(f"progressive_{station}_seed{args.seed}_*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in candidates:
        summary = read_json(summary_path)
        stage_dir = summary_path.parent / "stage3_daily"
        meta_path = stage_dir / "meta.json"
        metrics_path = stage_dir / "metrics.csv"
        if summary.get("status") != "completed" or not meta_path.exists() or not metrics_path.exists():
            continue
        meta = read_json(meta_path)
        if (
            normalize_backbone_name(meta.get("backbone_name")) == backbone
            and not bool(meta.get("uses_pretraining"))
            and normalize_model_agnostic_interface(
                meta.get("model_agnostic_interface", "legacy")
            )
            == profile_interface(args.optimization_profile)
        ):
            return summary_path.parent
    return None


def finetune_run_root(args: argparse.Namespace, backbone: str) -> Path:
    if args.optimization_profile == "default":
        return args.finetune_root / backbone
    return (
        args.finetune_root
        / "optimization"
        / args.optimization_profile
        / backbone
    )


def apply_optimization_profile(config: dict, profile: str, backbone: str) -> None:
    if profile in {
        "default",
        "unified_feature_token_v1",
        "unified_feature_token_residual_v2",
    }:
        return
    if profile in {
        "unified_feature_token_residual_adaptive_v3",
        "unified_feature_token_safe_transfer_v4",
    }:
        daily_stages = [
            stage
            for stage in config["progressive_stages"]
            if stage["resolution"] == "daily"
        ]
        if len(daily_stages) != 1:
            raise ValueError("Expected exactly one daily stage in the PTL preset.")
        daily_stages[0].update(
            {
                "epochs": 120,
                "base_lr": 2.5e-4,
                "weight_decay": 1e-4,
                "freeze_backbone_epochs": 0,
                "early_stopping_patience": 20,
                "early_stopping_min_delta": 1e-4,
                "scheduler_patience": 6,
                "scheduler_min_lr": 1e-5,
            }
        )
        if profile == "unified_feature_token_safe_transfer_v4":
            daily_stages[0].update(
                {
                    "load_weight_blend_alpha": 0.1,
                    "reset_model_seed": 42,
                }
            )
        return
    effective_profile = profile
    if profile == "joint_backbone_adaptive_v1":
        if backbone == "cnn":
            return
        effective_profile = "lstm_blend50_v3"
    lstm_profiles = {
        "lstm_direct_daily_v1",
        "lstm_head_warmup_v2",
        "lstm_blend50_v3",
        "lstm_blend25_v4",
    }
    if effective_profile not in lstm_profiles or backbone != "lstm":
        raise ValueError(
            f"Optimization profile {profile!r} is only valid for the LSTM backbone."
        )

    daily_stages = [
        stage for stage in config["progressive_stages"] if stage["resolution"] == "daily"
    ]
    if len(daily_stages) != 1:
        raise ValueError("Expected exactly one daily stage in the PTL preset.")
    daily_stages[0].update(
        {
            "epochs": 120,
            "base_lr": 2.5e-4,
            "freeze_backbone_epochs": 0,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
            "scheduler_patience": 6,
            "scheduler_min_lr": 1e-5,
        }
    )
    if effective_profile == "lstm_head_warmup_v2":
        for stage in config["progressive_stages"]:
            if stage["resolution"] != "daily":
                stage["freeze_backbone_epochs"] = int(stage["epochs"])
    elif effective_profile in {"lstm_blend50_v3", "lstm_blend25_v4"}:
        weekly_stages = [
            stage
            for stage in config["progressive_stages"]
            if stage["resolution"] == "weekly"
        ]
        if len(weekly_stages) != 1:
            raise ValueError("Expected exactly one weekly stage in the PTL preset.")
        weekly_stages[0]["load_weight_blend_alpha"] = (
            0.5 if effective_profile == "lstm_blend50_v3" else 0.25
        )


def build_custom_config(
    args: argparse.Namespace,
    backbone: str,
    station: str,
) -> dict:
    config = copy.deepcopy(build_finetune_preset(PRESET_NAME))
    if config is None:
        raise ValueError(f"Missing finetune preset: {PRESET_NAME}")
    apply_optimization_profile(config, args.optimization_profile, backbone)
    config.update(
        {
            "data_root": str(args.data_root.resolve()),
            "save_dir": str(finetune_run_root(args, backbone).resolve()),
            "feature_columns": list(FEATURE_COLUMNS),
            "target_station_names": [station],
            "time_start": args.time_start,
            "time_end": args.time_end,
            "stage_time_ranges": build_shared_stage_time_ranges(
                args.time_start,
                args.time_end,
            ),
            "invalid_window_policy": "all",
            "backbone_name": backbone,
            "model_agnostic_interface": profile_interface(args.optimization_profile),
            "save_model_weights": bool(args.save_model_weights),
            "optimization_profile": args.optimization_profile,
        }
    )
    for stage in config["progressive_stages"]:
        stage["invalid_window_policy"] = "all"
        stage["soft_gap_max_steps"] = None
    return config


def build_direct_config(
    args: argparse.Namespace,
    backbone: str,
    station: str,
) -> dict:
    preset = copy.deepcopy(build_finetune_preset(PRESET_NAME))
    if preset is None:
        raise ValueError(f"Missing finetune preset: {PRESET_NAME}")
    daily_stages = [
        stage for stage in preset["progressive_stages"] if stage["resolution"] == "daily"
    ]
    if len(daily_stages) != 1:
        raise ValueError("Expected exactly one daily stage in the PTL preset.")
    daily_stage = daily_stages[0]
    daily_stage.update(
        {
            "epochs": 120,
            "base_lr": 2.5e-4,
            "weight_decay": 1e-4,
            "freeze_backbone_epochs": 0,
            "invalid_window_policy": "all",
            "soft_gap_max_steps": None,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
            "scheduler_patience": 6,
            "scheduler_min_lr": 1e-5,
            "reset_model_seed": args.seed,
        }
    )
    return {
        **preset,
        "data_root": str(args.data_root.resolve()),
        "save_dir": str(direct_run_root(args, backbone).resolve()),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_station_names": [station],
        "time_start": args.time_start,
        "time_end": args.time_end,
        "stage_time_ranges": {"stage3_daily": (args.time_start, args.time_end)},
        "invalid_window_policy": "all",
        "backbone_name": backbone,
        "model_agnostic_interface": profile_interface(args.optimization_profile),
        "save_model_weights": bool(args.save_model_weights),
        "optimization_profile": f"direct_{args.optimization_profile}",
        "progressive_stages": [daily_stage],
    }


def run_finetune_worker(args: argparse.Namespace) -> None:
    if args.worker_pretrain_dir is None or args.worker_station is None:
        raise ValueError("Worker mode requires --worker-pretrain-dir and --worker-station.")
    backbone = selected_backbones(args)[0]
    custom_config = build_custom_config(args, backbone, args.worker_station)
    finetune_main(
        pretrain_model_dir=str(args.worker_pretrain_dir.resolve()),
        custom_config=custom_config,
        seed=args.seed,
    )


def run_direct_worker(args: argparse.Namespace) -> None:
    if args.worker_station is None:
        raise ValueError("Direct worker mode requires --worker-station.")
    backbone = selected_backbones(args)[0]
    finetune_main(
        pretrain_model_dir=None,
        custom_config=build_direct_config(args, backbone, args.worker_station),
        seed=args.seed,
    )


def run_command(command: list[str], log_path: Path, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        result = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed; see {log_path}")


def run_pretraining(args: argparse.Namespace, backbone: str) -> Path | None:
    existing = find_completed_pretrain(args, backbone)
    if existing is not None and not args.force:
        print(f"SKIP pretrain {backbone}: {existing}")
        return existing

    save_root = args.pretrain_root / backbone
    command = [
        sys.executable,
        str(SCRIPT_DIR / "Pretrain_cross_station_v2.py"),
        "--backbone",
        backbone,
        "--model-agnostic-interface",
        profile_interface(args.optimization_profile),
        "--seed",
        str(args.seed),
        "--save-root",
        str(save_root),
    ]
    if args.pretrain_epochs is not None:
        command.extend(["--epochs", str(args.pretrain_epochs)])
    run_command(
        command,
        args.pretrain_root / "logs" / f"pretrain_{backbone}_seed{args.seed}.log",
        args.dry_run,
    )
    if args.dry_run:
        return save_root / f"<generated-{backbone}-pretrain-run>"
    return find_completed_pretrain(args, backbone)


def run_direct_training(
    args: argparse.Namespace,
    backbone: str,
    stations: list[str],
) -> None:
    total = len(stations)
    for index, station in enumerate(stations, start=1):
        existing = find_completed_direct(args, backbone, station)
        if existing is not None and not args.force:
            print(f"[{index}/{total}] SKIP {backbone} Direct {station}: {existing}")
            continue
        print(f"[{index}/{total}] RUN {backbone} Direct {station}")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-direct",
            "--backbone",
            backbone,
            "--worker-station",
            station,
            "--seed",
            str(args.seed),
            "--data-root",
            str(args.data_root),
            "--direct-root",
            str(args.direct_root),
            "--time-start",
            args.time_start,
            "--time-end",
            args.time_end,
            "--optimization-profile",
            args.optimization_profile,
        ]
        if args.save_model_weights:
            command.append("--save-model-weights")
        run_command(
            command,
            args.direct_root
            / "logs"
            / args.optimization_profile
            / backbone
            / f"{station}_seed{args.seed}.log",
            args.dry_run,
        )


def run_finetuning(
    args: argparse.Namespace,
    backbone: str,
    pretrain_dir: Path,
    stations: list[str],
) -> None:
    total = len(stations)
    for index, station in enumerate(stations, start=1):
        existing = find_completed_finetune(args, backbone, station)
        if existing is not None and not args.force:
            print(f"[{index}/{total}] SKIP {backbone}+PTL {station}: {existing}")
            continue
        print(f"[{index}/{total}] RUN {backbone}+PTL {station}")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-finetune",
            "--backbone",
            backbone,
            "--worker-pretrain-dir",
            str(pretrain_dir),
            "--worker-station",
            station,
            "--seed",
            str(args.seed),
            "--data-root",
            str(args.data_root),
            "--finetune-root",
            str(args.finetune_root),
            "--time-start",
            args.time_start,
            "--time-end",
            args.time_end,
            "--optimization-profile",
            args.optimization_profile,
        ]
        if args.save_model_weights:
            command.append("--save-model-weights")
        run_command(
            command,
            args.finetune_root
            / "logs"
            / args.optimization_profile
            / backbone
            / f"{station}_seed{args.seed}.log",
            args.dry_run,
        )


def load_metrics(path: Path) -> dict:
    return pd.read_csv(path, index_col=0).replace({np.nan: None}).to_dict(orient="index")


def flatten_metrics(metrics: dict) -> dict:
    focus = [metrics[name] for name in FOCUS_FEATURES]
    row = {
        "overall_nse": float(metrics["__overall__"]["NSE"]),
        "overall_rmse": float(metrics["__overall__"]["RMSE"]),
        "overall_mae": float(metrics["__overall__"]["MAE"]),
        "focus_mean_nse": float(np.mean([entry["NSE"] for entry in focus])),
        "focus_mean_rmse": float(np.mean([entry["RMSE"] for entry in focus])),
        "focus_mean_mae": float(np.mean([entry["MAE"] for entry in focus])),
    }
    for feature in FEATURE_COLUMNS:
        for metric_name in ("NSE", "RMSE", "MAE"):
            row[f"{feature}_{metric_name.lower()}"] = float(metrics[feature][metric_name])
    return row


def build_summary(args: argparse.Namespace, backbones: list[str], stations: list[str]) -> None:
    station_class = pd.read_csv(args.station_class, encoding="utf-8-sig")
    station_metadata = station_class.set_index("station").to_dict(orient="index")
    manifest_rows = []
    if profile_interface(args.optimization_profile) != "legacy":
        direct_rows = []
        for backbone in backbones:
            for station in stations:
                run_dir = find_completed_direct(args, backbone, station)
                if run_dir is None:
                    manifest_rows.append(
                        {
                            "backbone": backbone,
                            "station": station,
                            "training_strategy": "Direct",
                            "status": "missing",
                        }
                    )
                    continue
                stage_dir = run_dir / "stage3_daily"
                metrics_path = stage_dir / "metrics.csv"
                meta = read_json(stage_dir / "meta.json")
                row = {
                    "station": station,
                    **station_metadata.get(station, {}),
                    "model": BACKBONE_LABELS[backbone],
                    "backbone": backbone,
                    "training_strategy": "Direct",
                    "seed": args.seed,
                    "optimization_profile": f"direct_{args.optimization_profile}",
                    "model_agnostic_interface": profile_interface(
                        args.optimization_profile
                    ),
                    "uses_pretraining": False,
                    "uses_progressive_transfer": False,
                    "train_windows": meta.get("train_windows"),
                    "val_windows": meta.get("val_windows"),
                    "test_windows": meta.get("test_windows"),
                    "soft_gap_mode": "off",
                    "invalid_window_policy": meta.get("invalid_window_policy"),
                    "run_dir": str(run_dir),
                    "metrics_path": str(metrics_path),
                    "pretrain_dir": None,
                }
                row.update(flatten_metrics(load_metrics(metrics_path)))
                direct_rows.append(row)
                manifest_rows.append(
                    {
                        "backbone": backbone,
                        "station": station,
                        "training_strategy": "Direct",
                        "status": "completed",
                        "finetune_run_dir": str(run_dir),
                        "metrics_path": str(metrics_path),
                    }
                )
        direct = pd.DataFrame(direct_rows)
    else:
        direct = pd.read_csv(args.direct_baseline_long, encoding="utf-8-sig")
        direct = direct[
            direct["station"].isin(stations)
            & direct["model"].isin([BACKBONE_LABELS[backbone] for backbone in backbones])
        ].copy()
        label_to_backbone = {label: name for name, label in BACKBONE_LABELS.items()}
        direct["backbone"] = direct["model"].map(label_to_backbone)
        direct["training_strategy"] = "Direct"
        direct["seed"] = args.seed

    ptl_rows = []
    for backbone in backbones:
        pretrain_dir = find_completed_pretrain(args, backbone)
        for station in stations:
            run_dir = find_completed_finetune(args, backbone, station)
            if run_dir is None:
                manifest_rows.append(
                    {
                        "backbone": backbone,
                        "station": station,
                        "training_strategy": "PTL",
                        "status": "missing",
                        "pretrain_dir": str(pretrain_dir) if pretrain_dir else None,
                    }
                )
                continue
            stage3_dir = run_dir / "stage3_daily"
            metrics_path = stage3_dir / "metrics.csv"
            meta_path = stage3_dir / "meta.json"
            meta = read_json(meta_path)
            row = {
                "station": station,
                **station_metadata.get(station, {}),
                "model": (
                    f"{BACKBONE_LABELS[backbone]}+PTL"
                    if args.optimization_profile == "default"
                    else (
                        f"{BACKBONE_LABELS[backbone]}+PTL"
                        f"[{args.optimization_profile}]"
                    )
                ),
                "backbone": backbone,
                "training_strategy": (
                    "Cross-station pretrain + progressive transfer"
                    if args.optimization_profile == "default"
                    else (
                        "Cross-station pretrain + progressive transfer "
                        f"({args.optimization_profile})"
                    )
                ),
                "seed": args.seed,
                "optimization_profile": args.optimization_profile,
                "model_agnostic_interface": profile_interface(
                    args.optimization_profile
                ),
                "uses_pretraining": True,
                "uses_progressive_transfer": True,
                "train_windows": meta.get("train_windows"),
                "val_windows": meta.get("val_windows"),
                "test_windows": meta.get("test_windows"),
                "soft_gap_mode": "off",
                "invalid_window_policy": meta.get("invalid_window_policy"),
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "pretrain_dir": str(pretrain_dir) if pretrain_dir else None,
            }
            row.update(flatten_metrics(load_metrics(metrics_path)))
            ptl_rows.append(row)
            manifest_rows.append(
                {
                    "backbone": backbone,
                    "station": station,
                    "training_strategy": "PTL",
                    "status": "completed",
                    "pretrain_dir": str(pretrain_dir) if pretrain_dir else None,
                    "finetune_run_dir": str(run_dir),
                    "metrics_path": str(metrics_path),
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    ptl = pd.DataFrame(ptl_rows)
    direct_columns = [column for column in ptl.columns if column in direct.columns]
    comparison = pd.concat([direct[direct_columns], ptl], ignore_index=True, sort=False)
    comparison.to_csv(
        args.output_root / "模型无关实验_逐站点结果.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(manifest_rows).to_csv(
        args.output_root / "模型无关实验_运行清单.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_rows = []
    for backbone in backbones:
        direct_rows = direct[direct["backbone"] == backbone].set_index("station")
        ptl_backbone = ptl[ptl["backbone"] == backbone].set_index("station") if not ptl.empty else pd.DataFrame()
        if ptl_backbone.empty:
            continue
        shared_stations = [station for station in stations if station in direct_rows.index and station in ptl_backbone.index]
        for station in shared_stations:
            row = {"station": station, "backbone": backbone, "seed": args.seed}
            for metric_name in (
                "overall_nse",
                "focus_mean_nse",
                "overall_rmse",
                "overall_mae",
                "focus_mean_rmse",
                "focus_mean_mae",
            ):
                direct_value = float(direct_rows.loc[station, metric_name])
                ptl_value = float(ptl_backbone.loc[station, metric_name])
                row[f"direct_{metric_name}"] = direct_value
                row[f"ptl_{metric_name}"] = ptl_value
                row[f"delta_{metric_name}"] = ptl_value - direct_value
            delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(
        args.output_root / "模型无关实验_逐站点增量.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not comparison.empty:
        numeric_metrics = [
            "overall_nse",
            "focus_mean_nse",
            "overall_rmse",
            "overall_mae",
            "focus_mean_rmse",
            "focus_mean_mae",
        ]
        means = comparison.groupby(["model", "backbone", "training_strategy"], as_index=False)[
            numeric_metrics
        ].mean()
        counts = comparison.groupby(["model", "backbone", "training_strategy"]).size().rename("station_count").reset_index()
        means = means.merge(counts, on=["model", "backbone", "training_strategy"], how="left")
        means.to_csv(
            args.output_root / "模型无关实验_模型均值.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"Summary written to {args.output_root}")


def main() -> None:
    args = parse_args()
    if args.worker_finetune:
        run_finetune_worker(args)
        return
    if args.worker_direct:
        run_direct_worker(args)
        return

    backbones = selected_backbones(args)
    stations = load_stations(args)
    if args.seed != 42:
        print(f"Warning: current paper experiments use seed 42; requested seed is {args.seed}.")

    pretrain_dirs = {}
    if args.step in {"all", "pretrain"}:
        for backbone in backbones:
            pretrain_dirs[backbone] = run_pretraining(args, backbone)

    if args.step in {"all", "direct"}:
        if profile_interface(args.optimization_profile) != "legacy":
            for backbone in backbones:
                run_direct_training(args, backbone, stations)
        elif args.step == "direct":
            raise ValueError(
                "The integrated Direct runner is only used by unified feature-token profiles."
            )

    if args.step in {"all", "finetune"}:
        for backbone in backbones:
            pretrain_dir = pretrain_dirs.get(backbone) or find_completed_pretrain(args, backbone)
            if pretrain_dir is None:
                raise FileNotFoundError(
                    f"No completed {backbone} pretrain run. Run --step pretrain first."
                )
            run_finetuning(args, backbone, pretrain_dir, stations)

    if args.step in {"all", "summary"} and not args.dry_run:
        build_summary(args, backbones, stations)

    setup = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "step": args.step,
        "seed": args.seed,
        "backbones": backbones,
        "stations": stations,
        "preset": PRESET_NAME,
        "optimization_profile": args.optimization_profile,
        "model_agnostic_interface": profile_interface(args.optimization_profile),
        "data_root": str(args.data_root.resolve()),
        "time_start": args.time_start,
        "time_end": args.time_end,
        "invalid_window_policy": "all",
        "soft_gap_max_steps": None,
    }
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "模型无关实验_设置.json").write_text(
            json.dumps(setup, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
