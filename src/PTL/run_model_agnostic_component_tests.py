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
    build_finetune_preset,
    build_shared_stage_time_ranges,
    main as finetune_main,
)
from model_agnostic_backbones import normalize_backbone_name


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
PRESET_NAME = "core3_progressive_v2pretrain_v2"
DEFAULT_STATIONS = ["大墩", "武林渡口"]
STRATEGY_LABELS = {
    "pretrain_only": "Cross-station pretrain + daily finetune",
    "progressive_only": "Random init + weekly-to-4d-to-daily transfer",
}

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
    REPO_ROOT
    / "results"
    / "ptl"
    / "finetune"
    / "runs"
    / "model_agnostic_component_tests"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_CURRENT_SUMMARY / "模型无关实验" / "分解实验_高基线站点"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run seed-42 CNN/LSTM component tests on strong Direct-baseline stations: "
            "cross-station pretraining only or progressive transfer only."
        )
    )
    parser.add_argument(
        "--step",
        choices=("all", "finetune", "summary"),
        default="all",
    )
    parser.add_argument("--backbone", action="append", choices=("cnn", "lstm"), default=[])
    parser.add_argument(
        "--strategy",
        action="append",
        choices=tuple(STRATEGY_LABELS),
        default=[],
    )
    parser.add_argument("--station", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--station-class", type=Path, default=DEFAULT_STATION_CLASS)
    parser.add_argument("--direct-baseline-long", type=Path, default=DEFAULT_DIRECT_BASELINE_LONG)
    parser.add_argument("--pretrain-root", type=Path, default=DEFAULT_PRETRAIN_ROOT)
    parser.add_argument("--finetune-root", type=Path, default=DEFAULT_FINETUNE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--time-start", default="2023-01-01 00:00:00")
    parser.add_argument("--time-end", default="2024-12-31 23:59:59")
    parser.add_argument("--save-model-weights", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--worker-finetune", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-pretrain-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-station", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def selected_backbones(args: argparse.Namespace) -> list[str]:
    return list(dict.fromkeys(args.backbone or ["cnn", "lstm"]))


def selected_strategies(args: argparse.Namespace) -> list[str]:
    return list(dict.fromkeys(args.strategy or list(STRATEGY_LABELS)))


def load_stations(args: argparse.Namespace) -> list[str]:
    station_frame = pd.read_csv(args.station_class, encoding="utf-8-sig")
    available = set(station_frame["station"].astype(str))
    stations = list(dict.fromkeys(args.station or DEFAULT_STATIONS))
    missing = [station for station in stations if station not in available]
    if missing:
        raise ValueError(f"Station(s) missing from classification table: {missing}")
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
        if not (config_path.parent / "model.pth").exists():
            continue
        config = read_json(config_path)
        if normalize_backbone_name(config.get("backbone_name")) == backbone:
            return config_path.parent
    return None


def strategy_run_root(args: argparse.Namespace, strategy: str, backbone: str) -> Path:
    return args.finetune_root / strategy / backbone


def find_completed_finetune(
    args: argparse.Namespace,
    strategy: str,
    backbone: str,
    station: str,
) -> Path | None:
    root = strategy_run_root(args, strategy, backbone)
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
        expected_pretraining = strategy == "pretrain_only"
        if (
            normalize_backbone_name(meta.get("backbone_name")) == backbone
            and bool(meta.get("uses_pretraining")) == expected_pretraining
        ):
            return summary_path.parent
    return None


def build_custom_config(
    args: argparse.Namespace,
    strategy: str,
    backbone: str,
    station: str,
) -> dict:
    config = copy.deepcopy(build_finetune_preset(PRESET_NAME))
    if config is None:
        raise ValueError(f"Missing finetune preset: {PRESET_NAME}")
    stages = copy.deepcopy(config["progressive_stages"])
    if strategy == "pretrain_only":
        daily_stages = [stage for stage in stages if stage["resolution"] == "daily"]
        if len(daily_stages) != 1:
            raise ValueError("Expected exactly one daily stage in the PTL preset.")
        stages = daily_stages
    elif strategy != "progressive_only":
        raise ValueError(f"Unsupported strategy: {strategy}")

    config.update(
        {
            "data_root": str(args.data_root.resolve()),
            "save_dir": str(strategy_run_root(args, strategy, backbone).resolve()),
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
            "save_model_weights": bool(args.save_model_weights),
            "progressive_stages": stages,
        }
    )
    for stage in config["progressive_stages"]:
        stage["invalid_window_policy"] = "all"
        stage["soft_gap_max_steps"] = None
    return config


def run_finetune_worker(args: argparse.Namespace) -> None:
    strategies = selected_strategies(args)
    backbones = selected_backbones(args)
    if len(strategies) != 1 or len(backbones) != 1 or args.worker_station is None:
        raise ValueError("Worker mode requires one strategy, one backbone, and one station.")
    strategy = strategies[0]
    backbone = backbones[0]
    if strategy == "pretrain_only" and args.worker_pretrain_dir is None:
        raise ValueError("pretrain_only worker mode requires --worker-pretrain-dir.")
    custom_config = build_custom_config(
        args,
        strategy,
        backbone,
        args.worker_station,
    )
    finetune_main(
        pretrain_model_dir=(
            str(args.worker_pretrain_dir.resolve())
            if strategy == "pretrain_only"
            else None
        ),
        custom_config=custom_config,
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


def run_finetuning(
    args: argparse.Namespace,
    strategies: list[str],
    backbones: list[str],
    stations: list[str],
) -> None:
    total = len(strategies) * len(backbones) * len(stations)
    index = 0
    for strategy in strategies:
        for backbone in backbones:
            pretrain_dir = find_completed_pretrain(args, backbone)
            if strategy == "pretrain_only" and pretrain_dir is None:
                raise FileNotFoundError(
                    f"No completed {backbone} pretrain run under {args.pretrain_root}."
                )
            for station in stations:
                index += 1
                existing = find_completed_finetune(args, strategy, backbone, station)
                if existing is not None and not args.force:
                    print(f"[{index}/{total}] SKIP {strategy} {backbone} {station}: {existing}")
                    continue
                print(f"[{index}/{total}] RUN {strategy} {backbone} {station}")
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-finetune",
                    "--strategy",
                    strategy,
                    "--backbone",
                    backbone,
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
                ]
                if strategy == "pretrain_only" and pretrain_dir is not None:
                    command.extend(["--worker-pretrain-dir", str(pretrain_dir)])
                if args.save_model_weights:
                    command.append("--save-model-weights")
                run_command(
                    command,
                    (
                        args.finetune_root
                        / "logs"
                        / strategy
                        / backbone
                        / f"{station}_seed{args.seed}.log"
                    ),
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


def build_selection_table(
    direct_all: pd.DataFrame,
    stations: list[str],
    backbones: list[str],
) -> pd.DataFrame:
    ranked = direct_all[direct_all["model"].isin([name.upper() for name in backbones])].copy()
    ranked["overall_nse_rank"] = ranked.groupby("model")["overall_nse"].rank(
        ascending=False,
        method="min",
    )
    ranked["focus_mean_nse_rank"] = ranked.groupby("model")["focus_mean_nse"].rank(
        ascending=False,
        method="min",
    )
    ranked = ranked[ranked["station"].isin(stations)].copy()
    ranked["selection_reason"] = ranked.apply(
        lambda row: (
            "best CNN Direct Overall NSE"
            if row["station"] == "大墩" and row["model"] == "CNN"
            else (
                "best LSTM Direct Overall NSE"
                if row["station"] == "武林渡口" and row["model"] == "LSTM"
                else "cross-backbone evaluation on the selected strong station"
            )
        ),
        axis=1,
    )
    return ranked[
        [
            "station",
            "model",
            "overall_nse",
            "overall_nse_rank",
            "focus_mean_nse",
            "focus_mean_nse_rank",
            "overall_rmse",
            "overall_mae",
            "selection_reason",
        ]
    ].sort_values(["station", "model"])


def build_summary(
    args: argparse.Namespace,
    strategies: list[str],
    backbones: list[str],
    stations: list[str],
) -> None:
    direct_all = pd.read_csv(args.direct_baseline_long, encoding="utf-8-sig")
    direct = direct_all[
        direct_all["station"].isin(stations)
        & direct_all["model"].isin([backbone.upper() for backbone in backbones])
    ].copy()
    direct["backbone"] = direct["model"].str.lower()
    direct["variant"] = "direct"
    direct["training_strategy"] = "Direct daily training"
    direct["seed"] = args.seed
    direct["uses_pretraining"] = False
    direct["uses_progressive_transfer"] = False

    result_rows = []
    manifest_rows = []
    validation_rows = []
    for strategy in strategies:
        for backbone in backbones:
            pretrain_dir = find_completed_pretrain(args, backbone)
            for station in stations:
                run_dir = find_completed_finetune(args, strategy, backbone, station)
                if run_dir is None:
                    manifest_rows.append(
                        {
                            "strategy": strategy,
                            "backbone": backbone,
                            "station": station,
                            "status": "missing",
                            "pretrain_dir": str(pretrain_dir) if pretrain_dir else None,
                        }
                    )
                    continue
                metrics_path = run_dir / "stage3_daily" / "metrics.csv"
                meta_path = run_dir / "stage3_daily" / "meta.json"
                meta = read_json(meta_path)
                run_summary = read_json(run_dir / "summary.json")
                stage_validation = []
                for stage_result in run_summary.get("stages", []):
                    stage_meta = read_json(Path(stage_result["save_dir"]) / "meta.json")
                    stage_validation.append(
                        {
                            "stage_name": stage_result["stage_name"],
                            "loaded_pretrain_keys": stage_meta.get("loaded_pretrain_keys"),
                            "loaded_stage_keys": stage_meta.get("loaded_stage_keys"),
                        }
                    )
                row = {
                    "station": station,
                    "model": f"{backbone.upper()} {strategy}",
                    "backbone": backbone,
                    "variant": strategy,
                    "training_strategy": STRATEGY_LABELS[strategy],
                    "seed": args.seed,
                    "uses_pretraining": strategy == "pretrain_only",
                    "uses_progressive_transfer": strategy == "progressive_only",
                    "train_windows": meta.get("train_windows"),
                    "val_windows": meta.get("val_windows"),
                    "test_windows": meta.get("test_windows"),
                    "invalid_window_policy": meta.get("invalid_window_policy"),
                    "soft_gap_mode": "off",
                    "run_dir": str(run_dir),
                    "metrics_path": str(metrics_path),
                    "pretrain_dir": (
                        str(pretrain_dir) if strategy == "pretrain_only" and pretrain_dir else None
                    ),
                }
                row.update(flatten_metrics(load_metrics(metrics_path)))
                result_rows.append(row)
                manifest_rows.append(
                    {
                        "strategy": strategy,
                        "backbone": backbone,
                        "station": station,
                        "status": "completed",
                        "pretrain_dir": row["pretrain_dir"],
                        "finetune_run_dir": str(run_dir),
                        "metrics_path": str(metrics_path),
                    }
                )
                validation_rows.append(
                    {
                        "strategy": strategy,
                        "backbone": backbone,
                        "station": station,
                        "stage_count": len(stage_validation),
                        "stage_names": " -> ".join(
                            stage["stage_name"] for stage in stage_validation
                        ),
                        "uses_pretraining": bool(meta.get("uses_pretraining")),
                        "initialization_source": meta.get("initialization_source"),
                        "first_stage_loaded_pretrain_keys": (
                            stage_validation[0]["loaded_pretrain_keys"]
                            if stage_validation
                            else None
                        ),
                        "later_stage_loaded_keys": " -> ".join(
                            str(stage["loaded_stage_keys"])
                            for stage in stage_validation[1:]
                        ),
                    }
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    common_columns = [column for column in results.columns if column in direct.columns]
    comparison = pd.concat(
        [direct[common_columns], results],
        ignore_index=True,
        sort=False,
    )
    comparison.to_csv(
        args.output_root / "高基线站点_分解实验_逐站点结果.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(manifest_rows).to_csv(
        args.output_root / "高基线站点_分解实验_运行清单.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(validation_rows).to_csv(
        args.output_root / "高基线站点_分解实验_路径验证.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_selection_table(direct_all, stations, backbones).to_csv(
        args.output_root / "高基线站点_选择依据.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_rows = []
    for result in result_rows:
        baseline = direct[
            (direct["station"] == result["station"])
            & (direct["backbone"] == result["backbone"])
        ].iloc[0]
        delta = {
            "station": result["station"],
            "backbone": result["backbone"],
            "variant": result["variant"],
            "training_strategy": result["training_strategy"],
            "seed": args.seed,
        }
        for metric_name in (
            "overall_nse",
            "focus_mean_nse",
            "overall_rmse",
            "overall_mae",
            "focus_mean_rmse",
            "focus_mean_mae",
        ):
            direct_value = float(baseline[metric_name])
            result_value = float(result[metric_name])
            delta[f"direct_{metric_name}"] = direct_value
            delta[f"result_{metric_name}"] = result_value
            delta[f"delta_{metric_name}"] = result_value - direct_value
        delta_rows.append(delta)
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(
        args.output_root / "高基线站点_分解实验_相对Direct增量.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = [
        "overall_nse",
        "focus_mean_nse",
        "overall_rmse",
        "overall_mae",
        "focus_mean_rmse",
        "focus_mean_mae",
    ]
    means = comparison.groupby(
        ["backbone", "variant", "training_strategy"],
        as_index=False,
    )[metrics].mean()
    counts = comparison.groupby(
        ["backbone", "variant", "training_strategy"]
    ).size().rename("station_count").reset_index()
    means.merge(
        counts,
        on=["backbone", "variant", "training_strategy"],
        how="left",
    ).to_csv(
        args.output_root / "高基线站点_分解实验_模型均值.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Summary written to {args.output_root}")


def main() -> None:
    args = parse_args()
    if args.worker_finetune:
        run_finetune_worker(args)
        return

    backbones = selected_backbones(args)
    strategies = selected_strategies(args)
    stations = load_stations(args)
    if args.seed != 42:
        print(f"Warning: current paper experiments use seed 42; requested seed is {args.seed}.")

    if args.step in {"all", "finetune"}:
        run_finetuning(args, strategies, backbones, stations)
    if args.step in {"all", "summary"} and not args.dry_run:
        build_summary(args, strategies, backbones, stations)

    setup = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "step": args.step,
        "seed": args.seed,
        "backbones": backbones,
        "strategies": strategies,
        "strategy_definitions": STRATEGY_LABELS,
        "stations": stations,
        "station_selection": {
            "大墩": "highest CNN Direct Overall NSE among the current 17 stations",
            "武林渡口": "highest LSTM Direct Overall NSE among the current 17 stations",
        },
        "preset": PRESET_NAME,
        "data_root": str(args.data_root.resolve()),
        "time_start": args.time_start,
        "time_end": args.time_end,
        "invalid_window_policy": "all",
        "soft_gap_max_steps": None,
    }
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "高基线站点_分解实验_设置.json").write_text(
            json.dumps(setup, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
