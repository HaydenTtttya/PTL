import argparse
import datetime
import json
from pathlib import Path

import pandas as pd

from finetune import (
    FINETUNE_RUNS_DIR,
    FinetuneConfig,
    NH4N_FEATURE_COLUMNS,
    build_shared_stage_time_ranges,
    build_stage_specs,
    build_weekly_to_daily_stages,
    find_latest_pretrain_run,
    load_stage_data,
    main as finetune_main,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "water_quality_processed_2021_2024"
DEFAULT_STATION_META = DEFAULT_DATA_ROOT / "station_meta.csv"
DEFAULT_OTHER_STATIONS = ("老口", "上中", "白马")
DEFAULT_TIME_START = "2023-01-01 00:00:00"
DEFAULT_TIME_END = "2024-12-31 23:59:59"
DEFAULT_STAGE3_SOFT_GAP_MAX_STEPS = 14
V3_PRETRAIN_ROOT_CANDIDATES = (
    REPO_ROOT / "results" / "cross_station" / "pretrain" / "v3_full_runs",
    REPO_ROOT / "results" / "cross_station" / "pretrain" / "v3_runs",
    REPO_ROOT / "results" / "cross_station" / "pretrain" / "v3_arch_runs",
    REPO_ROOT / "results" / "cross_station" / "pretrain" / "v3_smoke_runs",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PTL NH4N finetune tests on Pearl River stations from the 2021-2024 dataset with V3 pretrain weights.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Processed dataset root that contains 4h/daily/weekly folders and station_meta.csv.",
    )
    parser.add_argument(
        "--station-meta",
        type=Path,
        default=DEFAULT_STATION_META,
        help="station_meta.csv path used for station selection and missingness reporting.",
    )
    parser.add_argument(
        "--pretrain-dir",
        type=Path,
        default=None,
        help="Explicit V3 pretrain run directory. If omitted, the latest valid run is picked from known V3 folders.",
    )
    parser.add_argument(
        "--selection-profile",
        choices=("gx_other3", "pearl_auto"),
        default="gx_other3",
        help="Default station selection strategy when --station is not provided.",
    )
    parser.add_argument(
        "--station",
        action="append",
        default=[],
        help="Repeat to run explicit station names. This overrides --selection-profile.",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=10,
        help="Maximum station count for pearl_auto selection. Ignored for gx_other3 unless fewer stations are available.",
    )
    parser.add_argument(
        "--include-yangshuo",
        action="store_true",
        help="Include 阳朔 in pearl_auto selection. Explicit --station always wins.",
    )
    parser.add_argument(
        "--min-daily-records",
        type=int,
        default=900,
        help="Minimum daily record count in station_meta for pearl_auto station selection.",
    )
    parser.add_argument(
        "--time-start",
        default=DEFAULT_TIME_START,
        help="Inclusive finetune time range start.",
    )
    parser.add_argument(
        "--time-end",
        default=DEFAULT_TIME_END,
        help="Inclusive finetune time range end.",
    )
    parser.add_argument(
        "--stage3-soft-gap-max-steps",
        type=int,
        default=DEFAULT_STAGE3_SOFT_GAP_MAX_STEPS,
        help="Soft-gap tolerance for the daily stage input side. Targets remain strictly filtered.",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=None,
        help="Optional override for weekly stage epochs.",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=None,
        help="Optional override for 4d stage epochs.",
    )
    parser.add_argument(
        "--stage3-epochs",
        type=int,
        default=None,
        help="Optional override for daily stage epochs.",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(FINETUNE_RUNS_DIR),
        help="Parent directory for this batch run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to finetune.main.",
    )
    parser.add_argument(
        "--keep-unrunnable",
        action="store_true",
        help="Keep stations even if a stage has no train/val/test windows. By default these stations are dropped before training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only select stations and validate stage windows without launching training.",
    )
    return parser.parse_args()


def resolve_pretrain_dir(explicit_path):
    if explicit_path is not None:
        config_path = explicit_path / "config.json"
        model_path = explicit_path / "model.pth"
        if not config_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Invalid pretrain dir: {explicit_path}. Expected config.json and model.pth."
            )
        return explicit_path.resolve()

    for root in V3_PRETRAIN_ROOT_CANDIDATES:
        run_path = find_latest_pretrain_run(str(root))
        if run_path is not None:
            return Path(run_path).resolve()

    searched = ", ".join(str(path) for path in V3_PRETRAIN_ROOT_CANDIDATES)
    raise FileNotFoundError(f"Could not find a valid V3 pretrain run under: {searched}")


def load_station_meta(path, data_root):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    numeric_columns = [
        "剩余缺失值数_4h",
        "剩余缺失值数_daily",
        "剩余缺失值数_weekly",
        "记录数_4h",
        "记录数_daily",
        "记录数_weekly",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["daily_missing_ratio"] = (
        frame["剩余缺失值数_daily"] / (frame["剩余缺失值数_daily"] + frame["记录数_daily"])
    )
    frame["weekly_missing_ratio"] = (
        frame["剩余缺失值数_weekly"] / (frame["剩余缺失值数_weekly"] + frame["记录数_weekly"])
    )
    frame["has_daily_file"] = frame["断面名称"].apply(
        lambda name: (data_root / "daily" / f"{name}.csv").exists()
    )
    frame["has_weekly_file"] = frame["断面名称"].apply(
        lambda name: (data_root / "weekly" / f"{name}.csv").exists()
    )
    return frame


def choose_station_rows(meta_frame, args):
    available = meta_frame[meta_frame["has_daily_file"] & meta_frame["has_weekly_file"]].copy()
    pearl = available[available["流域"] == "珠江流域"].copy()

    if args.station:
        explicit_names = list(dict.fromkeys(args.station))
        selected_rows = []
        available_indexed = available.set_index("断面名称", drop=False)
        missing = [name for name in explicit_names if name not in available_indexed.index]
        if missing:
            raise ValueError(f"Unknown or unavailable stations: {missing}")
        for station_name in explicit_names:
            selected_rows.append(available_indexed.loc[station_name])
        return pd.DataFrame(selected_rows).reset_index(drop=True)

    if args.selection_profile == "gx_other3":
        default_names = [name for name in DEFAULT_OTHER_STATIONS if name in set(available["断面名称"])]
        if not default_names:
            raise ValueError("Default comparison stations are unavailable in station_meta.csv.")
        available_indexed = available.set_index("断面名称", drop=False)
        return pd.DataFrame([available_indexed.loc[name] for name in default_names]).reset_index(drop=True)

    auto = pearl.copy()
    if not args.include_yangshuo:
        auto = auto[auto["断面名称"] != "阳朔"]
    if args.min_daily_records is not None:
        auto = auto[auto["记录数_daily"] >= int(args.min_daily_records)]
    auto = auto.sort_values(
        ["daily_missing_ratio", "weekly_missing_ratio", "剩余缺失值数_daily", "断面名称"]
    ).reset_index(drop=True)
    if args.max_stations is not None:
        auto = auto.head(int(args.max_stations))
    if auto.empty:
        raise ValueError("No Pearl River stations matched the current selection filters.")
    return auto


def build_custom_config(args, station_names, save_dir):
    stage1_overrides = {}
    stage2_overrides = {}
    stage3_overrides = {
        "invalid_window_policy": "all",
        "soft_gap_max_steps": args.stage3_soft_gap_max_steps,
    }
    if args.stage1_epochs is not None:
        stage1_overrides["epochs"] = int(args.stage1_epochs)
    if args.stage2_epochs is not None:
        stage2_overrides["epochs"] = int(args.stage2_epochs)
    if args.stage3_epochs is not None:
        stage3_overrides["epochs"] = int(args.stage3_epochs)

    return {
        "data_root": str(args.data_root.resolve()),
        "save_dir": str(save_dir.resolve()),
        "feature_columns": list(NH4N_FEATURE_COLUMNS),
        "target_station_names": list(station_names),
        "time_start": args.time_start,
        "time_end": args.time_end,
        "stage_time_ranges": build_shared_stage_time_ranges(args.time_start, args.time_end),
        "invalid_window_policy": "target_only",
        "progressive_stages": build_weekly_to_daily_stages(
            stage1_overrides=stage1_overrides or None,
            stage2_overrides=stage2_overrides or None,
            stage3_overrides=stage3_overrides,
        ),
    }


def make_runtime_config(custom_config):
    config = FinetuneConfig()
    for key, value in custom_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.feature_columns = list(config.feature_columns)
    config.input_dim = len(config.feature_columns)
    return config


def build_stage_preview(runtime_config, station_rows):
    stage_specs = build_stage_specs(runtime_config)
    preview_rows = []

    for station_row in station_rows.to_dict(orient="records"):
        row = {
            "station_name": station_row["断面名称"],
            "province": station_row["省份"],
            "basin": station_row["流域"],
            "daily_records_meta": int(station_row["记录数_daily"]),
            "daily_missing_meta": int(station_row["剩余缺失值数_daily"]),
            "daily_missing_ratio": float(station_row["daily_missing_ratio"]),
            "weekly_records_meta": int(station_row["记录数_weekly"]),
            "weekly_missing_meta": int(station_row["剩余缺失值数_weekly"]),
            "weekly_missing_ratio": float(station_row["weekly_missing_ratio"]),
        }
        runnable = True

        for stage in stage_specs:
            stage_data = load_stage_data(runtime_config, station_row["断面名称"], stage)
            prefix = stage["name"]
            if stage_data is None:
                runnable = False
                row[f"{prefix}_status"] = "missing_or_insufficient"
                row[f"{prefix}_train_windows"] = 0
                row[f"{prefix}_val_windows"] = 0
                row[f"{prefix}_test_windows"] = 0
                row[f"{prefix}_invalid_records"] = None
                continue

            row[f"{prefix}_status"] = "ok"
            row[f"{prefix}_train_windows"] = int(stage_data["train_windows"])
            row[f"{prefix}_val_windows"] = int(stage_data["val_windows"])
            row[f"{prefix}_test_windows"] = int(stage_data["test_windows"])
            row[f"{prefix}_invalid_records"] = int(stage_data["invalid_records"])
            row[f"{prefix}_input_invalid_records"] = int(stage_data["input_invalid_records"])
            row[f"{prefix}_target_invalid_records"] = int(stage_data["target_invalid_records"])

            if (
                stage_data["train_windows"] <= 0
                or stage_data["val_windows"] <= 0
                or stage_data["test_windows"] <= 0
            ):
                runnable = False

        row["runnable"] = runnable
        preview_rows.append(row)

    return pd.DataFrame(preview_rows)


def print_station_preview(preview_frame):
    display_columns = [
        "station_name",
        "daily_missing_ratio",
        "weekly_missing_ratio",
        "stage1_weekly_train_windows",
        "stage1_weekly_val_windows",
        "stage1_weekly_test_windows",
        "stage2_4d_train_windows",
        "stage2_4d_val_windows",
        "stage2_4d_test_windows",
        "stage3_daily_train_windows",
        "stage3_daily_val_windows",
        "stage3_daily_test_windows",
        "runnable",
    ]
    available_columns = [column for column in display_columns if column in preview_frame.columns]
    print(preview_frame[available_columns].to_string(index=False))


def write_metadata(run_dir, args, pretrain_dir, custom_config, selected_rows, preview_frame):
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_rows.to_csv(run_dir / "selected_stations.csv", index=False, encoding="utf-8-sig")
    preview_frame.to_csv(run_dir / "station_stage_preview.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": "dry_run" if args.dry_run else "train",
        "data_root": str(args.data_root.resolve()),
        "station_meta": str(args.station_meta.resolve()),
        "pretrain_dir": str(pretrain_dir),
        "selection_profile": args.selection_profile,
        "requested_stations": list(args.station),
        "selected_station_names": selected_rows["断面名称"].tolist(),
        "runnable_station_names": preview_frame.loc[preview_frame["runnable"], "station_name"].tolist(),
        "dropped_station_names": preview_frame.loc[~preview_frame["runnable"], "station_name"].tolist(),
        "custom_config": custom_config,
    }
    with (run_dir / "run_setup.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    pretrain_dir = resolve_pretrain_dir(args.pretrain_dir)
    meta_frame = load_station_meta(args.station_meta, args.data_root)
    selected_rows = choose_station_rows(meta_frame, args)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.save_root / f"batch_pearl_other_v3_2021_2024_{timestamp}"
    station_names = selected_rows["断面名称"].tolist()
    custom_config = build_custom_config(args, station_names, run_dir)
    runtime_config = make_runtime_config(custom_config)
    preview_frame = build_stage_preview(runtime_config, selected_rows)

    if not args.keep_unrunnable:
        runnable_names = preview_frame.loc[preview_frame["runnable"], "station_name"].tolist()
        selected_rows = selected_rows[selected_rows["断面名称"].isin(runnable_names)].reset_index(drop=True)
        station_names = selected_rows["断面名称"].tolist()
        custom_config = build_custom_config(args, station_names, run_dir)

    write_metadata(run_dir, args, pretrain_dir, custom_config, selected_rows, preview_frame)

    print("=" * 80)
    print("Pearl River 2021-2024 PTL V3 test runner")
    print(f"pretrain_dir: {pretrain_dir}")
    print(f"data_root: {args.data_root.resolve()}")
    print(f"output_dir: {run_dir.resolve()}")
    print(f"selected_stations: {station_names}")
    print(f"stage3_soft_gap_max_steps: {args.stage3_soft_gap_max_steps}")
    print("=" * 80)
    print("Stage preview:")
    print_station_preview(preview_frame)

    if not station_names:
        raise RuntimeError("No runnable stations remain after preview filtering.")

    if args.dry_run:
        print("\nDry-run complete. Training was not launched.")
        return

    finetune_main(
        pretrain_model_dir=str(pretrain_dir),
        custom_config=custom_config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
