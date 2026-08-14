from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_SCRIPT = REPO_ROOT / "src" / "Base" / "analysis" / "run_training_availability_17stations_all_models.py"
TAIL_DATA_ROOT = REPO_ROOT / "data" / "water_quality_processed_2021_2024_train_tail_availability_17stations"
TAIL_PTL_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "ptl"
    / "finetune"
    / "runs"
    / "training_tail_availability_17stations_ptl_2023_2024"
)
TAIL_BASELINE_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "base"
    / "fair_compare"
    / "training_tail_availability_17stations_all_models_2023_2024"
)
TAIL_SUMMARY_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "summary"
    / "current_all_tested_stations_overall_nse"
    / "training_tail_availability_17stations_ptl"
)

DEFAULT_LEVELS = [40, 60, 80, 100]
MASK_LEVELS = [40, 60, 80]


def load_base_module():
    spec = importlib.util.spec_from_file_location("training_availability_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Unable to load base script: {BASE_SCRIPT}")
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def configure_base(levels: list[int]) -> None:
    base.AVAILABILITY_DATA_ROOT = TAIL_DATA_ROOT
    base.BASELINE_OUTPUT_ROOT = TAIL_BASELINE_OUTPUT_ROOT
    base.PTL_OUTPUT_ROOT = TAIL_PTL_OUTPUT_ROOT
    base.SUMMARY_OUTPUT_DIR = TAIL_SUMMARY_OUTPUT_DIR
    base.AVAILABILITY_LEVELS = list(levels)
    base.data_root_for_level = lambda level, seed: data_root_for_level(level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PTL training-availability tests with contiguous tail-history retention "
            "for the 17-station set."
        )
    )
    parser.add_argument(
        "--step",
        action="append",
        choices=("prepare-data", "run-baselines", "run-ptl-gated", "summarize", "all"),
        default=[],
    )
    parser.add_argument("--level", action="append", type=int, default=[])
    parser.add_argument("--station", action="append", default=[])
    parser.add_argument("--model", action="append", choices=base.MODEL_ORDER, default=[])
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--time-start", default=base.TIME_START)
    parser.add_argument("--time-end", default=base.TIME_END)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-runs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-if-level-missing",
        type=int,
        default=40,
        help="Stop the gated PTL run if this level has any missing station.",
    )
    return parser.parse_args()


def expand_steps(steps: list[str]) -> set[str]:
    selected = set(steps or ["all"])
    if "all" in selected:
        return {"prepare-data", "run-baselines", "run-ptl-gated", "summarize"}
    return selected


def truncate_station_file_tail(
    station: str,
    resolution: str,
    level: int,
    source_path: Path,
    output_path: Path,
    time_start: str,
    time_end: str,
) -> dict[str, object]:
    availability = level / 100.0
    expected_freq = base.PTL_CORE.RESOLUTION_TO_FREQ[resolution]
    loaded = base.PTL_CORE.load_water_frame(
        str(source_path),
        time_start=time_start,
        time_end=time_end,
        expected_freq=expected_freq,
        feature_columns=base.FEATURE_COLUMNS,
    )
    if loaded is None:
        raise FileNotFoundError(f"Unable to load {resolution} data for {station}: {source_path}")

    train_end, _ = base.PTL_CORE.compute_split_points(len(loaded), 0.7, 0.1)
    train_slice = loaded.iloc[:train_end].copy()
    valid_train = train_slice[~train_slice["__gap__"]].copy()
    valid_count = int(len(valid_train))
    kept_count = int(round(valid_count * availability))
    kept_count = max(0, min(valid_count, kept_count))
    masked_count = valid_count - kept_count

    if masked_count > 0:
        masked_timestamps = set(pd.to_datetime(valid_train.iloc[:masked_count]["timestamp"]))
    else:
        masked_timestamps = set()
    if kept_count > 0:
        kept_train = valid_train.iloc[masked_count:]
        kept_start = str(pd.to_datetime(kept_train["timestamp"].iloc[0]))
        kept_end = str(pd.to_datetime(kept_train["timestamp"].iloc[-1]))
    else:
        kept_start = ""
        kept_end = ""

    raw = pd.read_csv(source_path)
    raw, timestamp_column = base.normalize_timestamp_column(raw)
    raw_timestamps = pd.to_datetime(raw[timestamp_column], errors="coerce")
    row_mask = raw_timestamps.isin(masked_timestamps)
    raw.loc[row_mask, base.FEATURE_COLUMNS] = np.nan
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_path, index=False)

    return {
        "availability": availability,
        "level_name": f"train{level}_tail",
        "station_name": station,
        "resolution": resolution,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "train_records_after_load": int(train_end),
        "valid_train_rows_before_mask": valid_count,
        "kept_valid_train_rows": kept_count,
        "masked_valid_train_rows": masked_count,
        "realized_valid_train_availability": kept_count / valid_count if valid_count else np.nan,
        "mask_strategy": "contiguous_tail_train_history",
        "kept_train_start": kept_start,
        "kept_train_end": kept_end,
    }


def prepare_tail_data(
    station_class: pd.DataFrame,
    levels: list[int],
    time_start: str,
    time_end: str,
    force_data: bool,
) -> None:
    TAIL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base.SOURCE_DATA_ROOT / "station_meta.csv", TAIL_DATA_ROOT / "station_meta.csv")

    rows: list[dict[str, object]] = []
    for level in levels:
        if level == 100:
            continue
        for station in station_class["station"].tolist():
            for resolution in ("daily", "weekly"):
                source_path = base.SOURCE_DATA_ROOT / resolution / f"{station}.csv"
                output_path = TAIL_DATA_ROOT / f"train{level}_tail" / resolution / f"{station}.csv"
                if output_path.exists() and not force_data:
                    # Recompute manifest metadata deterministically from the source, but leave the file untouched.
                    row = truncate_station_file_tail(
                        station=station,
                        resolution=resolution,
                        level=level,
                        source_path=source_path,
                        output_path=output_path,
                        time_start=time_start,
                        time_end=time_end,
                    )
                else:
                    row = truncate_station_file_tail(
                        station=station,
                        resolution=resolution,
                        level=level,
                        source_path=source_path,
                        output_path=output_path,
                        time_start=time_start,
                        time_end=time_end,
                    )
                rows.append(row)

    manifest = pd.DataFrame(rows).sort_values(
        ["availability", "resolution", "station_name"],
        ascending=[False, True, True],
    )
    manifest_path = TAIL_DATA_ROOT / "train_availability_17stations_manifest_seed42.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    config = {
        "source_data_root": str(base.SOURCE_DATA_ROOT),
        "availability_data_root": str(TAIL_DATA_ROOT),
        "levels": levels,
        "time_start": time_start,
        "time_end": time_end,
        "logic": (
            "Only originally valid rows in the train split are masked. "
            "For each station/resolution/level, the retained rows are the latest contiguous "
            "target-site training history; validation and test splits are not additionally masked."
        ),
        "mask_strategy": "contiguous_tail_train_history",
    }
    (TAIL_DATA_ROOT / "train_availability_17stations_config_seed42.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared tail-history availability data: {manifest_path}", flush=True)


def data_root_for_level(level: int) -> Path:
    if level == 100:
        return base.SOURCE_DATA_ROOT
    return TAIL_DATA_ROOT / f"train{level}_tail"


def run_ptl_one_level(
    station_class: pd.DataFrame,
    level: int,
    seed: int,
    time_start: str,
    time_end: str,
    force_runs: bool,
    dry_run: bool,
) -> None:
    stations = station_class["station"].tolist()
    missing = [station for station in stations if base.latest_ptl_stage3(level, station) is None]
    if not missing and not force_runs:
        print(f"SKIP PTL train{level}: all {len(stations)} stations already complete.", flush=True)
        return
    station_batch = stations if force_runs else missing
    save_root = TAIL_PTL_OUTPUT_ROOT / f"train{level}"
    save_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(base.PTL_RUNNER),
        "--data-root",
        str(data_root_for_level(level)),
        "--station-meta",
        str(base.SOURCE_DATA_ROOT / "station_meta.csv"),
        "--save-root",
        str(save_root),
        "--time-start",
        time_start,
        "--time-end",
        time_end,
        "--seed",
        str(seed),
    ]
    for station in station_batch:
        command.extend(["--station", station])
    log_path = save_root / "logs" / f"ptl_train{level}.log"
    print(f"RUN PTL tail train{level}: {len(station_batch)} stations", flush=True)
    base.run_command(command, log_path, dry_run=dry_run)


def write_summary(
    station_class: pd.DataFrame,
    levels: list[int],
    models: list[str],
    seed: int,
    make_plots: bool,
) -> pd.DataFrame:
    frame = base.write_summary_tables(
        station_class=station_class,
        levels=levels,
        models=models,
        seed=seed,
    )
    if make_plots:
        make_tail_plots(frame)
    return frame


def make_tail_plots(frame: pd.DataFrame) -> None:
    base.configure_plot_style()
    TAIL_SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completed_models = frame.loc[frame["status"] == "completed", "model"].dropna().unique().tolist()
    model_count = len(completed_models)
    single_axis_legend_cols = 1 if model_count <= 2 else 2
    summary_legend_cols = min(4, max(1, model_count))
    subject = "PTL" if set(completed_models) == {"PTL"} else "Model"
    note = (
        r"Focus features: COD$_{Mn}$, DO, and pH; all variables include NH$_4$-N. "
        "Error bars show SEM; n=17 for all levels."
    )

    fig, axis = base.plt.subplots(figsize=(4.6, 3.35))
    base.draw_metric_axis(
        axis,
        frame,
        metric="overall_nse",
        title="Overall NSE under Truncated Training History",
        y_label="Mean overall NSE across 17 stations",
    )
    axis.text(0.0, -0.27, note, transform=axis.transAxes, ha="left", va="top", fontsize=8.2)
    axis.legend(frameon=False, fontsize=7.7, ncol=single_axis_legend_cols, loc="lower right")
    base.save_figure(fig, "fig_training_availability_overall_nse")
    base.plt.close(fig)

    fig, axis = base.plt.subplots(figsize=(4.6, 3.35))
    base.draw_metric_axis(
        axis,
        frame,
        metric="focus_mean_nse",
        title="Focus-Feature NSE under Truncated Training History",
        y_label="Mean focus-feature NSE across 17 stations",
    )
    axis.text(0.0, -0.27, note, transform=axis.transAxes, ha="left", va="top", fontsize=8.2)
    axis.legend(frameon=False, fontsize=7.7, ncol=single_axis_legend_cols, loc="lower right")
    base.save_figure(fig, "fig_training_availability_focus_nse")
    base.plt.close(fig)

    fig, axes = base.plt.subplots(1, 2, figsize=(7.8, 4.2), sharex=True)
    base.draw_metric_axis(
        axes[0],
        frame,
        metric="overall_nse",
        title="Overall NSE",
        y_label="Mean NSE across 17 stations",
    )
    base.draw_metric_axis(
        axes[1],
        frame,
        metric="focus_mean_nse",
        title="Focus-Feature NSE",
        y_label="Mean NSE across 17 stations",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    if axes[0].get_legend():
        axes[0].legend().remove()
    if axes[1].get_legend():
        axes[1].legend().remove()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8.0,
        ncol=summary_legend_cols,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.095),
    )
    fig.suptitle(f"{subject} Robustness under Truncated Target-Site Training History", fontsize=11.0, y=0.98)
    fig.text(0.02, 0.02, note, ha="left", va="bottom", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.24, 1, 0.92))
    base.save_figure(fig, "fig_training_availability_nse_summary")
    base.plt.close(fig)
    print(f"Wrote tail-history figures under: {TAIL_SUMMARY_OUTPUT_DIR}", flush=True)


def run_ptl_gated(
    station_class: pd.DataFrame,
    levels: list[int],
    seed: int,
    time_start: str,
    time_end: str,
    force_runs: bool,
    dry_run: bool,
    stop_if_level_missing: int,
) -> None:
    for level in levels:
        run_ptl_one_level(
            station_class=station_class,
            level=level,
            seed=seed,
            time_start=time_start,
            time_end=time_end,
            force_runs=force_runs,
            dry_run=dry_run,
        )
        frame = write_summary(station_class, [level], models=["PTL"], seed=seed, make_plots=False)
        missing = frame[frame["status"] != "completed"].copy()
        if level == stop_if_level_missing and not missing.empty:
            stop_path = TAIL_SUMMARY_OUTPUT_DIR / f"ptl_train{level}_missing_stop.csv"
            missing.to_csv(stop_path, index=False, encoding="utf-8-sig")
            print(
                f"STOP: PTL train{level} has {len(missing)} missing station(s). "
                f"Missing list: {stop_path}",
                flush=True,
            )
            return
    write_summary(station_class, levels, models=["PTL"], seed=seed, make_plots=True)


def main() -> None:
    args = parse_args()
    steps = expand_steps(args.step)
    levels = sorted(set(args.level or DEFAULT_LEVELS))
    configure_base(levels)
    models = [model for model in base.MODEL_ORDER if model in set(args.model or base.MODEL_ORDER)]
    station_class = base.load_station_class(args.station)
    TAIL_SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (TAIL_SUMMARY_OUTPUT_DIR / "run_setup.json").write_text(
        json.dumps(
            {
                "levels": levels,
                "models": models,
                "stations": station_class["station"].tolist(),
                "seed": args.seed,
                "epochs": args.epochs,
                "time_start": args.time_start,
                "time_end": args.time_end,
                "data_root": str(TAIL_DATA_ROOT),
                "baseline_output_root": str(TAIL_BASELINE_OUTPUT_ROOT),
                "ptl_output_root": str(TAIL_PTL_OUTPUT_ROOT),
                "summary_output_dir": str(TAIL_SUMMARY_OUTPUT_DIR),
                "mask_strategy": "contiguous_tail_train_history",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if "prepare-data" in steps:
        prepare_tail_data(
            station_class=station_class,
            levels=levels,
            time_start=args.time_start,
            time_end=args.time_end,
            force_data=args.force_data,
        )
    if "run-baselines" in steps:
        base.run_baselines(
            station_class=station_class,
            levels=levels,
            models=models,
            seed=args.seed,
            epochs=args.epochs,
            time_start=args.time_start,
            time_end=args.time_end,
            force_runs=args.force_runs,
            dry_run=args.dry_run,
        )
    if "run-ptl-gated" in steps:
        run_ptl_gated(
            station_class=station_class,
            levels=levels,
            seed=args.seed,
            time_start=args.time_start,
            time_end=args.time_end,
            force_runs=args.force_runs,
            dry_run=args.dry_run,
            stop_if_level_missing=args.stop_if_level_missing,
        )
    if "summarize" in steps:
        write_summary(station_class, levels, models=models, seed=args.seed, make_plots=True)


if __name__ == "__main__":
    main()
