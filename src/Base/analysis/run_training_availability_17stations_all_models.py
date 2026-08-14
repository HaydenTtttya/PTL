from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPO_ROOT / "src" / "Base" / "benchmarks"
PTL_DIR = REPO_ROOT / "src" / "PTL"
PTL_RUNNER = PTL_DIR / "run_pearl_other_stations_core3_progressive_v2pretrain_v2_2021_2024.py"
SOURCE_DATA_ROOT = REPO_ROOT / "data" / "water_quality_processed_2021_2024"
LEGACY_AVAILABILITY_ROOT = (
    REPO_ROOT / "data" / "water_quality_processed_2021_2024_train_availability_current_selected"
)
AVAILABILITY_DATA_ROOT = (
    REPO_ROOT / "data" / "water_quality_processed_2021_2024_train_availability_17stations"
)
BASELINE_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "base"
    / "fair_compare"
    / "training_availability_17stations_all_models_2023_2024"
)
PTL_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "ptl"
    / "finetune"
    / "runs"
    / "training_availability_17stations_all_models_2023_2024"
)
SUMMARY_ROOT = REPO_ROOT / "results" / "summary" / "current_all_tested_stations_overall_nse"
SUMMARY_OUTPUT_DIR = SUMMARY_ROOT / "training_availability_17stations_all_models"
STATION_CLASS_PATH = SUMMARY_ROOT / "均衡十五站方案_新增两站" / "站点分类.csv"
EXISTING_17_MODEL_TABLE = SUMMARY_ROOT / "均衡十五站方案_新增两站" / "模型对比长表.csv"
EXISTING_17_FEATURE_TABLE = SUMMARY_ROOT / "均衡十五站方案_新增两站" / "各指标_站点_模型_NSE长表.csv"

TIME_START = "2023-01-01 00:00:00"
TIME_END = "2024-12-31 23:59:59"
FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
AVAILABILITY_LEVELS = [25, 50, 75, 100]
MASK_LEVELS = [25, 50, 75]
SEED = 42

MODEL_ORDER = ["MLP", "CNN", "LSTM", "Bi-LSTM", "CNN-LSTM", "Transformer", "PTL"]
BASELINE_MODELS = {
    "MLP": {
        "script": "benchmark_daily_mlp.py",
        "subdir": "mlp",
        "prefix": "mlp",
    },
    "CNN": {
        "script": "benchmark_daily_cnn.py",
        "subdir": "cnn",
        "prefix": "cnn",
    },
    "LSTM": {
        "script": "benchmark_daily_lstm.py",
        "subdir": "lstm",
        "prefix": "lstm",
    },
    "Bi-LSTM": {
        "script": "benchmark_daily_bilstm.py",
        "subdir": "bilstm",
        "prefix": "bilstm",
    },
    "CNN-LSTM": {
        "script": "benchmark_daily_cnn_lstm.py",
        "subdir": "cnn_lstm",
        "prefix": "cnn_lstm",
    },
    "Transformer": {
        "script": "benchmark_daily_basic_transformer.py",
        "subdir": "basic_transformer",
        "prefix": "basic_transformer",
    },
}
MODEL_COLORS = {
    "MLP": "#7b3294",
    "CNN": "#1b9e77",
    "LSTM": "#d95f02",
    "Bi-LSTM": "#7570b3",
    "CNN-LSTM": "#e7298a",
    "Transformer": "#66a61e",
    "PTL": "#1f78b4",
}
MODEL_MARKERS = {
    "MLP": "o",
    "CNN": "s",
    "LSTM": "^",
    "Bi-LSTM": "D",
    "CNN-LSTM": "P",
    "Transformer": "X",
    "PTL": "*",
}


def load_module(module_name: str, file_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Unable to load module: {file_path}")
    spec.loader.exec_module(module)
    return module


PTL_CORE = load_module("ptl_progressive_core_for_availability_17", PTL_DIR / "progressive_core.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 17-station training-availability experiments for all models.",
    )
    parser.add_argument(
        "--step",
        action="append",
        choices=("prepare-data", "run-baselines", "run-ptl", "summarize", "all"),
        default=[],
        help="Repeatable. Defaults to all.",
    )
    parser.add_argument("--station", action="append", default=[])
    parser.add_argument("--model", action="append", choices=MODEL_ORDER, default=[])
    parser.add_argument("--level", action="append", type=int, choices=AVAILABILITY_LEVELS, default=[])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--time-start", default=TIME_START)
    parser.add_argument("--time-end", default=TIME_END)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-runs", action="store_true")
    parser.add_argument("--no-copy-legacy-masks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_station_class(requested_stations: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(STATION_CLASS_PATH, encoding="utf-8-sig")
    if "站点顺序" in frame.columns:
        frame = frame.sort_values("站点顺序")
    if requested_stations:
        requested = list(dict.fromkeys(requested_stations))
        indexed = frame.set_index("station", drop=False)
        missing = [station for station in requested if station not in indexed.index]
        if missing:
            raise ValueError(f"Requested station(s) are not in {STATION_CLASS_PATH}: {missing}")
        frame = pd.DataFrame([indexed.loc[station] for station in requested])
    return frame.reset_index(drop=True)


def stable_seed(base_seed: int, level: int, station: str, resolution: str) -> int:
    availability = level / 100.0
    payload = f"{base_seed}|{station}|{resolution}|{availability}".encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def normalize_timestamp_column(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "timestamp" in frame.columns:
        return frame, "timestamp"
    first_column = frame.columns[0]
    return frame.rename(columns={first_column: "timestamp"}), "timestamp"


def mask_station_file(
    station: str,
    resolution: str,
    level: int,
    seed: int,
    source_path: Path,
    output_path: Path,
    time_start: str,
    time_end: str,
) -> dict[str, object]:
    availability = level / 100.0
    expected_freq = PTL_CORE.RESOLUTION_TO_FREQ[resolution]
    loaded = PTL_CORE.load_water_frame(
        str(source_path),
        time_start=time_start,
        time_end=time_end,
        expected_freq=expected_freq,
        feature_columns=FEATURE_COLUMNS,
    )
    if loaded is None:
        raise FileNotFoundError(f"Unable to load {resolution} data for {station}: {source_path}")

    train_end, _ = PTL_CORE.compute_split_points(len(loaded), 0.7, 0.1)
    train_slice = loaded.iloc[:train_end].copy()
    valid_train = train_slice[~train_slice["__gap__"]].copy()
    valid_count = int(len(valid_train))
    kept_count = int(round(valid_count * availability))
    kept_count = max(0, min(valid_count, kept_count))
    masked_count = valid_count - kept_count

    rng_seed = stable_seed(seed, level, station, resolution)
    rng = np.random.default_rng(rng_seed)
    if masked_count > 0:
        masked_positions = rng.choice(valid_count, size=masked_count, replace=False)
        masked_timestamps = set(pd.to_datetime(valid_train.iloc[masked_positions]["timestamp"]))
    else:
        masked_timestamps = set()

    raw = pd.read_csv(source_path)
    raw, timestamp_column = normalize_timestamp_column(raw)
    raw_timestamps = pd.to_datetime(raw[timestamp_column], errors="coerce")
    row_mask = raw_timestamps.isin(masked_timestamps)
    raw.loc[row_mask, FEATURE_COLUMNS] = np.nan
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_path, index=False)

    return {
        "availability": availability,
        "level_name": f"train{level}_seed{seed}",
        "station_name": station,
        "resolution": resolution,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "train_records_after_load": int(train_end),
        "valid_train_rows_before_mask": valid_count,
        "kept_valid_train_rows": kept_count,
        "masked_valid_train_rows": masked_count,
        "realized_valid_train_availability": kept_count / valid_count if valid_count else np.nan,
        "rng_seed": int(rng_seed),
        "copied_from_legacy": "",
    }


def load_legacy_manifest() -> pd.DataFrame:
    path = LEGACY_AVAILABILITY_ROOT / "train_availability_current_selected_manifest_seed42.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def prepare_availability_data(
    station_class: pd.DataFrame,
    levels: list[int],
    seed: int,
    time_start: str,
    time_end: str,
    force_data: bool,
    copy_legacy_masks: bool,
) -> None:
    AVAILABILITY_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_DATA_ROOT / "station_meta.csv", AVAILABILITY_DATA_ROOT / "station_meta.csv")
    legacy_manifest = load_legacy_manifest()
    legacy_lookup = {}
    if not legacy_manifest.empty:
        for _, row in legacy_manifest.iterrows():
            legacy_lookup[
                (int(round(float(row["availability"]) * 100)), row["station_name"], row["resolution"])
            ] = row

    rows: list[dict[str, object]] = []
    for level in levels:
        if level == 100:
            continue
        for station in station_class["station"].tolist():
            for resolution in ("daily", "weekly"):
                source_path = SOURCE_DATA_ROOT / resolution / f"{station}.csv"
                output_path = (
                    AVAILABILITY_DATA_ROOT
                    / f"train{level}_seed{seed}"
                    / resolution
                    / f"{station}.csv"
                )
                legacy_row = legacy_lookup.get((level, station, resolution))
                legacy_path = Path(str(legacy_row["output_path"])) if legacy_row is not None else None
                if (
                    copy_legacy_masks
                    and level in {50, 75}
                    and legacy_path is not None
                    and legacy_path.exists()
                ):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if force_data or not output_path.exists():
                        shutil.copy2(legacy_path, output_path)
                    row = {
                        "availability": float(legacy_row["availability"]),
                        "level_name": f"train{level}_seed{seed}",
                        "station_name": station,
                        "resolution": resolution,
                        "source_path": str(source_path),
                        "output_path": str(output_path),
                        "train_records_after_load": int(legacy_row["train_records_after_load"]),
                        "valid_train_rows_before_mask": int(legacy_row["valid_train_rows_before_mask"]),
                        "kept_valid_train_rows": int(legacy_row["kept_valid_train_rows"]),
                        "masked_valid_train_rows": int(legacy_row["masked_valid_train_rows"]),
                        "realized_valid_train_availability": float(
                            legacy_row["realized_valid_train_availability"]
                        ),
                        "rng_seed": int(legacy_row["rng_seed"]),
                        "copied_from_legacy": str(legacy_path),
                    }
                else:
                    if not source_path.exists():
                        raise FileNotFoundError(f"Missing source data: {source_path}")
                    row = mask_station_file(
                        station=station,
                        resolution=resolution,
                        level=level,
                        seed=seed,
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
    manifest_path = AVAILABILITY_DATA_ROOT / "train_availability_17stations_manifest_seed42.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    config = {
        "source_data_root": str(SOURCE_DATA_ROOT),
        "availability_data_root": str(AVAILABILITY_DATA_ROOT),
        "station_class_path": str(STATION_CLASS_PATH),
        "levels": levels,
        "seed": seed,
        "time_start": time_start,
        "time_end": time_end,
        "logic": "Randomly mask valid target-site training rows; validation and test windows are unchanged.",
        "seed_method": "zlib.crc32(f\"{seed}|{station}|{resolution}|{availability}\")",
        "copy_legacy_masks": bool(copy_legacy_masks),
        "legacy_availability_root": str(LEGACY_AVAILABILITY_ROOT),
    }
    (AVAILABILITY_DATA_ROOT / "train_availability_17stations_config_seed42.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared availability data: {manifest_path}", flush=True)


def data_root_for_level(level: int, seed: int) -> Path:
    if level == 100:
        return SOURCE_DATA_ROOT
    return AVAILABILITY_DATA_ROOT / f"train{level}_seed{seed}"


def latest_baseline_run(output_dir: Path, prefix: str, station: str) -> Path | None:
    candidates = sorted(
        output_dir.glob(f"{prefix}_{station}_seed*_*/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent if candidates else None


def run_command(command: list[str], log_path: Path, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        result = subprocess.run(
            command,
            cwd=str(PTL_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}; see {log_path}")
    print(f"Finished in {time.time() - started_at:.1f}s; log={log_path}", flush=True)


def run_baselines(
    station_class: pd.DataFrame,
    levels: list[int],
    models: list[str],
    seed: int,
    epochs: int,
    time_start: str,
    time_end: str,
    force_runs: bool,
    dry_run: bool,
) -> None:
    selected_models = [model for model in models if model in BASELINE_MODELS]
    total = len(station_class) * len(levels) * len(selected_models)
    completed = 0
    for level in levels:
        data_root = data_root_for_level(level, seed)
        for station in station_class["station"].tolist():
            data_path = data_root / "daily" / f"{station}.csv"
            if not data_path.exists():
                raise FileNotFoundError(f"Missing daily data for train{level}: {data_path}")
            for model in selected_models:
                spec = BASELINE_MODELS[model]
                output_dir = BASELINE_OUTPUT_ROOT / f"train{level}" / spec["subdir"]
                output_dir.mkdir(parents=True, exist_ok=True)
                existing = latest_baseline_run(output_dir, spec["prefix"], station)
                completed += 1
                if existing is not None and not force_runs:
                    print(f"[{completed}/{total}] SKIP {model} train{level} {station}: {existing}", flush=True)
                    continue
                log_path = (
                    BASELINE_OUTPUT_ROOT
                    / f"train{level}"
                    / "logs"
                    / f"{spec['prefix']}_{station}.log"
                )
                command = [
                    sys.executable,
                    str(BENCHMARK_DIR / spec["script"]),
                    "--data-path",
                    str(data_path),
                    "--output-root",
                    str(output_dir),
                    "--ptl-reference-dir",
                    str(BASELINE_OUTPUT_ROOT / "_no_reference"),
                    "--station-name",
                    station,
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(epochs),
                    "--time-start",
                    time_start,
                    "--time-end",
                    time_end,
                    "--soft-gap-max-steps",
                    "6",
                    "--invalid-window-policy",
                    "all",
                ]
                print(f"[{completed}/{total}] RUN {model} train{level} {station}", flush=True)
                run_command(command, log_path, dry_run=dry_run)


def latest_ptl_stage3(level: int, station: str) -> Path | None:
    parent = PTL_OUTPUT_ROOT / f"train{level}"
    candidates = sorted(
        parent.glob(f"batch_*/progressive_{station}_seed42_*/stage3_daily/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent if candidates else None


def run_ptl(
    station_class: pd.DataFrame,
    levels: list[int],
    models: list[str],
    seed: int,
    time_start: str,
    time_end: str,
    force_runs: bool,
    dry_run: bool,
) -> None:
    if "PTL" not in models:
        return
    stations = station_class["station"].tolist()
    for level in levels:
        missing = [station for station in stations if latest_ptl_stage3(level, station) is None]
        if not missing and not force_runs:
            print(f"SKIP PTL train{level}: all {len(stations)} stations already complete.", flush=True)
            continue
        station_batch = stations if force_runs else missing
        data_root = data_root_for_level(level, seed)
        save_root = PTL_OUTPUT_ROOT / f"train{level}"
        save_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(PTL_RUNNER),
            "--data-root",
            str(data_root),
            "--station-meta",
            str(SOURCE_DATA_ROOT / "station_meta.csv"),
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
        print(f"RUN PTL train{level}: {len(station_batch)} stations", flush=True)
        run_command(command, log_path, dry_run=dry_run)


def to_float(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def read_metrics_csv(metrics_path: Path) -> dict[str, dict[str, object]]:
    return pd.read_csv(metrics_path, index_col=0).replace({np.nan: None}).to_dict(orient="index")


def flatten_metrics(metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    row: dict[str, object] = {}
    overall = metrics.get("__overall__", {})
    row["overall_nse"] = to_float(overall.get("NSE"))
    row["overall_rmse"] = to_float(overall.get("RMSE"))
    row["overall_mae"] = to_float(overall.get("MAE"))
    focus_nse = []
    focus_rmse = []
    focus_mae = []
    for feature in FOCUS_FEATURES:
        feature_metrics = metrics.get(feature, {})
        if feature_metrics.get("NSE") is not None:
            focus_nse.append(float(feature_metrics["NSE"]))
        if feature_metrics.get("RMSE") is not None:
            focus_rmse.append(float(feature_metrics["RMSE"]))
        if feature_metrics.get("MAE") is not None:
            focus_mae.append(float(feature_metrics["MAE"]))
    row["focus_mean_nse"] = float(np.mean(focus_nse)) if focus_nse else np.nan
    row["focus_mean_rmse"] = float(np.mean(focus_rmse)) if focus_rmse else np.nan
    row["focus_mean_mae"] = float(np.mean(focus_mae)) if focus_mae else np.nan
    for feature in FEATURE_COLUMNS:
        feature_metrics = metrics.get(feature, {})
        key = feature.lower()
        row[f"{key}_nse"] = to_float(feature_metrics.get("NSE"))
        row[f"{key}_rmse"] = to_float(feature_metrics.get("RMSE"))
        row[f"{key}_mae"] = to_float(feature_metrics.get("MAE"))
    return row


def collect_manifest_lookup() -> dict[tuple[str, int, str], dict[str, object]]:
    manifest_path = AVAILABILITY_DATA_ROOT / "train_availability_17stations_manifest_seed42.csv"
    if not manifest_path.exists():
        return {}
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    lookup = {}
    for _, row in manifest.iterrows():
        level = int(round(float(row["availability"]) * 100))
        lookup[(row["station_name"], level, row["resolution"])] = row.to_dict()
    return lookup


def read_run_meta(run_dir: Path) -> dict[str, object]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def collect_from_existing_100_summary(station: str, model: str) -> dict[str, object] | None:
    if not EXISTING_17_MODEL_TABLE.exists():
        return None
    model_frame = pd.read_csv(EXISTING_17_MODEL_TABLE, encoding="utf-8-sig")
    matched = model_frame[(model_frame["station"] == station) & (model_frame["model"] == model)]
    if matched.empty:
        return None
    base = matched.iloc[0].to_dict()
    row = {
        "status": "completed",
        "source": "existing_17_station_summary",
        "run_dir": "",
        "metrics_path": str(EXISTING_17_MODEL_TABLE),
        "meta_path": "",
        "overall_nse": to_float(base.get("overall_nse")),
        "overall_rmse": to_float(base.get("overall_rmse")),
        "overall_mae": to_float(base.get("overall_mae")),
        "focus_mean_nse": to_float(base.get("focus_mean_nse")),
        "focus_mean_rmse": to_float(base.get("focus_mean_rmse")),
        "focus_mean_mae": to_float(base.get("focus_mean_mae")),
        "train_windows": base.get("train_windows"),
        "val_windows": base.get("val_windows"),
        "test_windows": base.get("test_windows"),
        "uses_pretraining": base.get("uses_pretraining"),
        "uses_progressive_transfer": base.get("uses_progressive_transfer"),
    }
    if EXISTING_17_FEATURE_TABLE.exists():
        feature_frame = pd.read_csv(EXISTING_17_FEATURE_TABLE, encoding="utf-8-sig")
        feature_rows = feature_frame[
            (feature_frame["station"] == station)
            & (feature_frame["model"] == model)
            & (feature_frame["indicator"].isin(FEATURE_COLUMNS))
        ]
        for _, feature_row in feature_rows.iterrows():
            row[f"{str(feature_row['indicator']).lower()}_nse"] = to_float(feature_row["nse"])
    return row


def collect_model_row(
    station_row: dict[str, object],
    level: int,
    model: str,
    seed: int,
    manifest_lookup: dict[tuple[str, int, str], dict[str, object]],
) -> dict[str, object]:
    station = str(station_row["station"])
    row = {
        "station": station,
        "station_order": station_row.get("站点顺序"),
        "river_reach": station_row.get("river_reach"),
        "river_type": station_row.get("river_type"),
        "verified_waterbody": station_row.get("verified_waterbody"),
        "comparison_group": station_row.get("comparison_group"),
        "model": model,
        "training_availability_pct": level,
        "training_missing_pct": 100 - level,
        "status": "missing",
        "source": "",
    }
    manifest_row = manifest_lookup.get((station, level, "daily"))
    if level == 100:
        row.update(
            {
                "realized_valid_train_availability_pct": 100.0,
                "realized_valid_train_missing_pct": 0.0,
                "valid_train_rows_before_mask": np.nan,
                "kept_valid_train_rows": np.nan,
                "masked_valid_train_rows": np.nan,
            }
        )
    elif manifest_row:
        realized = float(manifest_row["realized_valid_train_availability"]) * 100.0
        row.update(
            {
                "realized_valid_train_availability_pct": realized,
                "realized_valid_train_missing_pct": 100.0 - realized,
                "valid_train_rows_before_mask": manifest_row["valid_train_rows_before_mask"],
                "kept_valid_train_rows": manifest_row["kept_valid_train_rows"],
                "masked_valid_train_rows": manifest_row["masked_valid_train_rows"],
            }
        )

    if model == "PTL":
        run_dir = latest_ptl_stage3(level, station)
        if run_dir is None:
            if level == 100:
                existing = collect_from_existing_100_summary(station, model)
                if existing is not None:
                    row.update(existing)
            return row
    else:
        spec = BASELINE_MODELS[model]
        output_dir = BASELINE_OUTPUT_ROOT / f"train{level}" / spec["subdir"]
        run_dir = latest_baseline_run(output_dir, spec["prefix"], station)
        if run_dir is None:
            if level == 100:
                existing = collect_from_existing_100_summary(station, model)
                if existing is not None:
                    row.update(existing)
            return row

    metrics_path = run_dir / "metrics.csv"
    meta = read_run_meta(run_dir)
    metrics = read_metrics_csv(metrics_path)
    row.update(flatten_metrics(metrics))
    row.update(
        {
            "status": "completed",
            "source": "new_run",
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
            "meta_path": str(run_dir / "meta.json"),
            "train_windows": meta.get("train_windows"),
            "val_windows": meta.get("val_windows"),
            "test_windows": meta.get("test_windows"),
            "records": meta.get("records"),
            "invalid_records": meta.get("invalid_records"),
            "best_epoch": meta.get("best_epoch"),
            "best_val_nse": meta.get("best_val_nse"),
            "test_nse_meta": meta.get("test_nse"),
            "uses_pretraining": model == "PTL",
            "uses_progressive_transfer": model == "PTL",
        }
    )
    return row


def write_summary_tables(
    station_class: pd.DataFrame,
    levels: list[int],
    models: list[str],
    seed: int,
) -> pd.DataFrame:
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_lookup = collect_manifest_lookup()
    rows = []
    for station_row in station_class.to_dict(orient="records"):
        for level in levels:
            for model in models:
                rows.append(collect_model_row(station_row, level, model, seed, manifest_lookup))
    frame = pd.DataFrame(rows)
    model_order_map = {model: index for index, model in enumerate(MODEL_ORDER)}
    frame["model_order"] = frame["model"].map(model_order_map)
    frame = frame.sort_values(["station_order", "training_availability_pct", "model_order"]).drop(
        columns=["model_order"]
    )
    long_path = SUMMARY_OUTPUT_DIR / "training_availability_17stations_all_models_metrics_long.csv"
    frame.to_csv(long_path, index=False, encoding="utf-8-sig")

    completed = frame[frame["status"] == "completed"].copy()
    group_summary = (
        completed.groupby(["model", "training_availability_pct", "training_missing_pct"], as_index=False)
        .agg(
            station_count=("station", "nunique"),
            mean_overall_nse=("overall_nse", "mean"),
            median_overall_nse=("overall_nse", "median"),
            std_overall_nse=("overall_nse", "std"),
            mean_focus_nse=("focus_mean_nse", "mean"),
            median_focus_nse=("focus_mean_nse", "median"),
            std_focus_nse=("focus_mean_nse", "std"),
            mean_train_windows=("train_windows", "mean"),
        )
    )
    group_summary["overall_nse_sem"] = group_summary["std_overall_nse"] / np.sqrt(
        group_summary["station_count"].clip(lower=1)
    )
    group_summary["focus_nse_sem"] = group_summary["std_focus_nse"] / np.sqrt(
        group_summary["station_count"].clip(lower=1)
    )
    group_summary["model_order"] = group_summary["model"].map(model_order_map)
    group_summary = group_summary.sort_values(["model_order", "training_availability_pct"]).drop(
        columns=["model_order"]
    )
    group_summary.to_csv(
        SUMMARY_OUTPUT_DIR / "training_availability_17stations_all_models_model_average_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for metric, filename in (
        ("overall_nse", "training_availability_17stations_all_models_overall_nse_wide.csv"),
        ("focus_mean_nse", "training_availability_17stations_all_models_focus_nse_wide.csv"),
    ):
        wide = completed.pivot_table(
            index=["station_order", "station"],
            columns=["model", "training_availability_pct"],
            values=metric,
            aggfunc="first",
        )
        wide.columns = [f"{model}_train{level}" for model, level in wide.columns]
        wide.reset_index().to_csv(SUMMARY_OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")

    missing = frame[frame["status"] != "completed"].copy()
    missing.to_csv(
        SUMMARY_OUTPUT_DIR / "training_availability_17stations_all_models_missing_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Wrote summary table: {long_path}", flush=True)
    return frame


def choose_font_name(candidates: list[str], fallback: str) -> str:
    available = set()
    for path in font_manager.findSystemFonts():
        try:
            available.add(font_manager.FontProperties(fname=path).get_name())
        except RuntimeError:
            continue
    for candidate in candidates:
        if candidate in available:
            return candidate
    return fallback


def configure_plot_style() -> None:
    font_name = choose_font_name(["Times New Roman", "Times"], "DejaVu Serif")
    plt.rcParams.update(
        {
            "font.family": font_name,
            "mathtext.fontset": "stix",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def aggregate_for_plot(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    completed = frame[frame["status"] == "completed"].copy()
    grouped = (
        completed.groupby(["model", "training_availability_pct"], as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"), count=("station", "nunique"))
        .sort_values(["model", "training_availability_pct"])
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
    return grouped


def draw_metric_axis(axis, frame: pd.DataFrame, metric: str, title: str, y_label: str) -> None:
    data = aggregate_for_plot(frame, metric)
    for model in MODEL_ORDER:
        model_data = data[data["model"] == model].sort_values("training_availability_pct")
        if model_data.empty:
            continue
        axis.errorbar(
            model_data["training_availability_pct"],
            model_data["mean"],
            yerr=model_data["sem"].fillna(0.0),
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=1.55,
            markersize=5.2 if model != "PTL" else 7.0,
            capsize=2.8,
            label=model,
        )
    axis.set_title(title, fontsize=10.5, pad=7)
    axis.set_xlabel("Training data availability (%)", fontsize=9.5)
    axis.set_ylabel(y_label, fontsize=9.5)
    axis.set_xticks(AVAILABILITY_LEVELS)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.65, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8.8)


def save_figure(fig, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(SUMMARY_OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches="tight")


def make_plots(frame: pd.DataFrame) -> None:
    configure_plot_style()
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    note = (
        r"Focus features: COD$_{Mn}$, DO, and pH; all variables include NH$_4$-N. "
        "Error bars show SEM; n=17 except PTL at 25% (n=9)."
    )

    fig, axis = plt.subplots(figsize=(4.6, 3.35))
    draw_metric_axis(
        axis,
        frame,
        metric="overall_nse",
        title="Overall NSE under Reduced Training Data",
        y_label="Mean overall NSE across target stations",
    )
    axis.text(0.0, -0.27, note, transform=axis.transAxes, ha="left", va="top", fontsize=8.2)
    axis.legend(frameon=False, fontsize=7.7, ncol=2, loc="lower right")
    save_figure(fig, "fig_training_availability_overall_nse")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.6, 3.35))
    draw_metric_axis(
        axis,
        frame,
        metric="focus_mean_nse",
        title="Focus-Feature NSE under Reduced Training Data",
        y_label="Mean focus-feature NSE across target stations",
    )
    axis.text(0.0, -0.27, note, transform=axis.transAxes, ha="left", va="top", fontsize=8.2)
    axis.legend(frameon=False, fontsize=7.7, ncol=2, loc="lower right")
    save_figure(fig, "fig_training_availability_focus_nse")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.2), sharex=True)
    draw_metric_axis(
        axes[0],
        frame,
        metric="overall_nse",
        title="Overall NSE",
        y_label="Mean NSE across target stations",
    )
    draw_metric_axis(
        axes[1],
        frame,
        metric="focus_mean_nse",
        title="Focus-Feature NSE",
        y_label="Mean NSE across target stations",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend().remove() if axes[0].get_legend() else None
    axes[1].legend().remove() if axes[1].get_legend() else None
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8.0,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.095),
    )
    fig.suptitle("Model Robustness under Reduced Target-Site Training Data", fontsize=11.0, y=0.98)
    fig.text(0.02, 0.02, note, ha="left", va="bottom", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.24, 1, 0.92))
    save_figure(fig, "fig_training_availability_nse_summary")
    plt.close(fig)
    print(f"Wrote figures under: {SUMMARY_OUTPUT_DIR}", flush=True)


def expand_steps(steps: list[str]) -> set[str]:
    selected = set(steps or ["all"])
    if "all" in selected:
        return {"prepare-data", "run-baselines", "run-ptl", "summarize"}
    return selected


def main() -> None:
    args = parse_args()
    steps = expand_steps(args.step)
    levels = sorted(set(args.level or AVAILABILITY_LEVELS))
    models = [model for model in MODEL_ORDER if model in set(args.model or MODEL_ORDER)]
    station_class = load_station_class(args.station)

    setup = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": sorted(steps),
        "levels": levels,
        "models": models,
        "stations": station_class["station"].tolist(),
        "seed": args.seed,
        "time_start": args.time_start,
        "time_end": args.time_end,
        "availability_data_root": str(AVAILABILITY_DATA_ROOT),
        "baseline_output_root": str(BASELINE_OUTPUT_ROOT),
        "ptl_output_root": str(PTL_OUTPUT_ROOT),
        "summary_output_dir": str(SUMMARY_OUTPUT_DIR),
    }
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_OUTPUT_DIR / "run_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if "prepare-data" in steps:
        prepare_availability_data(
            station_class=station_class,
            levels=levels,
            seed=args.seed,
            time_start=args.time_start,
            time_end=args.time_end,
            force_data=args.force_data,
            copy_legacy_masks=not args.no_copy_legacy_masks,
        )
    if "run-baselines" in steps:
        run_baselines(
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
    if "run-ptl" in steps:
        run_ptl(
            station_class=station_class,
            levels=levels,
            models=models,
            seed=args.seed,
            time_start=args.time_start,
            time_end=args.time_end,
            force_runs=args.force_runs,
            dry_run=args.dry_run,
        )
    if "summarize" in steps:
        frame = write_summary_tables(station_class, levels, models, seed=args.seed)
        make_plots(frame)


if __name__ == "__main__":
    main()
