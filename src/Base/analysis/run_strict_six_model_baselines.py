from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPO_ROOT / "src" / "Base" / "benchmarks"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "water_quality_processed_2021_2024"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "base" / "fair_compare" / "full_six_model_strict_2023_2024"
DEFAULT_TIME_START = "2023-01-01 00:00:00"
DEFAULT_TIME_END = "2024-12-31 23:59:59"
DEFAULT_STATIONS = [
    "盘溪大桥",
    "坝草",
    "老口",
    "鸦岗",
    "龟山塔",
    "珠海大桥",
    "新铺",
    "大墩",
    "五丰渡口",
    "上中",
    "白马",
    "阳朔",
    "交州",
    "桂花",
    "深圳河口",
    "象州运江老街",
]

MODEL_SPECS = {
    "MLP": {
        "script": "benchmark_daily_mlp.py",
        "subdir": "mlp",
        "run_prefix": "mlp",
    },
    "CNN": {
        "script": "benchmark_daily_cnn.py",
        "subdir": "cnn",
        "run_prefix": "cnn",
    },
    "LSTM": {
        "script": "benchmark_daily_lstm.py",
        "subdir": "lstm",
        "run_prefix": "lstm",
    },
    "Bi-LSTM": {
        "script": "benchmark_daily_bilstm.py",
        "subdir": "bilstm",
        "run_prefix": "bilstm",
    },
    "CNN-LSTM": {
        "script": "benchmark_daily_cnn_lstm.py",
        "subdir": "cnn_lstm",
        "run_prefix": "cnn_lstm",
    },
    "Transformer": {
        "script": "benchmark_daily_basic_transformer.py",
        "subdir": "basic_transformer",
        "run_prefix": "basic_transformer",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict no-gap daily baseline models for tested stations.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--time-start", default=DEFAULT_TIME_START)
    parser.add_argument("--time-end", default=DEFAULT_TIME_END)
    parser.add_argument("--station", action="append", default=[])
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODEL_SPECS),
        default=[],
        help="Repeat to run selected models. Defaults to all direct baselines.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def latest_completed_run(output_dir: Path, run_prefix: str, station_name: str):
    pattern = f"{run_prefix}_{station_name}_seed*_*/meta.json"
    candidates = sorted(output_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for meta_path in candidates:
        metrics_path = meta_path.parent / "metrics.csv"
        if metrics_path.exists():
            return meta_path.parent
    return None


def build_command(args: argparse.Namespace, model_name: str, station_name: str, output_dir: Path) -> list[str]:
    spec = MODEL_SPECS[model_name]
    data_path = args.data_root / "daily" / f"{station_name}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing daily data for {station_name}: {data_path}")

    return [
        sys.executable,
        str(BENCHMARK_DIR / spec["script"]),
        "--data-path",
        str(data_path),
        "--output-root",
        str(output_dir),
        "--ptl-reference-dir",
        str(args.output_root / "_no_reference"),
        "--station-name",
        station_name,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--time-start",
        args.time_start,
        "--time-end",
        args.time_end,
        "--soft-gap-max-steps",
        "0",
        "--invalid-window-policy",
        "all",
    ]


def main() -> None:
    args = parse_args()
    stations = list(dict.fromkeys(args.station or DEFAULT_STATIONS))
    models = list(dict.fromkeys(args.model or MODEL_SPECS.keys()))
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_root / "baseline_run_manifest.jsonl"
    total = len(stations) * len(models)
    completed = 0

    for station_name in stations:
        for model_name in models:
            spec = MODEL_SPECS[model_name]
            output_dir = args.output_root / spec["subdir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            existing = latest_completed_run(output_dir, spec["run_prefix"], station_name)
            if existing is not None and not args.force:
                completed += 1
                print(f"[{completed}/{total}] SKIP {model_name} {station_name}: {existing}", flush=True)
                continue

            command = build_command(args, model_name, station_name, output_dir)
            log_path = log_dir / f"{spec['run_prefix']}_{station_name}.log"
            print(f"[{completed + 1}/{total}] RUN {model_name} {station_name}", flush=True)
            print(" ".join(command), flush=True)

            if args.dry_run:
                completed += 1
                continue

            started_at = time.time()
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(" ".join(command) + "\n\n")
                result = subprocess.run(
                    command,
                    cwd=str(REPO_ROOT / "src" / "PTL"),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            finished_at = time.time()
            row = {
                "station": station_name,
                "model": model_name,
                "returncode": result.returncode,
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": finished_at - started_at,
                "log_path": str(log_path),
                "output_dir": str(output_dir),
                "command": command,
            }
            with manifest_path.open("a", encoding="utf-8") as manifest_file:
                manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")

            if result.returncode != 0:
                raise RuntimeError(f"{model_name} {station_name} failed; see {log_path}")
            completed += 1

    print(f"Completed baseline batch: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
