from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from progressive_core import compute_per_feature_metrics


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
SOURCE_PROFILE = "unified_feature_token_safe_transfer_v4"
OUTPUT_PROFILE = "mlp_direct_anchored_safe_transfer_v8"
DEFAULT_BLEND_ALPHA = 0.05
DEFAULT_CURRENT_SUMMARY = (
    REPO_ROOT
    / "results"
    / "summary"
    / "current_all_tested_stations_overall_nse"
    / "均衡十五站方案_新增两站"
)
DEFAULT_BASELINE_LONG = DEFAULT_CURRENT_SUMMARY / "模型对比长表.csv"
DEFAULT_STATION_CLASS = DEFAULT_CURRENT_SUMMARY / "站点分类.csv"
DEFAULT_FINETUNE_ROOT = (
    REPO_ROOT / "results" / "ptl" / "finetune" / "runs" / "model_agnostic_17stations"
)
DEFAULT_DIRECT_SEARCH_ROOT = REPO_ROOT / "results" / "base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fixed MLP safeguard forecast: Direct + alpha * (PTL - Direct)."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend-alpha", type=float, default=DEFAULT_BLEND_ALPHA)
    parser.add_argument("--baseline-long", type=Path, default=DEFAULT_BASELINE_LONG)
    parser.add_argument("--station-class", type=Path, default=DEFAULT_STATION_CLASS)
    parser.add_argument("--finetune-root", type=Path, default=DEFAULT_FINETUNE_ROOT)
    parser.add_argument("--direct-search-root", type=Path, default=DEFAULT_DIRECT_SEARCH_ROOT)
    parser.add_argument("--station", action="append", default=[])
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def focus_nse(metrics_path: Path) -> float:
    metrics = pd.read_csv(metrics_path, index_col=0)
    return float(metrics.loc[FOCUS_FEATURES, "NSE"].mean())


def find_direct_run(
    search_root: Path,
    station: str,
    expected_focus_nse: float,
) -> Path:
    matches = []
    for meta_path in search_root.rglob("meta.json"):
        run_dir = meta_path.parent
        metrics_path = run_dir / "metrics.csv"
        predictions_path = run_dir / "predictions.csv"
        if not metrics_path.exists() or not predictions_path.exists():
            continue
        try:
            meta = read_json(meta_path)
            if meta.get("station_name") != station:
                continue
            if abs(focus_nse(metrics_path) - expected_focus_nse) <= 1e-7:
                matches.append(run_dir)
        except (KeyError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
            continue
    if len(matches) != 1:
        raise ValueError(
            f"Expected one official Direct MLP run for {station}, found {len(matches)}."
        )
    return matches[0]


def find_source_ptl_run(finetune_root: Path, station: str, seed: int) -> Path:
    root = finetune_root / "optimization" / SOURCE_PROFILE / "mlp"
    candidates = sorted(
        root.glob(f"progressive_{station}_seed{seed}_*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in candidates:
        run_dir = summary_path.parent
        meta_path = run_dir / "stage3_daily" / "meta.json"
        predictions_path = run_dir / "stage3_daily" / "predictions.csv"
        if not meta_path.exists() or not predictions_path.exists():
            continue
        summary = read_json(summary_path)
        meta = read_json(meta_path)
        if (
            summary.get("status") == "completed"
            and meta.get("optimization_profile") == SOURCE_PROFILE
            and bool(meta.get("uses_pretraining"))
        ):
            return run_dir
    raise FileNotFoundError(f"Missing completed MLP PTL source run for {station}.")


def build_blended_predictions(
    direct_path: Path,
    ptl_path: Path,
    blend_alpha: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    direct = pd.read_csv(direct_path)
    ptl = pd.read_csv(ptl_path)
    merged = direct.merge(
        ptl,
        on="timestamp",
        how="inner",
        suffixes=("_direct", "_ptl"),
        validate="one_to_one",
    )
    if len(merged) != len(direct) or len(merged) != len(ptl):
        raise ValueError(
            f"Prediction timestamps do not align: Direct={len(direct)}, PTL={len(ptl)}, "
            f"merged={len(merged)}."
        )

    output = pd.DataFrame({"timestamp": merged["timestamp"]})
    predictions = []
    targets = []
    for feature in FEATURE_COLUMNS:
        direct_true = merged[f"True_{feature}_direct"].to_numpy(dtype=float)
        ptl_true = merged[f"True_{feature}_ptl"].to_numpy(dtype=float)
        if not np.allclose(direct_true, ptl_true, rtol=0.0, atol=1e-8):
            raise ValueError(f"Target mismatch for {feature} in {direct_path} and {ptl_path}.")
        direct_pred = merged[f"Pred_{feature}_direct"].to_numpy(dtype=float)
        ptl_pred = merged[f"Pred_{feature}_ptl"].to_numpy(dtype=float)
        blended_pred = direct_pred + blend_alpha * (ptl_pred - direct_pred)
        output[f"True_{feature}"] = direct_true
        output[f"Pred_{feature}"] = blended_pred
        targets.append(direct_true)
        predictions.append(blended_pred)
    return output, np.column_stack(predictions), np.column_stack(targets)


def write_run(
    args: argparse.Namespace,
    station: str,
    direct_run: Path,
    ptl_run: Path,
    timestamp: str,
) -> Path:
    output_root = (
        args.finetune_root / "optimization" / OUTPUT_PROFILE / "mlp"
    )
    run_dir = output_root / f"progressive_{station}_seed{args.seed}_{timestamp}"
    stage_dir = run_dir / "stage3_daily"
    stage_dir.mkdir(parents=True, exist_ok=False)

    direct_predictions = direct_run / "predictions.csv"
    ptl_predictions = ptl_run / "stage3_daily" / "predictions.csv"
    output, predictions, targets = build_blended_predictions(
        direct_predictions,
        ptl_predictions,
        args.blend_alpha,
    )
    metrics = compute_per_feature_metrics(
        predictions,
        targets,
        feature_names=FEATURE_COLUMNS,
    )
    output.to_csv(stage_dir / "predictions.csv", index=False)
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(stage_dir / "metrics.csv")

    source_meta = read_json(ptl_run / "stage3_daily" / "meta.json")
    meta = {
        "station_name": station,
        "backbone_name": "mlp",
        "model_agnostic_interface": "feature_token_residual_v2",
        "uses_pretraining": True,
        "uses_progressive_transfer": True,
        "initialization_source": "cross_station_pretrain",
        "pretrain_model_dir": source_meta.get("pretrain_model_dir"),
        "optimization_profile": OUTPUT_PROFILE,
        "source_ptl_profile": SOURCE_PROFILE,
        "derived_postprocess": "direct_anchored_ptl_residual",
        "blend_formula": "Direct + alpha * (PTL - Direct)",
        "blend_alpha": args.blend_alpha,
        "direct_run_dir": str(direct_run.resolve()),
        "source_ptl_run_dir": str(ptl_run.resolve()),
        "direct_predictions": str(direct_predictions.resolve()),
        "source_ptl_predictions": str(ptl_predictions.resolve()),
        "metrics": metrics["__overall__"],
    }
    (stage_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "station_name": station,
        "backbone_name": "mlp",
        "model_agnostic_interface": "feature_token_residual_v2",
        "uses_pretraining": True,
        "uses_progressive_transfer": True,
        "pretrain_model_dir": source_meta.get("pretrain_model_dir"),
        "optimization_profile": OUTPUT_PROFILE,
        "source_ptl_profile": SOURCE_PROFILE,
        "blend_alpha": args.blend_alpha,
        "status": "completed",
        "stages": [
            {
                "stage_name": "stage3_daily",
                "save_dir": str(stage_dir.resolve()),
                "derived_from_complete_ptl": True,
            }
        ],
        "save_dir": str(run_dir.resolve()),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("The current comparison is fixed to seed 42.")
    if not 0.0 < args.blend_alpha <= 1.0:
        raise ValueError("blend-alpha must be in (0, 1].")

    baseline = pd.read_csv(args.baseline_long, encoding="utf-8-sig")
    baseline = baseline[baseline["model"].eq("MLP")].set_index("station")
    station_class = pd.read_csv(args.station_class, encoding="utf-8-sig")
    if "站点顺序" in station_class.columns:
        station_class = station_class.sort_values("站点顺序")
    stations = station_class["station"].astype(str).tolist()
    if args.station:
        requested = list(dict.fromkeys(args.station))
        missing = sorted(set(requested).difference(stations))
        if missing:
            raise ValueError(f"Unknown stations: {missing}")
        stations = requested

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = []
    for index, station in enumerate(stations, start=1):
        expected_focus_nse = float(baseline.loc[station, "focus_mean_nse"])
        direct_run = find_direct_run(
            args.direct_search_root,
            station,
            expected_focus_nse,
        )
        ptl_run = find_source_ptl_run(args.finetune_root, station, args.seed)
        run_dir = write_run(args, station, direct_run, ptl_run, timestamp)
        manifest.append(
            {
                "station": station,
                "direct_run": str(direct_run.resolve()),
                "source_ptl_run": str(ptl_run.resolve()),
                "output_run": str(run_dir.resolve()),
                "blend_alpha": args.blend_alpha,
            }
        )
        print(f"[{index}/{len(stations)}] built {station}: {run_dir}")

    manifest_path = (
        args.finetune_root
        / "optimization"
        / OUTPUT_PROFILE
        / "mlp"
        / f"build_manifest_seed{args.seed}_{timestamp}.csv"
    )
    pd.DataFrame(manifest).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
