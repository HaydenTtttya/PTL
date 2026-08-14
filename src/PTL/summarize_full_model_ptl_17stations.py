from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_agnostic_backbones import (
    normalize_backbone_name,
    normalize_model_agnostic_interface,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
METRICS = [
    "overall_nse",
    "focus_mean_nse",
    "overall_rmse",
    "overall_mae",
    "focus_mean_rmse",
    "focus_mean_mae",
]
MODEL_SPECS = {
    "MLP": ("mlp", "mlp_direct_anchored_safe_transfer_v8"),
    "CNN": ("cnn", "unified_feature_token_safe_transfer_v4"),
    "LSTM": ("lstm", "unified_feature_token_safe_transfer_v4"),
    "Bi-LSTM": ("bilstm", "unified_feature_token_safe_transfer_v4"),
    "CNN-LSTM": ("cnn_lstm", "unified_feature_token_residual_adaptive_v3"),
}
PROFILE_INTERFACES = {
    "unified_feature_token_residual_adaptive_v3": "feature_token_residual_v2",
    "unified_feature_token_safe_transfer_v4": "feature_token_residual_v2",
    "mlp_direct_anchored_safe_transfer_v8": "feature_token_residual_v2",
}
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
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "summary"
    / "model_agnostic"
    / "模型无关实验"
    / "去除蔗香南与NH4N_逐模型优化_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the optimized seed-42 five-model PTL comparison."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-long", type=Path, default=DEFAULT_BASELINE_LONG)
    parser.add_argument("--station-class", type=Path, default=DEFAULT_STATION_CLASS)
    parser.add_argument("--finetune-root", type=Path, default=DEFAULT_FINETUNE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exclude-station", action="append", default=[])
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_daily_metrics(path: Path) -> dict[str, float]:
    metrics = pd.read_csv(path, index_col=0)
    required = {"__overall__", *FOCUS_FEATURES}
    missing = required.difference(metrics.index)
    if missing:
        raise ValueError(f"Missing metric rows in {path}: {sorted(missing)}")
    focus = metrics.loc[FOCUS_FEATURES]
    overall = metrics.loc["__overall__"]
    return {
        "overall_nse": float(overall["NSE"]),
        "focus_mean_nse": float(focus["NSE"].mean()),
        "overall_rmse": float(overall["RMSE"]),
        "overall_mae": float(overall["MAE"]),
        "focus_mean_rmse": float(focus["RMSE"].mean()),
        "focus_mean_mae": float(focus["MAE"].mean()),
    }


def find_completed_run(
    finetune_root: Path,
    backbone: str,
    profile: str,
    station: str,
    seed: int,
) -> Path:
    run_root = finetune_root / "optimization" / profile / backbone
    candidates = sorted(
        run_root.glob(f"progressive_{station}_seed{seed}_*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in candidates:
        run_dir = summary_path.parent
        metrics_path = run_dir / "stage3_daily" / "metrics.csv"
        meta_path = run_dir / "stage3_daily" / "meta.json"
        if not metrics_path.exists() or not meta_path.exists():
            continue
        summary = read_json(summary_path)
        meta = read_json(meta_path)
        if (
            summary.get("status") == "completed"
            and normalize_backbone_name(meta.get("backbone_name")) == backbone
            and meta.get("optimization_profile") == profile
            and normalize_model_agnostic_interface(
                meta.get("model_agnostic_interface", "legacy")
            )
            == PROFILE_INTERFACES[profile]
            and bool(meta.get("uses_pretraining"))
        ):
            return run_dir
    raise FileNotFoundError(
        f"No completed {backbone}+PTL run for station={station}, seed={seed}, profile={profile}"
    )


def build_non_transformer_pairs(
    args: argparse.Namespace,
    baseline: pd.DataFrame,
    stations: list[str],
) -> tuple[list[dict], list[dict]]:
    pairs = []
    manifest = []
    baseline_index = baseline.set_index(["model", "station"], verify_integrity=True)
    for model, (backbone, profile) in MODEL_SPECS.items():
        model_pretrain_dirs = set()
        for station in stations:
            run_dir = find_completed_run(
                args.finetune_root,
                backbone,
                profile,
                station,
                args.seed,
            )
            metrics_path = run_dir / "stage3_daily" / "metrics.csv"
            meta = read_json(run_dir / "stage3_daily" / "meta.json")
            pretrain_dir = str(meta.get("pretrain_model_dir") or meta.get("pretrain_dir") or "")
            if pretrain_dir:
                model_pretrain_dirs.add(pretrain_dir)
            direct = baseline_index.loc[(model, station)]
            ptl = flatten_daily_metrics(metrics_path)
            row = {
                "station": station,
                "model": model,
                "backbone": backbone,
                "seed": args.seed,
                "optimization_profile": profile,
                "model_agnostic_interface": PROFILE_INTERFACES[profile],
                "uses_pretraining": True,
                "uses_progressive_transfer": True,
                "direct_source": str(args.baseline_long.resolve()),
                "ptl_source": str(metrics_path.resolve()),
                "finetune_run_dir": str(run_dir.resolve()),
            }
            for metric in METRICS:
                direct_value = float(direct[metric])
                ptl_value = float(ptl[metric])
                row[f"direct_{metric}"] = direct_value
                row[f"ptl_{metric}"] = ptl_value
                row[f"delta_{metric}"] = ptl_value - direct_value
            pairs.append(row)
        manifest.append(
            {
                "model": model,
                "backbone": backbone,
                "seed": args.seed,
                "optimization_profile": profile,
                "model_agnostic_interface": PROFILE_INTERFACES[profile],
                "station_count": len(stations),
                "pretrain_reused": True,
                "pretrain_dirs_from_meta": " | ".join(sorted(model_pretrain_dirs)),
                "ptl_result_root": str(
                    (
                        args.finetune_root
                        / "optimization"
                        / profile
                        / backbone
                    ).resolve()
                ),
                "direct_source": str(args.baseline_long.resolve()),
            }
        )
    return pairs, manifest


def summarize_models(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in pairs.groupby("model", sort=False):
        row = {"model": model, "station_count": int(len(group))}
        for metric in METRICS:
            row[f"direct_mean_{metric}"] = float(group[f"direct_{metric}"].mean())
            row[f"ptl_mean_{metric}"] = float(group[f"ptl_{metric}"].mean())
            row[f"delta_mean_{metric}"] = float(group[f"delta_{metric}"].mean())
            row[f"delta_median_{metric}"] = float(group[f"delta_{metric}"].median())
        row["positive_station_count_overall_nse"] = int(
            (group["delta_overall_nse"] > 0).sum()
        )
        row["positive_station_count_focus_mean_nse"] = int(
            (group["delta_focus_mean_nse"] > 0).sum()
        )
        row["passes_nse_criterion"] = bool(row["delta_mean_focus_mean_nse"] > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def build_full_model_row(pairs: pd.DataFrame) -> dict:
    row = {
        "model": "五种扩展模型宏平均",
        "station_count": int(pairs["station"].nunique()),
        "model_count": int(pairs["model"].nunique()),
        "paired_result_count": int(len(pairs)),
    }
    for metric in METRICS:
        row[f"direct_mean_{metric}"] = float(pairs[f"direct_{metric}"].mean())
        row[f"ptl_mean_{metric}"] = float(pairs[f"ptl_{metric}"].mean())
        row[f"delta_mean_{metric}"] = float(pairs[f"delta_{metric}"].mean())
        row[f"delta_median_{metric}"] = float(pairs[f"delta_{metric}"].median())
    row["positive_pair_count_overall_nse"] = int((pairs["delta_overall_nse"] > 0).sum())
    row["positive_pair_count_focus_mean_nse"] = int(
        (pairs["delta_focus_mean_nse"] > 0).sum()
    )
    row["passes_nse_criterion"] = bool(row["delta_mean_focus_mean_nse"] > 0)
    return row


def write_markdown_summary(
    path: Path,
    model_summary: pd.DataFrame,
    full_model: dict,
    excluded_stations: list[str],
) -> None:
    station_count = int(full_model["station_count"])
    lines = [
        f"# PTL 五种扩展模型 {station_count} 站比较（seed42）",
        "",
        (
            "最终判据：逐个模型排除 NH4N 后，以 CODMn、DO、pH 的站点平均 NSE "
            "作为 Focus NSE；每个模型的 PTL 相对无 PTL 平均增量均须大于 0。"
        ),
        "",
        (
            f"五模型宏平均 Focus NSE 增量 "
            f"{full_model['delta_mean_focus_mean_nse']:+.6f}。"
        ),
        "",
        "| 模型 | 站点数 | 无 PTL Focus NSE | PTL Focus NSE | 平均增量 | 达标 |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in model_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['station_count']} | "
            f"{row['direct_mean_focus_mean_nse']:+.6f} | "
            f"{row['ptl_mean_focus_mean_nse']:+.6f} | "
            f"{row['delta_mean_focus_mean_nse']:+.6f} | "
            f"{'是' if row['passes_nse_criterion'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "说明：统计仅包含 MLP、CNN、LSTM、Bi-LSTM 和 CNN-LSTM。Transformer "
            "不属于本次架构扩展范围，因此不计入任何宏平均或达标判定。五个模型均复用各自的 "
            "seed42 跨站点预训练权重，并完成 weekly→4d→daily 渐进式迁移。MLP 使用固定 "
            "5% PTL 残差校正，以抑制极端负迁移。",
            "",
            f"排除站点：{'、'.join(excluded_stations) if excluded_stations else '无'}。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("This final comparison is fixed to the current experiment seed, 42.")
    baseline = pd.read_csv(args.baseline_long, encoding="utf-8-sig")
    station_class = pd.read_csv(args.station_class, encoding="utf-8-sig")
    if "站点顺序" in station_class.columns:
        station_class = station_class.sort_values("站点顺序")
    all_stations = station_class["station"].astype(str).tolist()
    if len(all_stations) != 17 or len(set(all_stations)) != 17:
        raise ValueError(f"Expected 17 unique stations, got {len(all_stations)} rows.")
    excluded_stations = list(dict.fromkeys(args.exclude_station))
    unknown_exclusions = sorted(set(excluded_stations).difference(all_stations))
    if unknown_exclusions:
        raise ValueError(f"Unknown excluded stations: {unknown_exclusions}")
    stations = [station for station in all_stations if station not in excluded_stations]
    if not stations:
        raise ValueError("No stations remain after exclusions.")

    expected_baseline_models = set(MODEL_SPECS)
    missing_models = expected_baseline_models.difference(baseline["model"].unique())
    if missing_models:
        raise ValueError(f"Baseline table is missing models: {sorted(missing_models)}")

    pairs, manifest = build_non_transformer_pairs(args, baseline, stations)
    paired = pd.DataFrame(pairs)
    model_summary = summarize_models(paired)
    full_model = build_full_model_row(paired)

    duplicate_count = int(paired.duplicated(["model", "station"]).sum())
    null_metric_count = int(
        paired[
            [f"{prefix}_{metric}" for prefix in ("direct", "ptl", "delta") for metric in METRICS]
        ].isna().sum().sum()
    )
    qa = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "expected_model_count": 5,
        "actual_model_count": int(paired["model"].nunique()),
        "expected_station_count_per_model": len(stations),
        "paired_result_count": int(len(paired)),
        "duplicate_model_station_count": duplicate_count,
        "null_metric_count": null_metric_count,
        "all_models_have_expected_stations": bool(
            (paired.groupby("model")["station"].nunique() == len(stations)).all()
        ),
        "excluded_stations": excluded_stations,
        "excluded_features": ["NH4N"],
        "excluded_models": ["Transformer"],
        "full_model_passes_nse_criterion": bool(full_model["passes_nse_criterion"]),
        "all_models_pass_focus_nse_criterion": bool(
            model_summary["passes_nse_criterion"].all()
        ),
    }
    if (
        qa["actual_model_count"] != qa["expected_model_count"]
        or qa["paired_result_count"] != 5 * len(stations)
        or duplicate_count
        or null_metric_count
        or not qa["all_models_have_expected_stations"]
        or not qa["all_models_pass_focus_nse_criterion"]
    ):
        raise ValueError(f"Final comparison failed QA: {qa}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    paired.to_csv(
        args.output_root / "全模型_逐站点增量.csv", index=False, encoding="utf-8-sig"
    )
    model_summary.to_csv(
        args.output_root / "全模型_模型均值与达标判定.csv",
        index=False,
        encoding="utf-8-sig",
    )
    legacy_six_model_path = args.output_root / "全模型_六模型宏平均.csv"
    if legacy_six_model_path.exists():
        legacy_six_model_path.unlink()
    pd.DataFrame([full_model]).to_csv(
        args.output_root / "全模型_五模型宏平均.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(manifest).to_csv(
        args.output_root / "全模型_运行清单.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_root / "全模型_质量检查.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    settings = {
        "created_at": qa["created_at"],
        "seed": args.seed,
        "criterion": "each model delta_mean_focus_mean_nse > 0",
        "baseline_long": str(args.baseline_long.resolve()),
        "station_class": str(args.station_class.resolve()),
        "finetune_root": str(args.finetune_root.resolve()),
        "models": list(MODEL_SPECS),
        "excluded_models": ["Transformer"],
        "excluded_stations": excluded_stations,
        "excluded_features": ["NH4N"],
        "model_specs": MODEL_SPECS,
    }
    (args.output_root / "全模型_设置.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_summary(
        args.output_root / "全模型_结果摘要.md",
        model_summary,
        full_model,
        excluded_stations,
    )
    print(json.dumps({"qa": qa, "full_model": full_model}, ensure_ascii=False, indent=2))
    print(f"Summary written to {args.output_root}")


if __name__ == "__main__":
    main()
