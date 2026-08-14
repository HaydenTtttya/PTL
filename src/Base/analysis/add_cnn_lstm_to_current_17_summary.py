from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_DIR = (
    REPO_ROOT
    / "results"
    / "summary"
    / "current_all_tested_stations_overall_nse"
    / "均衡十五站方案_新增两站"
)
RUN_ROOT = REPO_ROOT / "results" / "base" / "fair_compare" / "current_17_cnn_lstm_strict_2023_2024" / "cnn_lstm"
PLOT_DATA_DIR = SUMMARY_DIR / "作图数据"

MODEL_NAME = "CNN-LSTM"
MODEL_ORDER = ["MLP", "CNN", "LSTM", "Bi-LSTM", "CNN-LSTM", "Transformer", "PTL"]
FEATURES = ["CODMn", "DO", "NH4N", "pH"]
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
INDICATORS = ["Overall", "Focus", *FEATURES]


def latest_run(station: str) -> Path:
    candidates = sorted(
        RUN_ROOT.glob(f"cnn_lstm_{station}_seed*_*/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Missing CNN-LSTM run for station: {station}")
    return candidates[0].parent


def read_metrics(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "metrics.csv", index_col=0)


def read_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))


def summarize_metrics(metrics: pd.DataFrame) -> dict[str, float]:
    focus = metrics.loc[FOCUS_FEATURES]
    row = {
        "overall_nse": float(metrics.loc["__overall__", "NSE"]),
        "overall_rmse": float(metrics.loc["__overall__", "RMSE"]),
        "overall_mae": float(metrics.loc["__overall__", "MAE"]),
        "focus_mean_nse": float(focus["NSE"].mean()),
        "focus_mean_rmse": float(focus["RMSE"].mean()),
        "focus_mean_mae": float(focus["MAE"].mean()),
    }
    for feature in FEATURES:
        row[f"{feature}_nse"] = float(metrics.loc[feature, "NSE"])
        row[f"{feature}_rmse"] = float(metrics.loc[feature, "RMSE"])
        row[f"{feature}_mae"] = float(metrics.loc[feature, "MAE"])
    return row


def backup_outputs() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = SUMMARY_DIR / "_backup_before_cnn_lstm" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "模型对比长表.csv",
        "Overall_NSE模型对比.csv",
        "Focus_NSE模型对比.csv",
        "分组均值_Overall_NSE.csv",
        "分组均值_Focus_NSE.csv",
        "各指标_站点_模型_NSE长表.csv",
        "上游_各指标模型对比.csv",
        "中游_各指标模型对比.csv",
        "下游_各指标模型对比.csv",
    ]
    plot_files = [
        "模型对比长表_作图用.csv",
        "Overall_NSE模型对比_宽表_作图用.csv",
        "Focus_NSE模型对比_宽表_作图用.csv",
        "分组均值_Overall_NSE_作图用.csv",
        "分组均值_Focus_NSE_作图用.csv",
        "各指标_站点_模型_NSE长表_作图用.csv",
        "上游_各指标_站点_模型_NSE_作图用.csv",
        "中游_各指标_站点_模型_NSE_作图用.csv",
        "下游_各指标_站点_模型_NSE_作图用.csv",
        "箱型图统计_六模型_Overall_NSE.csv",
        "箱型图统计_六模型_Focus_NSE.csv",
    ]
    for name in files:
        src = SUMMARY_DIR / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
    plot_backup = backup_dir / "作图数据"
    plot_backup.mkdir(exist_ok=True)
    for name in plot_files:
        src = PLOT_DATA_DIR / name
        if src.exists():
            shutil.copy2(src, plot_backup / name)
    return backup_dir


def build_cnn_lstm_rows(base_long: pd.DataFrame, station_class: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audit_rows = []
    for _, station_row in station_class.sort_values("站点顺序").iterrows():
        station = station_row["station"]
        run_dir = latest_run(station)
        metrics = read_metrics(run_dir)
        meta = read_meta(run_dir)
        metric_summary = summarize_metrics(metrics)
        template = base_long[base_long["station"] == station].iloc[0].to_dict()
        row = {column: template.get(column) for column in base_long.columns}
        row.update(
            {
                "model": MODEL_NAME,
                "overall_nse": metric_summary["overall_nse"],
                "focus_mean_nse": metric_summary["focus_mean_nse"],
                "overall_rmse": metric_summary["overall_rmse"],
                "overall_mae": metric_summary["overall_mae"],
                "focus_mean_rmse": metric_summary["focus_mean_rmse"],
                "focus_mean_mae": metric_summary["focus_mean_mae"],
                "train_windows": int(meta["train_windows"]),
                "val_windows": int(meta["val_windows"]),
                "test_windows": int(meta["test_windows"]),
                "soft_gap_mode": "off",
                "invalid_window_policy": meta.get("invalid_window_policy"),
                "uses_pretraining": False,
                "uses_progressive_transfer": False,
            }
        )
        rows.append(row)
        audit_rows.append(
            {
                "station": station,
                "model": MODEL_NAME,
                "run_dir": str(run_dir),
                "metrics_path": str(run_dir / "metrics.csv"),
                "parameter_count": int(meta["parameter_count"]),
                "time_start": meta.get("time_start"),
                "time_end": meta.get("time_end"),
                "soft_gap_max_steps": int(meta.get("soft_gap_max_steps", 0)),
                "invalid_window_policy": meta.get("invalid_window_policy"),
                "train_windows": int(meta["train_windows"]),
                "val_windows": int(meta["val_windows"]),
                "test_windows": int(meta["test_windows"]),
                **metric_summary,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def sort_models(frame: pd.DataFrame) -> pd.DataFrame:
    station_order = frame["station"].drop_duplicates().reset_index(drop=True)
    station_order_map = {station: index for index, station in enumerate(station_order)}
    model_order_map = {model: index for index, model in enumerate(MODEL_ORDER)}
    return (
        frame.assign(
            _station_order=lambda df: df["station"].map(station_order_map),
            _model_order=lambda df: df["model"].map(model_order_map),
        )
        .sort_values(["_station_order", "_model_order"])
        .drop(columns=["_station_order", "_model_order"])
        .reset_index(drop=True)
    )


def build_wide(frame: pd.DataFrame, value_col: str, stations: list[str]) -> pd.DataFrame:
    wide = (
        frame.pivot_table(index="station", columns="model", values=value_col, aggfunc="first")
        .reindex(stations)
        .reindex(columns=MODEL_ORDER)
        .reset_index()
    )
    return wide


def build_group_means(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    means = (
        frame.groupby(["river_reach", "river_type", "model"], as_index=False)[value_col]
        .mean()
        .pivot_table(index=["river_reach", "river_type"], columns="model", values=value_col, aggfunc="first")
        .reindex(columns=MODEL_ORDER)
        .reset_index()
    )
    counts = frame.groupby(["river_reach", "river_type"])["station"].nunique().reset_index(name="station_count")
    return means.merge(counts, on=["river_reach", "river_type"], how="left")


def build_indicator_long(previous_indicator: pd.DataFrame, station_class: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    class_cols = [
        "station",
        "river_reach",
        "river_type",
        "reach_type_note",
        "verified_waterbody",
    ]
    class_frame = station_class[class_cols + ["站点顺序"]].copy()

    base_indicator = previous_indicator[previous_indicator["model"] != MODEL_NAME].copy()
    base_indicator = base_indicator.drop(columns=["站点顺序"], errors="ignore")
    base_indicator = base_indicator.merge(class_frame[["station", "站点顺序"]], on="station", how="left")

    for _, station_row in class_frame.sort_values("站点顺序").iterrows():
        station_audit = audit[audit["station"] == station_row["station"]].iloc[0]
        values = {
            "Overall": station_audit["overall_nse"],
            "Focus": station_audit["focus_mean_nse"],
            **{feature: station_audit[f"{feature}_nse"] for feature in FEATURES},
        }
        for indicator in INDICATORS:
            rows.append(
                {
                    "station": station_row["station"],
                    "river_reach": station_row["river_reach"],
                    "river_type": station_row["river_type"],
                    "reach_type_note": station_row["reach_type_note"],
                    "verified_waterbody": station_row["verified_waterbody"],
                    "model": MODEL_NAME,
                    "indicator": indicator,
                    "nse": values[indicator],
                    "train_windows": station_audit["train_windows"],
                    "val_windows": station_audit["val_windows"],
                    "test_windows": station_audit["test_windows"],
                    "站点顺序": station_row["站点顺序"],
                }
            )
    indicator_frame = pd.concat([base_indicator, pd.DataFrame(rows)], ignore_index=True)
    model_order_map = {model: index for index, model in enumerate(MODEL_ORDER)}
    indicator_order_map = {indicator: index for index, indicator in enumerate(INDICATORS)}
    return (
        indicator_frame.assign(
            _model_order=lambda df: df["model"].map(model_order_map),
            _indicator_order=lambda df: df["indicator"].map(indicator_order_map),
        )
        .sort_values(["站点顺序", "_indicator_order", "_model_order"])
        .drop(columns=["_model_order", "_indicator_order"])
        .reset_index(drop=True)
    )


def box_stats(values: pd.Series) -> dict:
    clean = values.dropna().astype(float)
    return {
        "样本数": int(clean.shape[0]),
        "最小值": float(clean.min()) if len(clean) else np.nan,
        "第一四分位数": float(clean.quantile(0.25)) if len(clean) else np.nan,
        "中位数": float(clean.median()) if len(clean) else np.nan,
        "第三四分位数": float(clean.quantile(0.75)) if len(clean) else np.nan,
        "最大值": float(clean.max()) if len(clean) else np.nan,
        "均值": float(clean.mean()) if len(clean) else np.nan,
        "低于-1数量": int((clean < -1).sum()),
    }


def build_model_box_stats(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    scenario_parts = [("全部站点", frame)]
    for reach in ["上游", "中游", "下游"]:
        scenario_parts.append((reach, frame[frame["river_reach"] == reach]))
    type_labels = {
        "干流/主要水道": "干流主要水道",
        "支流/区域河流": "支流区域河流",
    }
    for river_type, label in type_labels.items():
        scenario_parts.append((label, frame[frame["river_type"] == river_type]))
    rows = []
    for scenario, subset in scenario_parts:
        for model in MODEL_ORDER:
            model_values = subset.loc[subset["model"] == model, value_col]
            rows.append({"场景": scenario, "模型": model, **box_stats(model_values)})
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    backup_dir = backup_outputs()
    base_long = pd.read_csv(SUMMARY_DIR / "模型对比长表.csv")
    previous_indicator = pd.read_csv(SUMMARY_DIR / "各指标_站点_模型_NSE长表.csv")
    station_class = pd.read_csv(SUMMARY_DIR / "站点分类.csv")
    stations = station_class.sort_values("站点顺序")["station"].tolist()
    cnn_rows, audit = build_cnn_lstm_rows(base_long, station_class)

    updated_long = pd.concat([base_long[base_long["model"] != MODEL_NAME], cnn_rows], ignore_index=True)
    updated_long = sort_models(updated_long)
    overall_wide = build_wide(updated_long, "overall_nse", stations)
    focus_wide = build_wide(updated_long, "focus_mean_nse", stations)
    overall_group = build_group_means(updated_long, "overall_nse")
    focus_group = build_group_means(updated_long, "focus_mean_nse")
    indicator_long = build_indicator_long(previous_indicator, station_class, audit)

    write_csv(updated_long, SUMMARY_DIR / "模型对比长表.csv")
    write_csv(overall_wide, SUMMARY_DIR / "Overall_NSE模型对比.csv")
    write_csv(focus_wide, SUMMARY_DIR / "Focus_NSE模型对比.csv")
    write_csv(overall_group, SUMMARY_DIR / "分组均值_Overall_NSE.csv")
    write_csv(focus_group, SUMMARY_DIR / "分组均值_Focus_NSE.csv")
    write_csv(indicator_long.drop(columns=["站点顺序"]), SUMMARY_DIR / "各指标_站点_模型_NSE长表.csv")
    write_csv(audit, SUMMARY_DIR / "CNN_LSTM补跑_17站模型清单.csv")

    for reach in ["上游", "中游", "下游"]:
        write_csv(
            indicator_long[indicator_long["river_reach"] == reach],
            SUMMARY_DIR / f"{reach}_各指标模型对比.csv",
        )

    write_csv(station_class, PLOT_DATA_DIR / "站点分类_作图用.csv")
    write_csv(updated_long, PLOT_DATA_DIR / "模型对比长表_作图用.csv")
    write_csv(overall_wide, PLOT_DATA_DIR / "Overall_NSE模型对比_宽表_作图用.csv")
    write_csv(focus_wide, PLOT_DATA_DIR / "Focus_NSE模型对比_宽表_作图用.csv")
    write_csv(overall_group, PLOT_DATA_DIR / "分组均值_Overall_NSE_作图用.csv")
    write_csv(focus_group, PLOT_DATA_DIR / "分组均值_Focus_NSE_作图用.csv")
    write_csv(indicator_long, PLOT_DATA_DIR / "各指标_站点_模型_NSE长表_作图用.csv")
    for reach in ["上游", "中游", "下游"]:
        write_csv(
            indicator_long[indicator_long["river_reach"] == reach],
            PLOT_DATA_DIR / f"{reach}_各指标_站点_模型_NSE_作图用.csv",
        )

    overall_box = build_model_box_stats(updated_long, "overall_nse")
    focus_box = build_model_box_stats(updated_long, "focus_mean_nse")
    write_csv(overall_box, PLOT_DATA_DIR / "箱型图统计_七模型_Overall_NSE.csv")
    write_csv(focus_box, PLOT_DATA_DIR / "箱型图统计_七模型_Focus_NSE.csv")
    write_csv(overall_box, PLOT_DATA_DIR / "箱型图统计_六模型_Overall_NSE.csv")
    write_csv(focus_box, PLOT_DATA_DIR / "箱型图统计_六模型_Focus_NSE.csv")

    print(f"backup_dir={backup_dir}")
    print(f"updated_long_rows={len(updated_long)}")
    print(f"indicator_rows={len(indicator_long)}")
    print(f"cnn_lstm_runs={len(audit)}")
    print(audit[["station", "overall_nse", "focus_mean_nse", "soft_gap_max_steps", "invalid_window_policy"]].to_string(index=False))


if __name__ == "__main__":
    main()
