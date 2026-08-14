from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_DIR = (
    REPO_ROOT / "results" / "summary" / "current_all_tested_stations_overall_nse"
)
FIGURE_DIR = REPO_ROOT / "results" / "figures" / "current_all_tested_stations_overall_nse"
PTL_ROOT = (
    REPO_ROOT / "results" / "ptl" / "finetune" / "runs"
    / "train_availability_current_selected_2023_2024"
)
TRANSFORMER_ROOT_TEMPLATE = (
    REPO_ROOT / "results" / "base" / "fair_compare"
    / "basic_transformer_2023_2024_current_selected_train{level}"
)
CLASSIFICATION_PATH = SUMMARY_DIR / "selected_replaced_station_classification.csv"
MANIFEST_PATH = (
    REPO_ROOT / "data"
    / "water_quality_processed_2021_2024_train_availability_current_selected"
    / "train_availability_current_selected_manifest_seed42.csv"
)

AVAILABILITY_LEVELS = [100, 75, 50]
MISSING_BY_LEVEL = {100: 0, 75: 25, 50: 50}
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
ALL_FEATURES = ["CODMn", "DO", "NH4N", "pH"]
MODEL_ORDER = ["PTL", "Transformer"]
MODEL_COLORS = {"PTL": "#2b8cbe", "Transformer": "#d95f02"}
MODEL_MARKERS = {"PTL": "o", "Transformer": "s"}


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


ENGLISH_FONT = choose_font_name(["Times New Roman", "Times"], "DejaVu Serif")
CJK_FONT = choose_font_name(
    ["Songti SC", "PingFang SC", "Arial Unicode MS", "Heiti TC", "SimSun"],
    "DejaVu Sans",
)
ENGLISH_PROP = FontProperties(family=ENGLISH_FONT)
CJK_PROP = FontProperties(family=CJK_FONT)


def configure_plot_style() -> None:
    plt.rcParams["font.family"] = ENGLISH_FONT
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"


def load_stations() -> list[str]:
    stations = pd.read_csv(CLASSIFICATION_PATH, encoding="utf-8-sig")["station"].tolist()
    if not stations:
        raise ValueError(f"No stations found in {CLASSIFICATION_PATH}")
    return stations


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        return pd.DataFrame()
    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")
    manifest = manifest[manifest["resolution"] == "daily"].copy()
    manifest["training_availability_pct"] = (manifest["availability"] * 100).round().astype(int)
    return manifest


def latest_path(paths: list[Path], description: str) -> Path:
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"Missing {description}")
    return paths[-1]


def latest_ptl_metrics(level: int, station: str) -> tuple[Path, Path]:
    parent = PTL_ROOT / f"train{level}"
    metrics_path = latest_path(
        list(parent.glob(f"batch_*/progressive_{station}_seed42_*/stage3_daily/metrics.csv")),
        f"PTL metrics for train{level} {station}",
    )
    return metrics_path, metrics_path.with_name("meta.json")


def latest_transformer_metrics(level: int, station: str) -> tuple[Path, Path]:
    parent = Path(str(TRANSFORMER_ROOT_TEMPLATE).format(level=level))
    metrics_path = latest_path(
        list(parent.glob(f"basic_transformer_{station}_seed42_*/metrics.csv")),
        f"Transformer metrics for train{level} {station}",
    )
    return metrics_path, metrics_path.with_name("meta.json")


def to_float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def read_metrics(metrics_path: Path, meta_path: Path) -> dict[str, object]:
    metrics = pd.read_csv(metrics_path, index_col=0, encoding="utf-8-sig")
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    overall = metrics.loc["__overall__"]
    focus = metrics.loc[FOCUS_FEATURES]
    row: dict[str, object] = {
        "overall_nse": to_float(overall["NSE"]),
        "overall_rmse": to_float(overall["RMSE"]),
        "overall_mae": to_float(overall["MAE"]),
        "focus_mean_nse": to_float(pd.to_numeric(focus["NSE"], errors="coerce").mean()),
        "focus_mean_rmse": to_float(pd.to_numeric(focus["RMSE"], errors="coerce").mean()),
        "focus_mean_mae": to_float(pd.to_numeric(focus["MAE"], errors="coerce").mean()),
        "train_windows": meta.get("train_windows"),
        "val_windows": meta.get("val_windows"),
        "test_windows": meta.get("test_windows"),
        "records": meta.get("records"),
        "invalid_records": meta.get("invalid_records"),
        "best_epoch": meta.get("best_epoch"),
        "best_val_nse": meta.get("best_val_nse"),
        "test_nse_meta": meta.get("test_nse"),
        "metrics_path": str(metrics_path),
        "meta_path": str(meta_path),
    }
    for feature in ALL_FEATURES:
        feature_row = metrics.loc[feature]
        key = feature.lower().replace("nh4n", "nh4n")
        row[f"{key}_nse"] = to_float(feature_row["NSE"])
        row[f"{key}_rmse"] = to_float(feature_row["RMSE"])
        row[f"{key}_mae"] = to_float(feature_row["MAE"])
    return row


def build_long_table(stations: list[str], manifest: pd.DataFrame) -> pd.DataFrame:
    manifest_lookup: dict[tuple[str, int], pd.Series] = {}
    if not manifest.empty:
        for _, row in manifest.iterrows():
            manifest_lookup[(row["station_name"], int(row["training_availability_pct"]))] = row

    rows: list[dict[str, object]] = []
    for station_index, station in enumerate(stations, start=1):
        for level in AVAILABILITY_LEVELS:
            manifest_row = manifest_lookup.get((station, level))
            if manifest_row is None:
                realized_availability = 1.0
                valid_rows_before = np.nan
                kept_rows = np.nan
                masked_rows = np.nan
            else:
                realized_availability = float(manifest_row["realized_valid_train_availability"])
                valid_rows_before = int(manifest_row["valid_train_rows_before_mask"])
                kept_rows = int(manifest_row["kept_valid_train_rows"])
                masked_rows = int(manifest_row["masked_valid_train_rows"])

            for model in MODEL_ORDER:
                if model == "PTL":
                    metrics_path, meta_path = latest_ptl_metrics(level, station)
                else:
                    metrics_path, meta_path = latest_transformer_metrics(level, station)

                metric_row = read_metrics(metrics_path, meta_path)
                rows.append(
                    {
                        "station_name": station,
                        "station_order": station_index,
                        "model": model,
                        "training_availability_pct": level,
                        "training_missing_pct": MISSING_BY_LEVEL[level],
                        "realized_valid_train_availability_pct": realized_availability * 100,
                        "realized_valid_train_missing_pct": (1.0 - realized_availability) * 100,
                        "valid_train_rows_before_mask": valid_rows_before,
                        "kept_valid_train_rows": kept_rows,
                        "masked_valid_train_rows": masked_rows,
                        **metric_row,
                    }
                )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["station_order", "training_missing_pct", "model"],
        key=lambda series: series.map({model: idx for idx, model in enumerate(MODEL_ORDER)})
        if series.name == "model"
        else series,
    ).reset_index(drop=True)


def write_tables(frame: pd.DataFrame) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        SUMMARY_DIR / "current_selected_training_availability_metrics_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    wide = frame.pivot_table(
        index=["station_order", "station_name"],
        columns=["model", "training_availability_pct"],
        values="focus_mean_nse",
        aggfunc="first",
    )
    wide.columns = [f"{model}_focus_nse_train{level}" for model, level in wide.columns]
    wide = wide.reset_index().sort_values("station_order")
    ordered_wide_columns = ["station_order", "station_name"] + [
        f"{model}_focus_nse_train{level}"
        for model in MODEL_ORDER
        for level in AVAILABILITY_LEVELS
    ]
    wide = wide[ordered_wide_columns]
    wide.to_csv(
        SUMMARY_DIR / "current_selected_training_availability_focus_nse_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_means = (
        frame.groupby(["model", "training_availability_pct", "training_missing_pct"], as_index=False)
        .agg(
            mean_overall_nse=("overall_nse", "mean"),
            mean_focus_nse=("focus_mean_nse", "mean"),
            mean_focus_rmse=("focus_mean_rmse", "mean"),
            mean_focus_mae=("focus_mean_mae", "mean"),
            mean_train_windows=("train_windows", "mean"),
            station_count=("station_name", "nunique"),
        )
        .sort_values(
            ["model", "training_missing_pct"],
            key=lambda series: series.map({model: idx for idx, model in enumerate(MODEL_ORDER)})
            if series.name == "model"
            else series,
        )
    )
    group_means.to_csv(
        SUMMARY_DIR / "current_selected_training_availability_group_means.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_metric_grid(frame: pd.DataFrame, metric: str, y_label: str, output_stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stations = (
        frame[["station_name", "station_order"]]
        .drop_duplicates()
        .sort_values("station_order")["station_name"]
        .tolist()
    )
    ncols = 5
    nrows = int(np.ceil(len(stations) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18.2, 9.8), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()

    y_values = frame[metric].astype(float)
    y_min = min(-0.1, float(np.nanmin(y_values)) - 0.05)
    y_max = min(1.0, max(0.9, float(np.nanmax(y_values)) + 0.04))
    if metric == "overall_nse":
        y_min = max(-2.1, float(np.nanmin(y_values)) - 0.08)
        y_max = min(1.0, max(0.9, float(np.nanmax(y_values)) + 0.04))

    for ax, station in zip(axes, stations):
        station_frame = frame[frame["station_name"] == station]
        for model in MODEL_ORDER:
            series = (
                station_frame[station_frame["model"] == model]
                .sort_values("training_missing_pct")
            )
            ax.plot(
                series["training_missing_pct"],
                series[metric],
                marker=MODEL_MARKERS[model],
                markersize=4.8,
                linewidth=1.9,
                color=MODEL_COLORS[model],
                label=model,
                zorder=3,
            )
        ax.set_title(station, fontproperties=CJK_PROP, fontsize=12.5, pad=7)
        ax.set_xticks([0, 25, 50])
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", linestyle="--", linewidth=0.75, alpha=0.42, zorder=0)
        ax.tick_params(axis="both", labelsize=10, labelbottom=True)

    for ax in axes[len(stations):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.942),
        ncol=2,
        frameon=False,
        prop=FontProperties(family=ENGLISH_FONT, size=13),
        handlelength=2.6,
    )
    for text in legend.get_texts():
        text.set_fontproperties(ENGLISH_PROP)

    fig.text(
        0.5,
        0.034,
        "Training Data Missing Ratio (%)",
        ha="center",
        va="center",
        fontproperties=ENGLISH_PROP,
        fontsize=13.2,
    )
    fig.text(
        0.018,
        0.5,
        y_label,
        ha="center",
        va="center",
        rotation="vertical",
        fontproperties=ENGLISH_PROP,
        fontsize=13.2,
    )
    fig.suptitle(
        "PTL vs Transformer Under Training Data Missingness",
        fontproperties=FontProperties(family=ENGLISH_FONT, size=17, weight="bold"),
        y=0.988,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.075,
        top=0.865,
        wspace=0.12,
        hspace=0.55,
    )
    fig.savefig(FIGURE_DIR / f"{output_stem}.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_group_mean(frame: pd.DataFrame) -> None:
    grouped = (
        frame.groupby(["model", "training_missing_pct"], as_index=False)
        .agg(mean_focus_nse=("focus_mean_nse", "mean"))
        .sort_values(["model", "training_missing_pct"])
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for model in MODEL_ORDER:
        series = grouped[grouped["model"] == model]
        ax.plot(
            series["training_missing_pct"],
            series["mean_focus_nse"],
            marker=MODEL_MARKERS[model],
            markersize=5.5,
            linewidth=2.2,
            color=MODEL_COLORS[model],
            label=model,
            zorder=3,
        )
    ax.set_xticks([0, 25, 50])
    ax.set_xlabel("Training Data Missing Ratio (%)", fontproperties=ENGLISH_PROP, fontsize=12.5)
    ax.set_ylabel("Mean Focus NSE (CODMn/DO/pH)", fontproperties=ENGLISH_PROP, fontsize=12.5)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(frameon=False, prop=FontProperties(family=ENGLISH_FONT, size=11.5))
    ax.tick_params(axis="both", labelsize=10.5)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "current_selected_training_missing_focus_nse_group_mean_times.png",
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURE_DIR / "current_selected_training_missing_focus_nse_group_mean_times.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def validate_outputs(frame: pd.DataFrame, stations: list[str]) -> None:
    expected = len(stations) * len(AVAILABILITY_LEVELS) * len(MODEL_ORDER)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(frame)}")
    counts = frame.groupby(["station_name", "model"]).size()
    bad = counts[counts != len(AVAILABILITY_LEVELS)]
    if not bad.empty:
        raise ValueError(f"Incomplete station/model rows:\n{bad}")
    if frame[["overall_nse", "focus_mean_nse"]].isna().any().any():
        raise ValueError("NSE table contains NaN values")


def main() -> None:
    configure_plot_style()
    stations = load_stations()
    frame = build_long_table(stations, load_manifest())
    validate_outputs(frame, stations)
    write_tables(frame)
    plot_metric_grid(
        frame,
        "focus_mean_nse",
        "Focus NSE (CODMn/DO/pH)",
        "current_selected_training_missing_focus_nse_curve_times",
    )
    plot_metric_grid(
        frame,
        "overall_nse",
        "Overall NSE",
        "current_selected_training_missing_overall_nse_curve_times",
    )
    plot_group_mean(frame)

    print(f"English font: {ENGLISH_FONT}")
    print(f"CJK font: {CJK_FONT}")
    print(f"Rows: {len(frame)}")
    print(f"Summary dir: {SUMMARY_DIR}")
    print(f"Figure dir: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
