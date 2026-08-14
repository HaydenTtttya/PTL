#!/usr/bin/env python3
"""Build the six-group Focus RMSE/MAE comparison figure."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKBOOK = (
    REPO_ROOT
    / "outputs"
    / "019ffaca-6be2-7310-b25f-6f7e2110ed5f"
    / "PTL五模型_RMSE_MAE作图数据_seed42.xlsx"
)
SUMMARY_DIR = (
    REPO_ROOT
    / "results"
    / "summary"
    / "current_all_tested_stations_overall_nse"
    / "均衡十五站方案_新增两站"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "figures"
    / "current_all_tested_stations_overall_nse"
    / "均衡十五站方案_新增两站"
    / "paper_figures_en_journal"
)
STYLE_SCRIPT = Path(__file__).with_name("make_paper_english_figures_17stations.py")

BACKBONE_ORDER = ["MLP", "CNN", "LSTM", "Bi-LSTM", "CNN-LSTM", "Transformer"]
CONDITION_ORDER = ["Without PTL", "With PTL"]
STEM = "fig8_six_model_rmse_mae_ptl_comparison_en"


def load_figure_style():
    spec = importlib.util.spec_from_file_location("paper_figure_style", STYLE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load figure style from {STYLE_SCRIPT}")
    spec.loader.exec_module(module)
    return module


def sem(series: pd.Series) -> float:
    return float(series.astype(float).sem(ddof=1))


def build_plot_data(workbook_path: Path) -> pd.DataFrame:
    station_df = pd.read_excel(workbook_path, sheet_name="源数据_逐站点")
    summary_df = pd.read_excel(workbook_path, sheet_name="模型均值_宽表")

    rows: list[dict[str, object]] = []
    first_five = BACKBONE_ORDER[:-1]
    for model in first_five:
        sub = station_df[station_df["模型"] == model].copy()
        if sub["站点"].nunique() != 16:
            raise ValueError(f"{model} does not contain exactly 16 stations")
        for condition, prefix in (("Without PTL", "无PTL"), ("With PTL", "PTL")):
            for metric in ("RMSE", "MAE"):
                values = sub[f"{prefix}_Focus_{metric}"].astype(float)
                rows.append(
                    {
                        "backbone": model,
                        "condition": condition,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "sem": sem(values),
                        "station_count": int(values.count()),
                    }
                )

    comparison_df = pd.read_csv(SUMMARY_DIR / "模型对比长表.csv", encoding="utf-8-sig")
    comparison_df = comparison_df[
        comparison_df["model"].isin(["Transformer", "PTL"])
        & (comparison_df["station"] != "蔗香南")
    ].copy()
    transformer_stations = set(
        comparison_df.loc[comparison_df["model"] == "Transformer", "station"]
    )
    ptl_stations = set(comparison_df.loc[comparison_df["model"] == "PTL", "station"])
    workbook_stations = set(station_df["站点"])
    if transformer_stations != ptl_stations or transformer_stations != workbook_stations:
        raise ValueError("Transformer/PTL stations do not match the workbook's 16-station set")

    for condition, model in (("Without PTL", "Transformer"), ("With PTL", "PTL")):
        sub = comparison_df[comparison_df["model"] == model]
        for metric, column in (("RMSE", "focus_mean_rmse"), ("MAE", "focus_mean_mae")):
            values = sub[column].astype(float)
            rows.append(
                {
                    "backbone": "Transformer",
                    "condition": condition,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sem": sem(values),
                    "station_count": int(values.count()),
                }
            )

    plot_df = pd.DataFrame(rows)
    expected_rows = len(BACKBONE_ORDER) * len(CONDITION_ORDER) * 2
    if len(plot_df) != expected_rows or set(plot_df["station_count"]) != {16}:
        raise ValueError("Incomplete six-group RMSE/MAE comparison data")

    for model in first_five:
        summary = summary_df[summary_df["模型"] == model].iloc[0]
        for condition, prefix in (("Without PTL", "无PTL"), ("With PTL", "PTL")):
            for metric in ("RMSE", "MAE"):
                actual = plot_df.loc[
                    (plot_df["backbone"] == model)
                    & (plot_df["condition"] == condition)
                    & (plot_df["metric"] == metric),
                    "mean",
                ].iloc[0]
                expected = float(summary[f"{prefix}_{metric}"])
                if not np.isclose(actual, expected, atol=1e-12):
                    raise ValueError(f"{model} {condition} {metric} does not match workbook summary")

    return plot_df


def add_value_labels(
    ax,
    bars,
    errors: np.ndarray,
    pad: float,
    condition: str,
) -> None:
    vertical_stagger = pad * 3.0 if condition == "Without PTL" else 0.0
    for model, bar, error in zip(BACKBONE_ORDER, bars, errors):
        model_stagger = (
            pad * 2.0
            if model == "CNN-LSTM" and condition == "Without PTL"
            else 0.0
        )
        label = ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + error + pad + vertical_stagger + model_stagger,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#000000",
            clip_on=False,
            zorder=5,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=3.0, foreground="#FFFFFF")]
        )


def make_figure(plot_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    style = load_figure_style()
    style.configure_style()

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.79, bottom=0.18, wspace=0.18)

    x = np.arange(len(BACKBONE_ORDER), dtype=float)
    width = 0.34
    offsets = {"Without PTL": -width / 2, "With PTL": width / 2}
    faces = {
        "Without PTL": "#DEE9FE",
        "With PTL": "#7F9EE4",
    }
    edges = {
        "Without PTL": "#7389B5",
        "With PTL": "#2E4780",
    }

    for ax, metric, panel in zip(axes, ("RMSE", "MAE"), ("a", "b")):
        metric_df = plot_df[plot_df["metric"] == metric]
        panel_max = 0.0
        for condition in CONDITION_ORDER:
            values = np.array(
                [
                    metric_df.loc[
                        (metric_df["backbone"] == model)
                        & (metric_df["condition"] == condition),
                        "mean",
                    ].iloc[0]
                    for model in BACKBONE_ORDER
                ],
                dtype=float,
            )
            errors = np.array(
                [
                    metric_df.loc[
                        (metric_df["backbone"] == model)
                        & (metric_df["condition"] == condition),
                        "sem",
                    ].iloc[0]
                    for model in BACKBONE_ORDER
                ],
                dtype=float,
            )
            bars = ax.bar(
                x + offsets[condition],
                values,
                width=width,
                color=faces[condition],
                edgecolor=edges[condition],
                linewidth=1.2,
                yerr=errors,
                capsize=4,
                error_kw={
                    "ecolor": edges[condition],
                    "elinewidth": 1.0,
                    "capthick": 1.0,
                },
                label=condition,
                zorder=3,
            )
            panel_max = max(panel_max, float(np.max(values + errors)))
            add_value_labels(
                ax,
                bars,
                errors,
                pad=0.012 if metric == "RMSE" else 0.009,
                condition=condition,
            )

        ax.set_title(metric, loc="left", pad=10, fontsize=12)
        ax.set_ylabel(f"Mean {metric}", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_xticks(x, BACKBONE_ORDER, rotation=20, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, panel_max * 1.22)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        style.clean_axes(ax, grid_axis=None)
        style.emphasize_axes(ax, labelsize=12)
        style.panel_label(ax, panel, parentheses=True, fontsize=12, x=-0.10, y=1.03)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        frameon=True,
        fancybox=False,
        facecolor="#FFFFFF",
        edgecolor="#000000",
        framealpha=1.0,
        fontsize=12,
        columnspacing=1.4,
        handletextpad=0.55,
        borderpad=0.5,
    )
    legend.get_frame().set_linewidth(0.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("pdf", "png", "svg"):
        output = output_dir / f"{STEM}.{extension}"
        fig.savefig(output, dpi=450, bbox_inches="tight")
        outputs.append(output)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_df = build_plot_data(args.workbook)
    for output in make_figure(plot_df, args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
