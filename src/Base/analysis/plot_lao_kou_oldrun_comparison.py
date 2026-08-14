from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "laokou_oldrun_comparison"

FEATURES = ["CODMn", "DO", "pH"]
METRICS = ["MAE", "RMSE", "NSE", "MAPE"]

RUNS = {
    "MLP": REPO_ROOT
    / "results/base/fair_compare/mlp_2023_2024_oldrun_range/mlp_老口_seed42_20260425_133834/metrics.csv",
    "CNN": REPO_ROOT
    / "results/base/fair_compare/cnn_2023_2024_oldrun_range/cnn_老口_seed42_20260425_133859/metrics.csv",
    "Transformer": REPO_ROOT
    / "results/base/fair_compare/basic_transformer_2023_2024_oldrun_range"
    / "basic_transformer_老口_seed42_20260425_140528/metrics.csv",
    "PTL": REPO_ROOT
    / "results/ptl/finetune/runs/batch_pearl_other_core3_progressive_v2pretrain_v2_2021_2024_20260409_145922"
    / "progressive_老口_seed42_20260409_145922/stage3_daily/metrics.csv",
}

COLORS = {
    "MLP": "#d95f02",
    "CNN": "#1b9e77",
    "Transformer": "#7570b3",
    "PTL": "#2b8cbe",
}


def load_metrics(run_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    loaded = {}
    missing = []
    for model_name, path in run_paths.items():
        if not path.exists():
            missing.append(str(path))
            continue
        loaded[model_name] = pd.read_csv(path, index_col=0)
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"缺少以下 metrics.csv:\n{missing_text}")
    return loaded


def build_summary_table(model_metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_name, metrics_frame in model_metrics.items():
        for feature_name in FEATURES:
            for metric_name in METRICS:
                rows.append(
                    {
                        "model": model_name.replace("\n", " "),
                        "feature": feature_name,
                        "metric": "MAPE (%)" if metric_name == "MAPE" else metric_name,
                        "value": float(metrics_frame.loc[feature_name, metric_name]),
                    }
                )
    return pd.DataFrame(rows)


def configure_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.unicode_minus"] = False


def plot_grouped_metrics(
    model_metrics: dict[str, pd.DataFrame],
    title: str,
    output_stem: Path,
    figsize: tuple[float, float],
):
    model_names = list(model_metrics.keys())
    x = np.arange(len(FEATURES), dtype=float)
    width = min(0.75 / len(model_names), 0.16)
    offsets = (np.arange(len(model_names)) - (len(model_names) - 1) / 2.0) * width

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985)

    for metric_index, metric_name in enumerate(METRICS):
        ax = axes[metric_index // 2, metric_index % 2]

        for model_index, model_name in enumerate(model_names):
            values = [
                float(model_metrics[model_name].loc[feature_name, metric_name])
                for feature_name in FEATURES
            ]
            ax.bar(
                x + offsets[model_index],
                values,
                width=width,
                label=model_name,
                color=COLORS.get(model_name, "#4c78a8"),
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )

        display_metric_name = "MAPE (%)" if metric_name == "MAPE" else metric_name
        ax.set_title(display_metric_name, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(FEATURES, fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.55, zorder=0)

        if metric_name == "NSE":
            ax.set_ylim(0, 1.05)
        else:
            all_values = [
                float(model_metrics[model_name].loc[feature_name, metric_name])
                for model_name in model_names
                for feature_name in FEATURES
            ]
            upper = max(all_values) * 1.15 if all_values else 1.0
            ax.set_ylim(0, upper)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(len(model_names), 5),
        frameon=False,
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    metrics = load_metrics(RUNS)

    build_summary_table(metrics).to_csv(
        OUTPUT_DIR / "laokou_oldrun_mlp_cnn_transformer_ptl_comparison_summary.csv",
        index=False,
    )
    plot_grouped_metrics(
        metrics,
        title="LaoKou Old-Run Range Performance Comparison: PTL vs Baselines",
        output_stem=OUTPUT_DIR / "laokou_oldrun_mlp_cnn_transformer_ptl_comparison",
        figsize=(10.0, 8.0),
    )

    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
