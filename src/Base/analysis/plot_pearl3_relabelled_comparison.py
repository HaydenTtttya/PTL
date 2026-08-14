from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

INPUT_SUMMARY = (
    REPO_ROOT
    / "results"
    / "summary"
    / "pearl5_2023_2024_oldrun_range"
    / "pearl5_model_summary.csv"
)

SUMMARY_OUTPUT_DIR = REPO_ROOT / "results" / "summary" / "pearl3_2023_2024_oldrun_range"
FIGURE_OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "pearl5_2023_2024_oldrun_range"

OUTPUT_PREFIX = "pearl3_s3_s5_relabelled"

STATION_MAPPING = {
    "新铺": {"station_label": "S1", "station_label_original": "S3"},
    "大墩": {"station_label": "S2", "station_label_original": "S4"},
    "五丰渡口": {"station_label": "S3", "station_label_original": "S5"},
}

MODEL_ORDER = ["MLP", "CNN", "Transformer", "PTL"]
COLORS = {
    "MLP": "#d95f02",
    "CNN": "#1b9e77",
    "Transformer": "#7570b3",
    "PTL": "#2b8cbe",
}


def configure_plot_style() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.unicode_minus"] = False


def load_relabelled_summary() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_SUMMARY, encoding="utf-8-sig")
    frame = frame[frame["station"].isin(STATION_MAPPING)].copy()
    if frame.empty:
        raise ValueError(f"No mapped stations found in {INPUT_SUMMARY}")

    frame["station_label"] = frame["station"].map(
        lambda name: STATION_MAPPING[name]["station_label"]
    )
    frame["station_label_original"] = frame["station"].map(
        lambda name: STATION_MAPPING[name]["station_label_original"]
    )
    frame["station_order"] = frame["station_label"].str.extract(r"S(\d+)").astype(int)
    frame["model_order"] = frame["model"].map(
        {model_name: index for index, model_name in enumerate(MODEL_ORDER)}
    )
    frame = frame.sort_values(["station_order", "model_order"]).reset_index(drop=True)
    return frame


def write_summary_tables(frame: pd.DataFrame) -> None:
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_summary = frame.drop(columns=["station_order", "model_order"])
    model_summary.to_csv(
        SUMMARY_OUTPUT_DIR / f"{OUTPUT_PREFIX}_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    average_summary = (
        frame.groupby("model", as_index=False)
        .agg(
            mean_overall_nse=("overall_nse", "mean"),
            mean_focus_nse=("focus_mean_nse", "mean"),
            mean_focus_rmse=("focus_mean_rmse", "mean"),
        )
        .sort_values("mean_focus_nse", ascending=False)
    )
    average_summary["rank_by_focus_nse"] = np.arange(1, len(average_summary) + 1)
    average_summary.to_csv(
        SUMMARY_OUTPUT_DIR / f"{OUTPUT_PREFIX}_model_average_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mapping_rows = [
        {
            "station_label": config["station_label"],
            "station_name": station_name,
            "station_label_original": config["station_label_original"],
        }
        for station_name, config in STATION_MAPPING.items()
    ]
    pd.DataFrame(mapping_rows).to_csv(
        SUMMARY_OUTPUT_DIR / f"{OUTPUT_PREFIX}_station_label_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_focus_mean_nse(frame: pd.DataFrame) -> None:
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    station_labels = (
        frame[["station_label", "station_order"]]
        .drop_duplicates()
        .sort_values("station_order")["station_label"]
        .tolist()
    )
    x = np.arange(len(station_labels), dtype=float)
    width = min(0.75 / len(MODEL_ORDER), 0.16)
    offsets = (np.arange(len(MODEL_ORDER)) - (len(MODEL_ORDER) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    for model_index, model_name in enumerate(MODEL_ORDER):
        model_frame = frame[frame["model"] == model_name]
        values = [
            float(
                model_frame.loc[
                    model_frame["station_label"] == station_label, "focus_mean_nse"
                ].iloc[0]
            )
            for station_label in station_labels
        ]
        ax.bar(
            x + offsets[model_index],
            values,
            width=width,
            label=model_name,
            color=COLORS[model_name],
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, zorder=2)
    ax.grid(axis="y", linestyle="--", alpha=0.55, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(station_labels, fontsize=15)
    ax.set_ylabel("Mean NSE (CODMn, DO, pH)", fontsize=15)
    ax.set_title("Pearl River Three-Station Comparison", fontsize=18, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(-0.8, 1.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=len(MODEL_ORDER),
        frameon=False,
        fontsize=13,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_stem = FIGURE_OUTPUT_DIR / f"{OUTPUT_PREFIX}_focus_mean_nse_comparison"
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_plot_style()
    frame = load_relabelled_summary()
    write_summary_tables(frame)
    plot_focus_mean_nse(frame)
    print(f"Summary saved to: {SUMMARY_OUTPUT_DIR}")
    print(f"Figure saved to: {FIGURE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
