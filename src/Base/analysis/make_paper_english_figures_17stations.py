from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_ROOT = REPO_ROOT / "results" / "summary" / "current_all_tested_stations_overall_nse"
FIG_ROOT = REPO_ROOT / "results" / "figures" / "current_all_tested_stations_overall_nse"
CASE_DIR = SUMMARY_ROOT / "均衡十五站方案_新增两站"
CASE_FIG_DIR = FIG_ROOT / "均衡十五站方案_新增两站"
SHAP_DIR = CASE_DIR / "SHAP分析"
TRAIN_AVAIL_DIR = SUMMARY_ROOT / "training_tail_availability_17stations_ptl"
OUT_DIR = CASE_FIG_DIR / "paper_figures_en_journal"
INCLUDE_FIGURE_HEADERS = False

MODEL_ORDER = ["MLP", "CNN", "LSTM", "Bi-LSTM", "CNN-LSTM", "Transformer", "PTL"]
INDICATOR_ORDER = ["CODMn", "DO", "NH4N", "pH"]
TARGET_ORDER = ["CODMn", "DO", "NH4N", "pH"]
LAG_ORDER = list(range(1, 13))

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue_xlight": "#EAF1FE",
    "blue_light": "#CEDFFE",
    "blue_base": "#A3BEFA",
    "blue_mid": "#5477C4",
    "blue_dark": "#2E4780",
    "gold_xlight": "#FFF4C2",
    "gold_light": "#FFEA8F",
    "gold_base": "#FFE15B",
    "gold_mid": "#B8A037",
    "gold_dark": "#736422",
    "orange_xlight": "#FFEDDE",
    "orange_light": "#FFBDA1",
    "orange_base": "#F0986E",
    "orange_mid": "#CC6F47",
    "orange_dark": "#804126",
    "olive_base": "#A3D576",
    "olive_dark": "#386411",
    "pink_base": "#F390CA",
    "pink_dark": "#8A3A6F",
    "neutral_xlight": "#F4F5F7",
    "neutral_light": "#E2E5EA",
    "neutral_base": "#C5CAD3",
    "neutral_mid": "#7A828F",
    "neutral_dark": "#464C55",
}

MODEL_COLORS = {
    "MLP": COLORS["neutral_light"],
    "CNN": COLORS["gold_light"],
    "LSTM": COLORS["neutral_base"],
    "Bi-LSTM": COLORS["neutral_mid"],
    "CNN-LSTM": COLORS["orange_light"],
    "Transformer": COLORS["blue_base"],
    "PTL": COLORS["orange_base"],
}

MODEL_EDGES = {
    "MLP": COLORS["neutral_mid"],
    "CNN": COLORS["gold_mid"],
    "LSTM": COLORS["neutral_mid"],
    "Bi-LSTM": COLORS["neutral_dark"],
    "CNN-LSTM": COLORS["orange_mid"],
    "Transformer": COLORS["blue_dark"],
    "PTL": COLORS["orange_dark"],
}

CHEM_LABELS = {
    "CODMn": r"COD$_{\mathrm{Mn}}$",
    "DO": "DO",
    "NH4N": r"NH$_4$-N",
    "pH": "pH",
}

REACH_LABELS = {
    "上游": "Upstream",
    "中游": "Midstream",
    "下游": "Downstream",
}

TYPE_LABELS = {
    "干流/主要水道": "Mainstem/major channel",
    "支流/区域河流": "Tributary/regional river",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "savefig.facecolor": TOKENS["surface"],
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "mathtext.default": "regular",
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "text.color": TOKENS["ink"],
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_header(fig, title: str, subtitle: str, *, x: float = 0.02, y: float = 0.985) -> None:
    if not INCLUDE_FIGURE_HEADERS:
        return
    fig.text(
        x,
        y,
        textwrap.fill(title, width=110, break_long_words=False),
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        x,
        y - 0.045,
        textwrap.fill(subtitle, width=142, break_long_words=False),
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )


def clean_axes(ax, *, grid_axis: str | None = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=TOKENS["grid"], linewidth=0.8, alpha=0.9)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def emphasize_axes(ax, *, color: str = "#000000", labelsize: float | None = None) -> None:
    for side in ("left", "bottom"):
        ax.spines[side].set_color(color)
    ax.tick_params(axis="both", colors=color, labelsize=labelsize)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)


def add_axis_arrowheads(ax, *, color: str = "#000000") -> None:
    arrowprops = {
        "arrowstyle": "-|>",
        "color": color,
        "linewidth": 0.8,
        "mutation_scale": 9,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    ax.annotate(
        "",
        xy=(1.018, 0.0),
        xytext=(1.0, 0.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=arrowprops,
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(0.0, 1.018),
        xytext=(0.0, 1.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=arrowprops,
        annotation_clip=False,
    )


def panel_label(
    ax,
    label: str,
    *,
    parentheses: bool = False,
    fontsize: float = 8.5,
    x: float = -0.08,
    y: float = 1.06,
) -> None:
    ax.text(
        x,
        y,
        f"({label})" if parentheses else label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=TOKENS["ink"],
    )


def save_figure(fig, stem: str, *, dpi: int = 450) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png", "svg"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def export_existing_study_area() -> list[Path]:
    journal_stem = CASE_FIG_DIR / "experiment_17_selected_pearl_river_basin_stations_trimmed_abbrev_provinces_journal_no_map_text"
    fallback_stem = CASE_FIG_DIR / "experiment_17_selected_pearl_river_basin_stations_trimmed_abbrev_provinces_no_map_text"
    source_stem = journal_stem if journal_stem.with_suffix(".pdf").exists() else fallback_stem
    outputs = []
    for ext in ("pdf", "png", "svg"):
        source = source_stem.with_suffix(f".{ext}")
        target = OUT_DIR / f"fig1_study_area_17stations_en.{ext}"
        if source.exists():
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            outputs.append(target)
    drawio = source_stem.with_suffix(".drawio")
    if drawio.exists():
        shutil.copy2(drawio, OUT_DIR / "fig1_study_area_17stations_en.drawio")
    return outputs


def draw_box(ax, xy, wh, title, body, *, face, edge):
    x, y = xy
    w, h = wh
    title_lines = title.count("\n") + 1
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.025,
        y + h - 0.052,
        title,
        fontsize=10.8,
        fontweight="bold",
        va="top",
        linespacing=1.15,
    )
    ax.text(
        x + 0.025,
        y + h - 0.105 - 0.045 * (title_lines - 1),
        body,
        fontsize=9.2,
        color=TOKENS["muted"],
        va="top",
        linespacing=1.35,
    )


def draw_arrow(ax, start, end, *, color=COLORS["neutral_dark"], rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def make_fig2_framework() -> list[Path]:
    fig, ax = plt.subplots(figsize=(14.6, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_header(
        fig,
        "Progressive transfer learning framework for daily water-quality prediction",
        "The model first learns cross-station structure from weekly Yangtze records, then transfers the shared representation to 17 Pearl River target stations through weekly, 4-day, and daily adaptation stages.",
        x=0.04,
        y=0.965,
    )

    def arch_box(
        xy,
        wh,
        title,
        body="",
        *,
        face="#FFFFFF",
        edge=COLORS["neutral_mid"],
        title_size=8.9,
        body_size=7.4,
        title_pad=0.020,
        body_pad=0.060,
        linewidth=1.2,
        rounding=0.014,
        align="left",
    ):
        x, y = xy
        w, h = wh
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={rounding}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
        )
        ax.add_patch(patch)
        tx = x + w / 2 if align == "center" else x + 0.014
        ha = "center" if align == "center" else "left"
        ax.text(
            tx,
            y + h - title_pad,
            title,
            ha=ha,
            va="top",
            fontsize=title_size,
            fontweight="bold",
            linespacing=1.12,
            color=TOKENS["ink"],
        )
        if body:
            title_lines = title.count("\n") + 1
            ax.text(
                tx,
                y + h - body_pad - 0.018 * (title_lines - 1),
                body,
                ha=ha,
                va="top",
                fontsize=body_size,
                linespacing=1.23,
                color=TOKENS["muted"],
            )
        return patch

    def arch_arrow(
        start,
        end,
        *,
        color=COLORS["neutral_dark"],
        rad=0.0,
        linestyle="-",
        linewidth=1.35,
        mutation_scale=12,
    ):
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arrow)

    def connector_label(x, y, text, *, color=TOKENS["muted"], size=7.2):
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=color,
        )

    # Main architecture regions.
    arch_box(
        (0.025, 0.115),
        (0.285, 0.780),
        "A. Source-domain pretraining",
        "",
        face=COLORS["blue_xlight"],
        edge=COLORS["blue_dark"],
        title_size=10.5,
        title_pad=0.030,
        linewidth=1.5,
    )
    arch_box(
        (0.355, 0.115),
        (0.300, 0.780),
        "B. Shared PTL backbone",
        "",
        face="#FFFFFF",
        edge=COLORS["neutral_mid"],
        title_size=10.5,
        title_pad=0.030,
        linewidth=1.5,
    )
    arch_box(
        (0.700, 0.115),
        (0.275, 0.780),
        "C. Target-domain transfer",
        "",
        face=COLORS["orange_xlight"],
        edge=COLORS["orange_dark"],
        title_size=10.5,
        title_pad=0.030,
        linewidth=1.5,
    )

    # Source-domain data and masked reconstruction task.
    arch_box(
        (0.052, 0.735),
        (0.230, 0.095),
        "Yangtze weekly windows",
        "18 stations, 2007-2018\nCOD$_{\\mathrm{Mn}}$, DO, NH$_4$-N, pH",
        face="#FFFFFF",
        edge=COLORS["blue_mid"],
        title_size=8.8,
        body_size=7.2,
    )
    for idx, label in enumerate(["S1", "S2", "...", "S18"]):
        y = 0.655 - idx * 0.037
        arch_box(
            (0.065, y),
            (0.055, 0.021),
            label,
            "",
            face=COLORS["blue_light"] if label != "..." else COLORS["neutral_xlight"],
            edge=COLORS["blue_mid"] if label != "..." else COLORS["neutral_mid"],
            title_size=6.8,
            title_pad=0.011,
            linewidth=0.9,
            rounding=0.008,
            align="center",
        )
        ax.plot([0.134, 0.256], [y + 0.0105, y + 0.0105], color=COLORS["blue_mid"], linewidth=1.0, alpha=0.75)
        for k in range(5):
            x = 0.145 + k * 0.022
            ax.plot([x, x + 0.010], [y + 0.0105, y + 0.0105], color="#FFFFFF", linewidth=2.2, alpha=0.95)
    arch_box(
        (0.052, 0.390),
        (0.230, 0.125),
        "Masked reconstruction objective",
        "15% station/feature masks\navailable-station mask\nlocal + cross-station losses",
        face=COLORS["gold_xlight"],
        edge=COLORS["gold_dark"],
        title_size=8.8,
        body_size=7.1,
    )
    arch_box(
        (0.052, 0.235),
        (0.230, 0.115),
        "Learned transferable state",
        "Backbone parameters exported as\nPTL-compatible initialization",
        face="#FFFFFF",
        edge=COLORS["blue_dark"],
        title_size=8.8,
        body_size=7.2,
    )
    arch_arrow((0.167, 0.735), (0.167, 0.668), color=COLORS["blue_dark"])
    arch_arrow((0.167, 0.390), (0.167, 0.362), color=COLORS["gold_dark"])

    # Shared single-station temporal encoder.
    y_top = 0.735
    module_specs = [
        (0.380, y_top, 0.105, 0.080, "Normalize", "window\nmean/std", COLORS["neutral_xlight"], COLORS["neutral_mid"]),
        (0.500, y_top, 0.105, 0.080, "Temporal\nadapter", "depthwise\nConv1D", COLORS["blue_xlight"], COLORS["blue_mid"]),
        (0.380, 0.610, 0.105, 0.080, "Time-feature\nembedding", "seq tokens\n-> d=256", COLORS["blue_xlight"], COLORS["blue_mid"]),
        (0.500, 0.610, 0.105, 0.080, "Transformer\nencoder", "3 layers\n8 heads", COLORS["gold_xlight"], COLORS["gold_dark"]),
        (0.440, 0.485, 0.105, 0.080, "Flatten\nhead", "forecast /\nreconstruct", COLORS["orange_xlight"], COLORS["orange_mid"]),
    ]
    for x, y, w, h, title, body, face, edge in module_specs:
        arch_box(
            (x, y),
            (w, h),
            title,
            body,
            face=face,
            edge=edge,
            title_size=7.5,
            body_size=6.5,
            title_pad=0.017,
            body_pad=0.046,
            align="center",
        )
    arch_arrow((0.485, y_top + 0.040), (0.500, y_top + 0.040), color=COLORS["neutral_dark"], mutation_scale=10)
    arch_arrow((0.552, y_top), (0.432, 0.690), color=COLORS["neutral_dark"], rad=0.18, mutation_scale=10)
    arch_arrow((0.485, 0.650), (0.500, 0.650), color=COLORS["neutral_dark"], mutation_scale=10)
    arch_arrow((0.552, 0.610), (0.492, 0.565), color=COLORS["neutral_dark"], rad=0.08, mutation_scale=10)

    arch_box(
        (0.380, 0.300),
        (0.105, 0.090),
        "Station\nidentity",
        "embedding",
        face=COLORS["blue_xlight"],
        edge=COLORS["blue_mid"],
        title_size=7.6,
        body_size=6.5,
        align="center",
    )
    arch_box(
        (0.500, 0.300),
        (0.118, 0.090),
        "Cross-station\nattention",
        "4 heads\npair bias",
        face=COLORS["gold_xlight"],
        edge=COLORS["gold_dark"],
        title_size=7.6,
        body_size=6.5,
        align="center",
    )
    arch_box(
        (0.440, 0.185),
        (0.120, 0.075),
        "Fusion gate",
        "local + cross\nreconstruction",
        face=COLORS["orange_xlight"],
        edge=COLORS["orange_mid"],
        title_size=7.6,
        body_size=6.4,
        align="center",
    )
    arch_arrow((0.485, 0.345), (0.500, 0.345), color=COLORS["neutral_dark"], mutation_scale=10)
    arch_arrow((0.558, 0.300), (0.515, 0.260), color=COLORS["neutral_dark"], mutation_scale=10)
    arch_arrow((0.492, 0.485), (0.558, 0.390), color=COLORS["neutral_mid"], rad=-0.10, mutation_scale=10)

    # Target-domain progressive adaptation.
    arch_box(
        (0.727, 0.747),
        (0.220, 0.075),
        "Pearl River target stations",
        "17 stations, 2021-2024\ntrain/validation/test split fixed",
        face="#FFFFFF",
        edge=COLORS["orange_mid"],
        title_size=8.4,
        body_size=6.8,
    )
    stage_boxes = [
        (
            0.700,
            "Stage 1: weekly",
            "56 days -> 8 tokens\none-step target\ninitialize Stage 2",
            COLORS["blue_xlight"],
            COLORS["blue_mid"],
        ),
        (
            0.525,
            "Stage 2: 4-day",
            "32 days -> 8 tokens\none-step target\ninitialize Stage 3",
            COLORS["gold_xlight"],
            COLORS["gold_dark"],
        ),
        (
            0.350,
            "Stage 3: daily",
            "12 days -> 12 tokens\nnext-day prediction",
            "#FFFFFF",
            COLORS["orange_dark"],
        ),
    ]
    for y, title, body, face, edge in stage_boxes:
        arch_box(
            (0.727, y - 0.090),
            (0.220, 0.105),
            title,
            body,
            face=face,
            edge=edge,
            title_size=8.2,
            body_size=6.8,
        )
    arch_box(
        (0.727, 0.142),
        (0.220, 0.100),
        "Outputs and checks",
        "COD$_{\\mathrm{Mn}}$, DO, NH$_4$-N, pH forecasts\nNSE, RMSE, MAE; SHAP interpretation",
        face=COLORS["neutral_xlight"],
        edge=COLORS["neutral_dark"],
        title_size=8.2,
        body_size=6.7,
    )
    arch_arrow((0.920, 0.747), (0.920, 0.715), color=COLORS["orange_dark"], mutation_scale=10)
    arch_arrow((0.837, 0.610), (0.837, 0.540), color=COLORS["neutral_dark"], mutation_scale=10)
    arch_arrow((0.837, 0.435), (0.837, 0.365), color=COLORS["neutral_dark"], mutation_scale=10)
    arch_arrow((0.837, 0.260), (0.837, 0.242), color=COLORS["orange_dark"], mutation_scale=10)
    connector_label(0.868, 0.575, "weights", size=6.9)
    connector_label(0.868, 0.400, "weights", size=6.9)

    # Cross-region information flow.
    arch_arrow((0.282, 0.785), (0.380, 0.775), color=COLORS["blue_dark"])
    connector_label(0.331, 0.808, "station windows", color=COLORS["blue_dark"])
    arch_arrow((0.545, 0.525), (0.727, 0.665), color=COLORS["orange_dark"], rad=-0.30, linestyle="--", linewidth=1.35)
    connector_label(0.632, 0.632, "pretrained backbone", color=COLORS["orange_dark"])
    ax.text(
        0.505,
        0.140,
        "Source-only pretraining; target test data remain isolated until final evaluation.",
        ha="center",
        va="center",
        fontsize=7.4,
        color=TOKENS["muted"],
    )

    return save_figure(fig, "fig2_ptl_framework_en")


def make_fig2_concrete_architecture() -> list[Path]:
    font_candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    chinese_font = next((path for path in font_candidates if Path(path).exists()), None)
    font_prop = font_manager.FontProperties(fname=chinese_font) if chinese_font else None

    fig, ax = plt.subplots(figsize=(16.0, 9.0))
    fig.patch.set_facecolor("#DCDCDC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def put_text(
        x,
        y,
        text,
        *,
        fontsize=12,
        color="#1F2430",
        weight="normal",
        ha="center",
        va="center",
        linespacing=1.15,
    ):
        ax.text(
            x,
            y,
            text,
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            fontproperties=font_prop,
            linespacing=linespacing,
        )

    def box(
        x,
        y,
        w,
        h,
        text,
        *,
        face="#FFFFFF",
        edge="#1F2430",
        text_color="#1F2430",
        fontsize=12,
        weight="normal",
        linewidth=1.6,
        rounding=0.002,
        shadow=False,
        zorder=2,
    ):
        if shadow:
            shadow_patch = FancyBboxPatch(
                (x + 0.006, y - 0.006),
                w,
                h,
                boxstyle=f"round,pad=0.004,rounding_size={rounding}",
                facecolor="#000000",
                edgecolor="none",
                alpha=0.22,
                zorder=zorder - 1,
            )
            ax.add_patch(shadow_patch)
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={rounding}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.add_patch(patch)
        put_text(
            x + w / 2,
            y + h / 2,
            text,
            fontsize=fontsize,
            color=text_color,
            weight=weight,
        )
        return patch

    def arrow(start, end, *, color="#333333", linewidth=1.25, rad=0.0, style="-|>", zorder=4):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle=style,
                mutation_scale=13,
                linewidth=linewidth,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                zorder=zorder,
            )
        )

    def callout_arrow(start, end, label, *, y_offset=0.030):
        arrow(start, end, color="#8B0000", linewidth=3.0)
        put_text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + y_offset,
            label,
            fontsize=15,
            weight="bold",
            color="#111111",
        )

    white_panel = FancyBboxPatch(
        (0.030, 0.075),
        0.930,
        0.850,
        boxstyle="round,pad=0,rounding_size=0",
        facecolor="#FFFFFF",
        edgecolor="#FFFFFF",
        linewidth=0,
        zorder=0,
    )
    ax.add_patch(white_panel)

    navy = "#0D2548"
    red = "#DE0000"
    dark_red = "#8B0000"
    lavender = "#F0EEFF"
    purple = "#9B84FF"
    gray_line = "#333333"

    # Stage-0 pretraining and exported backbone state.
    box(
        0.058,
        0.660,
        0.195,
        0.080,
        "Stage0 跨站预训练",
        face=navy,
        edge="#07152A",
        text_color="#E9EDF5",
        fontsize=15.5,
        weight="bold",
        linewidth=2.2,
    )
    box(
        0.035,
        0.470,
        0.245,
        0.120,
        "Masked Reconstruction\n学习通用 Backbone",
        face=navy,
        edge="#07152A",
        text_color="#E9EDF5",
        fontsize=14.3,
        weight="bold",
        linewidth=2.2,
    )
    box(
        0.053,
        0.300,
        0.200,
        0.080,
        "导出单站 Backbone 权重",
        face=navy,
        edge="#07152A",
        text_color="#E9EDF5",
        fontsize=14.0,
        weight="bold",
        linewidth=2.2,
    )
    arrow((0.155, 0.660), (0.155, 0.590), color=navy, linewidth=2.2)
    arrow((0.155, 0.470), (0.155, 0.380), color=navy, linewidth=2.2)

    # Progressive target-domain transfer stages.
    stage_x = 0.305
    stage_w = 0.210
    stage_specs = [
        (0.775, "Stage1 Weekly"),
        (0.665, "Stage1 目标站\nWeekly Adapter"),
        (0.555, "Stage2 4-day"),
        (0.445, "Stage2 目标站\n4-day Adapter"),
        (0.335, "Stage3 Daily"),
        (0.225, "最终目标站预测"),
    ]
    previous_center = None
    for y, label in stage_specs:
        box(
            stage_x,
            y,
            stage_w,
            0.075,
            label,
            face=red,
            edge=dark_red,
            text_color="#FFFFFF",
            fontsize=18 if "\n" not in label else 17,
            weight="bold",
            linewidth=2.3,
            shadow=True,
        )
        center = (stage_x + stage_w / 2, y + 0.0375)
        if previous_center is not None:
            arrow((previous_center[0], previous_center[1] - 0.042), (center[0], center[1] + 0.042), color=dark_red, linewidth=2.0)
        previous_center = center

    arrow((0.253, 0.340), (stage_x, 0.702), color=dark_red, linewidth=2.4, rad=-0.25)
    put_text(0.289, 0.625, "初始化", fontsize=13, weight="bold", color="#111111")

    # Detailed cross-station pretraining module.
    right_x, right_y, right_w, right_h = 0.615, 0.040, 0.350, 0.930
    right_panel = FancyBboxPatch(
        (right_x, right_y),
        right_w,
        right_h,
        boxstyle="round,pad=0.006,rounding_size=0",
        facecolor="#FFFFFF",
        edgecolor=dark_red,
        linewidth=2.2,
        zorder=1,
    )
    ax.add_patch(right_panel)

    callout_arrow((stage_x + stage_w, 0.702), (right_x, 0.702), "调用", y_offset=0.030)
    callout_arrow((stage_x + stage_w, 0.482), (right_x, 0.482), "共享权重", y_offset=0.030)

    cx = right_x + right_w / 2
    top_w = 0.105
    box(cx - top_w / 2, 0.895, top_w, 0.058, "跨站输入\n[目标站 + 源站1..N]", face=lavender, edge=purple, fontsize=9.2)
    box(cx - top_w / 2, 0.792, top_w, 0.060, "每个站点独立编码\n共享单站 Backbone", face=lavender, edge=purple, fontsize=8.8)
    box(cx - 0.135, 0.705, 0.125, 0.044, "每站聚合成 Station Token", face=lavender, edge=purple, fontsize=8.2)
    arrow((cx, 0.895), (cx, 0.852), color=gray_line)
    arrow((cx - 0.030, 0.792), (cx - 0.070, 0.749), color=gray_line, rad=0.10)

    inner = FancyBboxPatch(
        (right_x + 0.095, 0.365),
        0.145,
        0.290,
        boxstyle="round,pad=0.004,rounding_size=0",
        facecolor="none",
        edgecolor=red,
        linewidth=2.5,
        zorder=3,
    )
    ax.add_patch(inner)

    box(right_x + 0.110, 0.595, 0.115, 0.044, "Coverage / Missing Mask", face=lavender, edge=purple, fontsize=7.5)
    box(right_x + 0.108, 0.488, 0.120, 0.074, "Cross-Station Interaction\nStation-level Multi-Head\nAttention", face=lavender, edge=purple, fontsize=7.2)
    box(right_x + 0.126, 0.392, 0.084, 0.044, "Target Context", face=lavender, edge=purple, fontsize=7.6)
    arrow((right_x + 0.168, 0.705), (right_x + 0.168, 0.639), color=gray_line)
    arrow((right_x + 0.168, 0.595), (right_x + 0.168, 0.562), color=gray_line)
    arrow((right_x + 0.168, 0.488), (right_x + 0.168, 0.436), color=gray_line)

    box(right_x + 0.010, 0.392, 0.105, 0.044, "目标站 Local Token", face=lavender, edge=purple, fontsize=7.8)
    box(right_x + 0.236, 0.392, 0.110, 0.044, "每站得到 Feature Tokens", face=lavender, edge=purple, fontsize=7.0)
    box(right_x + 0.236, 0.305, 0.110, 0.044, "目标站 Feature Tokens", face=lavender, edge=purple, fontsize=7.0)
    box(right_x + 0.105, 0.305, 0.114, 0.044, "Context + Local Fusion", face=lavender, edge=purple, fontsize=7.8)
    box(right_x + 0.184, 0.228, 0.104, 0.060, "Fusion Gate\n局部预测 + 跨站增量", face=lavender, edge=purple, fontsize=7.2)
    box(right_x + 0.178, 0.136, 0.116, 0.044, "Prediction / Reconstruction Head", face=lavender, edge=purple, fontsize=6.6)
    box(right_x + 0.172, 0.060, 0.128, 0.044, "输出目标站未来预测", face=lavender, edge=purple, fontsize=7.4)

    arrow((right_x + 0.290, 0.792), (right_x + 0.310, 0.436), color=gray_line, rad=-0.42)
    arrow((right_x + 0.072, 0.705), (right_x + 0.060, 0.436), color=gray_line, rad=0.36)
    arrow((right_x + 0.062, 0.392), (right_x + 0.122, 0.349), color=gray_line, rad=0.25)
    arrow((right_x + 0.168, 0.392), (right_x + 0.166, 0.349), color=gray_line)
    arrow((right_x + 0.291, 0.392), (right_x + 0.291, 0.349), color=gray_line)
    arrow((right_x + 0.291, 0.305), (right_x + 0.252, 0.288), color=gray_line, rad=-0.15)
    arrow((right_x + 0.162, 0.305), (right_x + 0.218, 0.288), color=gray_line, rad=0.15)
    arrow((right_x + 0.236, 0.228), (right_x + 0.236, 0.180), color=gray_line)
    arrow((right_x + 0.236, 0.136), (right_x + 0.236, 0.104), color=gray_line)

    put_text(
        0.340,
        0.115,
        "Stage0 学习跨站共享表示；Stage1-3 继承单站 Backbone\n并逐级适配目标站。",
        fontsize=8.1,
        color="#666666",
        ha="center",
    )

    return save_figure(fig, "fig2_ptl_framework_en", dpi=500)


def make_fig2_detailed_architecture() -> list[Path]:
    font_candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    chinese_font = next((path for path in font_candidates if Path(path).exists()), None)
    font_prop = font_manager.FontProperties(fname=chinese_font) if chinese_font else None

    fig, ax = plt.subplots(figsize=(16.0, 8.8))
    fig.patch.set_facecolor(TOKENS["surface"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def text(
        x,
        y,
        value,
        *,
        fontsize=9,
        color=TOKENS["ink"],
        weight="normal",
        ha="center",
        va="center",
        linespacing=1.18,
        zorder=5,
    ):
        ax.text(
            x,
            y,
            value,
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            fontproperties=font_prop,
            ha=ha,
            va=va,
            linespacing=linespacing,
            zorder=zorder,
        )

    def rounded(
        x,
        y,
        w,
        h,
        title,
        body="",
        *,
        face="#FFFFFF",
        edge=COLORS["neutral_mid"],
        title_color=TOKENS["ink"],
        body_color=TOKENS["muted"],
        title_size=9.0,
        body_size=7.5,
        linewidth=1.1,
        rounding=0.014,
        pad=0.010,
        title_y=0.024,
        body_y=0.062,
        align="center",
        zorder=2,
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad={pad},rounding_size={rounding}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.add_patch(patch)
        ha = "center" if align == "center" else "left"
        tx = x + w / 2 if align == "center" else x + 0.014
        text(
            tx,
            y + h - title_y,
            title,
            fontsize=title_size,
            color=title_color,
            weight="bold",
            ha=ha,
            va="top",
        )
        if body:
            title_lines = title.count("\n") + 1
            text(
                tx,
                y + h - body_y - 0.015 * (title_lines - 1),
                body,
                fontsize=body_size,
                color=body_color,
                ha=ha,
                va="top",
            )
        return patch

    def chip(x, y, w, h, label, *, face="#FFFFFF", edge=COLORS["neutral_mid"], color=TOKENS["ink"], size=7.2):
        rounded(
            x,
            y,
            w,
            h,
            label,
            "",
            face=face,
            edge=edge,
            title_color=color,
            title_size=size,
            linewidth=0.9,
            rounding=0.008,
            title_y=0.012,
            pad=0.006,
        )

    def arrow(start, end, *, color=COLORS["neutral_dark"], linewidth=1.15, rad=0.0, dashed=False, scale=11):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=scale,
                linewidth=linewidth,
                color=color,
                linestyle="--" if dashed else "-",
                connectionstyle=f"arc3,rad={rad}",
                zorder=4,
            )
        )

    # Background panels.
    panels = [
        (0.030, 0.075, 0.305, 0.850, "A. Stage0 跨站掩码预训练", COLORS["blue_xlight"], COLORS["blue_dark"]),
        (0.365, 0.075, 0.230, 0.850, "B. 权重导出与初始化", COLORS["neutral_xlight"], COLORS["neutral_dark"]),
        (0.625, 0.075, 0.345, 0.850, "C. 目标站渐进迁移", COLORS["orange_xlight"], COLORS["orange_dark"]),
    ]
    for x, y, w, h, title, face, edge in panels:
        rounded(
            x,
            y,
            w,
            h,
            title,
            "",
            face=face,
            edge=edge,
            title_size=11.2,
            linewidth=1.6,
            rounding=0.020,
            title_y=0.035,
            pad=0.012,
            align="left",
            zorder=1,
        )

    # A. Cross-station pretraining details.
    rounded(
        0.055,
        0.782,
        0.255,
        0.080,
        "多站周尺度输入",
        "Yangtze basin, 18 stations, 2007-2018\nTensor: B x N x 168 x 4",
        face="#FFFFFF",
        edge=COLORS["blue_mid"],
        title_size=8.6,
        body_size=7.2,
    )
    rounded(
        0.055,
        0.660,
        0.255,
        0.090,
        "Mask 构造",
        "station / feature / temporal masks\nmask ratio=0.15; coverage mask tracks missing stations",
        face=COLORS["gold_xlight"],
        edge=COLORS["gold_dark"],
        title_size=8.6,
        body_size=7.0,
    )
    rounded(
        0.055,
        0.460,
        0.255,
        0.165,
        "共享单站 Backbone 编码",
        "",
        face="#FFFFFF",
        edge=COLORS["blue_dark"],
        title_size=8.7,
        title_y=0.022,
    )
    chip(0.072, 0.540, 0.050, 0.040, "Normalize", face=COLORS["neutral_xlight"])
    chip(0.130, 0.540, 0.062, 0.040, "Temporal\nAdapter", face=COLORS["blue_xlight"], edge=COLORS["blue_mid"], size=6.5)
    chip(0.200, 0.540, 0.082, 0.040, "Time-feature\nEmbedding", face=COLORS["blue_xlight"], edge=COLORS["blue_mid"], size=6.4)
    chip(0.100, 0.485, 0.085, 0.045, "Transformer\nEncoder", face=COLORS["gold_xlight"], edge=COLORS["gold_dark"], size=6.7)
    chip(0.197, 0.485, 0.075, 0.045, "Flatten\nHead", face=COLORS["orange_xlight"], edge=COLORS["orange_mid"], size=6.8)
    for start, end in [
        ((0.122, 0.560), (0.130, 0.560)),
        ((0.192, 0.560), (0.200, 0.560)),
        ((0.241, 0.540), (0.143, 0.530)),
        ((0.185, 0.507), (0.197, 0.507)),
    ]:
        arrow(start, end, scale=8, linewidth=0.9)

    rounded(
        0.055,
        0.285,
        0.255,
        0.135,
        "跨站交互与上下文融合",
        "",
        face="#FFFFFF",
        edge=COLORS["gold_dark"],
        title_size=8.7,
        title_y=0.022,
    )
    chip(0.070, 0.355, 0.068, 0.036, "Station\nToken", face=COLORS["blue_xlight"], edge=COLORS["blue_mid"], size=6.4)
    chip(0.147, 0.355, 0.063, 0.036, "Identity\nBias", face=COLORS["blue_xlight"], edge=COLORS["blue_mid"], size=6.4)
    chip(0.220, 0.355, 0.073, 0.036, "Coverage\nMask", face=COLORS["neutral_xlight"], size=6.4)
    chip(0.082, 0.307, 0.105, 0.036, "Station-level MHA\n4 heads + pair bias", face=COLORS["gold_xlight"], edge=COLORS["gold_dark"], size=6.1)
    chip(0.202, 0.307, 0.083, 0.036, "Fusion\nGate", face=COLORS["orange_xlight"], edge=COLORS["orange_mid"], size=6.5)
    arrow((0.138, 0.373), (0.147, 0.373), scale=8, linewidth=0.9)
    arrow((0.210, 0.373), (0.220, 0.373), scale=8, linewidth=0.9)
    arrow((0.135, 0.355), (0.132, 0.343), scale=8, linewidth=0.9)
    arrow((0.187, 0.325), (0.202, 0.325), scale=8, linewidth=0.9)

    rounded(
        0.055,
        0.145,
        0.255,
        0.090,
        "预训练目标",
        "local reconstruction + cross-station reconstruction\nloss = L_local + 0.5 L_cross-all + L_cross-masked",
        face=COLORS["neutral_xlight"],
        edge=COLORS["neutral_dark"],
        title_size=8.6,
        body_size=6.9,
    )
    for start, end in [
        ((0.183, 0.782), (0.183, 0.750)),
        ((0.183, 0.660), (0.183, 0.625)),
        ((0.183, 0.460), (0.183, 0.420)),
        ((0.183, 0.285), (0.183, 0.235)),
    ]:
        arrow(start, end, color=COLORS["blue_dark"], linewidth=1.2, scale=10)

    # B. Export and initialization.
    rounded(
        0.392,
        0.735,
        0.175,
        0.095,
        "导出可迁移参数",
        "TemporalAdapter\nEmbedding / Encoder / Forecast Head",
        face="#FFFFFF",
        edge=COLORS["blue_dark"],
        title_size=8.7,
        body_size=7.0,
    )
    rounded(
        0.392,
        0.565,
        0.175,
        0.120,
        "兼容加载",
        "load_matching_weights()\nonly same-name + same-shape weights\nskip incompatible heads if needed",
        face=COLORS["gold_xlight"],
        edge=COLORS["gold_dark"],
        title_size=8.7,
        body_size=6.8,
    )
    rounded(
        0.392,
        0.385,
        0.175,
        0.130,
        "不直接进入微调的部分",
        "Cross-station MHA\nstation identity embedding\ncoverage-mask fusion gate",
        face="#FFFFFF",
        edge=COLORS["neutral_mid"],
        title_size=8.7,
        body_size=6.9,
    )
    rounded(
        0.392,
        0.180,
        0.175,
        0.125,
        "初始化序列",
        "theta_0 -> theta_1 -> theta_2 -> theta_3\nStage best weights hand off to next stage",
        face=COLORS["orange_xlight"],
        edge=COLORS["orange_dark"],
        title_size=8.7,
        body_size=6.9,
    )
    arrow((0.310, 0.535), (0.392, 0.780), color=COLORS["blue_dark"], rad=0.10, linewidth=1.4)
    arrow((0.480, 0.735), (0.480, 0.685), color=COLORS["neutral_dark"])
    arrow((0.480, 0.565), (0.480, 0.515), color=COLORS["neutral_dark"])
    arrow((0.480, 0.385), (0.480, 0.305), color=COLORS["neutral_mid"], dashed=True, linewidth=1.0)

    # C. Progressive adaptation details.
    rounded(
        0.650,
        0.782,
        0.290,
        0.080,
        "目标域输入",
        "Pearl River 17 stations, 2021-2024\nfeatures: CODMn, DO, NH4-N, pH",
        face="#FFFFFF",
        edge=COLORS["orange_mid"],
        title_size=8.7,
        body_size=7.0,
    )

    stage_cards = [
        (0.650, 0.600, 0.290, 0.130, "Stage 1: weekly adaptation", "56 days -> 8 tokens; one-step forecast"),
        (0.650, 0.420, 0.290, 0.130, "Stage 2: 4-day adaptation", "32 days -> 8 tokens; initialized by Stage 1"),
        (0.650, 0.240, 0.290, 0.130, "Stage 3: daily fine-tuning", "12 days -> 12 tokens; next-day forecast; soft gap <= 6"),
    ]
    for x, y, w, h, title, body in stage_cards:
        rounded(
            x,
            y,
            w,
            h,
            title,
            body,
            face="#FFFFFF",
            edge=COLORS["orange_dark"],
            title_size=8.6,
            body_size=6.35,
            align="left",
            linewidth=1.25,
            title_y=0.025,
            body_y=0.058,
        )
        chip(x + 0.018, y + 0.014, 0.052, 0.030, "Norm", face=COLORS["neutral_xlight"], size=6.2)
        chip(x + 0.078, y + 0.014, 0.072, 0.030, "Adapter", face=COLORS["blue_xlight"], edge=COLORS["blue_mid"], size=6.2)
        chip(x + 0.158, y + 0.014, 0.072, 0.030, "Encoder", face=COLORS["gold_xlight"], edge=COLORS["gold_dark"], size=6.2)
        chip(x + 0.238, y + 0.014, 0.038, 0.030, "Head", face=COLORS["orange_xlight"], edge=COLORS["orange_mid"], size=6.0)
        arrow((x + 0.070, y + 0.029), (x + 0.078, y + 0.029), scale=7, linewidth=0.8)
        arrow((x + 0.150, y + 0.029), (x + 0.158, y + 0.029), scale=7, linewidth=0.8)
        arrow((x + 0.230, y + 0.029), (x + 0.238, y + 0.029), scale=7, linewidth=0.8)

    arrow((0.795, 0.782), (0.795, 0.730), color=COLORS["orange_dark"], linewidth=1.25)
    arrow((0.795, 0.600), (0.795, 0.550), color=COLORS["orange_dark"], linewidth=1.25)
    text(0.835, 0.575, "theta_1", fontsize=7.0, color=COLORS["orange_dark"], weight="bold")
    arrow((0.795, 0.420), (0.795, 0.370), color=COLORS["orange_dark"], linewidth=1.25)
    text(0.835, 0.395, "theta_2", fontsize=7.0, color=COLORS["orange_dark"], weight="bold")

    rounded(
        0.650,
        0.115,
        0.290,
        0.090,
        "输出与评估",
        "target-station next-day prediction\nmetrics: NSE, RMSE, MAE; SHAP interpretation",
        face=COLORS["neutral_xlight"],
        edge=COLORS["neutral_dark"],
        title_size=8.6,
        body_size=7.0,
    )
    arrow((0.795, 0.240), (0.795, 0.205), color=COLORS["orange_dark"], linewidth=1.25)

    # Cross-column transfers.
    arrow((0.567, 0.625), (0.650, 0.665), color=COLORS["orange_dark"], rad=-0.06, linewidth=1.4)
    text(0.609, 0.655, "initialize", fontsize=7.6, color=COLORS["orange_dark"], weight="bold")

    text(
        0.500,
        0.045,
        "Figure 2. Detailed PTL architecture: cross-station masked pretraining learns a transferable single-station backbone, then weekly, 4-day, and daily stages progressively adapt it to each target station.",
        fontsize=8.2,
        color=TOKENS["muted"],
    )

    return save_figure(fig, "fig2_ptl_framework_en", dpi=500)


def make_fig2_reference_style_architecture() -> list[Path]:
    fig, ax = plt.subplots(figsize=(15.6, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = TOKENS["ink"]
    muted = TOKENS["muted"]
    blue = COLORS["blue_mid"]
    blue_dark = COLORS["blue_dark"]
    gold = COLORS["gold_mid"]
    gold_dark = COLORS["gold_dark"]
    orange = COLORS["orange_mid"]
    orange_dark = COLORS["orange_dark"]

    def label(x, y, value, *, size=8.5, color=ink, weight="normal", ha="center", va="center", zorder=6):
        ax.text(
            x,
            y,
            value,
            fontsize=size,
            color=color,
            fontweight=weight,
            ha=ha,
            va=va,
            linespacing=1.15,
            zorder=zorder,
        )

    def box(
        x,
        y,
        w,
        h,
        value,
        *,
        face="#FFFFFF",
        edge=COLORS["neutral_mid"],
        color=ink,
        size=8.0,
        weight="normal",
        linewidth=1.0,
        rounding=0.009,
        zorder=3,
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.006,rounding_size={rounding}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.add_patch(patch)
        label(x + w / 2, y + h / 2, value, size=size, color=color, weight=weight, zorder=zorder + 1)
        return patch

    def arrow(start, end, *, color=COLORS["neutral_dark"], width=1.15, dashed=False, rad=0.0, scale=11):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=scale,
                linewidth=width,
                linestyle="--" if dashed else "-",
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                zorder=5,
            )
        )

    def panel_title(x, y, panel, title):
        label(x, y, panel, size=10.5, weight="bold", ha="left")
        label(x + 0.025, y, title, size=10.5, weight="bold", ha="left")

    def waveform_stack(x, y, w, h, *, color=blue, count=4, seed=0):
        rng = np.random.default_rng(seed)
        frame = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.003,rounding_size=0.004",
            facecolor="#FFFFFF",
            edgecolor=COLORS["neutral_mid"],
            linewidth=0.9,
            zorder=3,
        )
        ax.add_patch(frame)
        xs = np.linspace(x + 0.008, x + w - 0.008, 70)
        for idx in range(count):
            center = y + h * (idx + 0.5) / count
            phase = rng.uniform(0, 2 * np.pi)
            values = (
                0.11 * h * np.sin(np.linspace(0, 3.5 * np.pi, len(xs)) + phase)
                + 0.035 * h * rng.normal(size=len(xs))
            )
            ax.plot(xs, center + values, color=color, linewidth=0.9, clip_on=True, zorder=4)
        return frame

    def token_row(x, y, w, h, count, *, colors, masked=None, edge=COLORS["neutral_mid"]):
        gap = w * 0.025
        token_w = (w - gap * (count - 1)) / count
        masked = set(masked or [])
        centers = []
        for idx in range(count):
            tx = x + idx * (token_w + gap)
            face = "#FFFFFF" if idx in masked else colors[idx % len(colors)]
            linestyle = "--" if idx in masked else "-"
            token = FancyBboxPatch(
                (tx, y),
                token_w,
                h,
                boxstyle="round,pad=0.002,rounding_size=0.002",
                facecolor=face,
                edgecolor=edge,
                linewidth=0.8,
                linestyle=linestyle,
                zorder=4,
            )
            ax.add_patch(token)
            centers.append((tx + token_w / 2, y + h / 2))
        return centers

    # (a) End-to-end PTL overview.
    panel_title(0.025, 0.955, "a", "PTL overview")
    source_panel = FancyBboxPatch(
        (0.030, 0.625),
        0.455,
        0.285,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        facecolor=COLORS["blue_xlight"],
        edgecolor=blue_dark,
        linewidth=1.25,
        zorder=1,
    )
    target_panel = FancyBboxPatch(
        (0.515, 0.625),
        0.455,
        0.285,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        facecolor=COLORS["orange_xlight"],
        edgecolor=orange_dark,
        linewidth=1.25,
        zorder=1,
    )
    ax.add_patch(source_panel)
    ax.add_patch(target_panel)
    label(0.050, 0.880, "Source-domain masked pretraining", size=9.2, weight="bold", ha="left", color=blue_dark)
    label(0.535, 0.880, "Target-domain progressive adaptation", size=9.2, weight="bold", ha="left", color=orange_dark)

    waveform_stack(0.050, 0.710, 0.075, 0.115, color=blue, count=4, seed=3)
    label(0.0875, 0.690, r"$X_s\in\mathbb{R}^{B\times N\times168\times4}$", size=6.8, color=muted)
    box(0.155, 0.725, 0.060, 0.080, "Mask\n$r=0.15$", face=COLORS["gold_xlight"], edge=gold_dark, size=7.2, weight="bold")
    box(0.245, 0.700, 0.085, 0.130, "Shared station\nencoder\n$\\theta_0$", face="#FFFFFF", edge=blue_dark, size=7.5, weight="bold")
    box(0.360, 0.700, 0.095, 0.130, "Station-token\nattention\n+ fusion gate", face=COLORS["gold_xlight"], edge=gold_dark, size=7.4, weight="bold")
    arrow((0.125, 0.768), (0.155, 0.768), color=blue_dark)
    arrow((0.215, 0.768), (0.245, 0.768), color=blue_dark)
    arrow((0.330, 0.768), (0.360, 0.768), color=gold_dark)
    label(0.407, 0.674, r"$\hat X_{local},\hat X_{cross}$", size=7.0, color=gold_dark)
    label(0.252, 0.647, r"$\mathcal{L}_{pre}=\mathcal{L}_{local}+0.5\mathcal{L}_{cross-all}+\mathcal{L}_{cross-mask}$", size=7.0, color=blue_dark)
    arrow((0.407, 0.700), (0.407, 0.676), color=gold_dark, width=1.0, scale=9)

    waveform_stack(0.535, 0.720, 0.060, 0.095, color=orange, count=4, seed=7)
    label(0.565, 0.697, "target station", size=6.8, color=muted)
    stages = [
        (0.625, "Weekly", "56 d -> 8"),
        (0.725, "4-day", "32 d -> 8"),
        (0.825, "Daily", "12 d -> 12"),
    ]
    for x, title, meta in stages:
        box(x, 0.715, 0.075, 0.100, f"{title}\n{meta}", face="#FFFFFF", edge=orange_dark, size=7.2, weight="bold")
    waveform_stack(0.925, 0.735, 0.035, 0.065, color=orange_dark, count=1, seed=11)
    label(0.943, 0.712, r"$\hat y_{t+1}$", size=7.2, color=orange_dark)
    arrow((0.595, 0.765), (0.625, 0.765), color=orange_dark)
    arrow((0.700, 0.765), (0.725, 0.765), color=orange_dark)
    arrow((0.800, 0.765), (0.825, 0.765), color=orange_dark)
    arrow((0.900, 0.765), (0.925, 0.765), color=orange_dark)
    label(0.712, 0.685, r"$\theta_1$", size=6.8, color=orange_dark)
    label(0.812, 0.685, r"$\theta_2$", size=6.8, color=orange_dark)
    label(0.858, 0.650, "fixed validation/test split", size=6.9, color=muted)

    arrow((0.288, 0.830), (0.662, 0.815), color=COLORS["neutral_dark"], dashed=True, rad=-0.08, width=1.2)
    label(0.480, 0.855, r"export compatible backbone weights $\theta_0$", size=7.0, color=COLORS["neutral_dark"])

    # (b) Shared station backbone, expanded in the style of architecture papers.
    panel_title(0.025, 0.565, "b", "Shared WaterQualityTransformer backbone")
    backbone_panel = FancyBboxPatch(
        (0.030, 0.090),
        0.455,
        0.430,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor="#FFFFFF",
        edgecolor=COLORS["neutral_mid"],
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(backbone_panel)
    waveform_stack(0.050, 0.275, 0.075, 0.100, color=blue, count=4, seed=13)
    label(0.087, 0.252, r"$x\in\mathbb{R}^{B\times T\times F}$", size=6.7, color=muted)
    block_specs = [
        (0.155, 0.285, 0.060, 0.080, "Instance\nNorm", COLORS["neutral_xlight"], COLORS["neutral_mid"]),
        (0.245, 0.285, 0.065, 0.080, "Temporal\nAdapter", COLORS["blue_xlight"], blue),
        (0.340, 0.285, 0.070, 0.080, "Time-feature\nEmbedding", COLORS["blue_xlight"], blue),
    ]
    for x, y, w, h, value, face, edge in block_specs:
        box(x, y, w, h, value, face=face, edge=edge, size=6.8, weight="bold")
    arrow((0.125, 0.325), (0.155, 0.325), color=blue_dark)
    arrow((0.215, 0.325), (0.245, 0.325), color=blue_dark)
    arrow((0.310, 0.325), (0.340, 0.325), color=blue_dark)
    token_row(0.342, 0.225, 0.067, 0.030, 4, colors=[COLORS["blue_light"], COLORS["gold_light"]])
    arrow((0.375, 0.285), (0.375, 0.255), color=blue_dark, scale=9)
    box(0.260, 0.130, 0.120, 0.070, "Transformer Encoder x3", face=COLORS["gold_xlight"], edge=gold_dark, size=7.2, weight="bold")
    box(0.405, 0.130, 0.055, 0.070, "Flatten\nHead", face=COLORS["orange_xlight"], edge=orange, size=6.8, weight="bold")
    arrow((0.375, 0.225), (0.320, 0.200), color=gold_dark, rad=0.08)
    arrow((0.380, 0.165), (0.405, 0.165), color=orange_dark)
    label(0.432, 0.112, "forecast / reconstruction", size=6.7, color=muted)

    box(0.055, 0.115, 0.155, 0.105, "", face=COLORS["neutral_xlight"], edge=COLORS["neutral_mid"], size=7.1, weight="bold")
    label(0.1325, 0.205, "Encoder layer", size=6.8, weight="bold")
    chip_x = [0.065, 0.100, 0.135, 0.170]
    chip_labels = ["MHA", "Add\nNorm", "FFN", "Add\nNorm"]
    for idx, (x, value) in enumerate(zip(chip_x, chip_labels)):
        face = COLORS["gold_light"] if idx in {0, 2} else COLORS["blue_light"]
        box(x, 0.135, 0.028, 0.045, value, face=face, edge=COLORS["neutral_mid"], size=5.4, linewidth=0.7, rounding=0.004)
        if idx < len(chip_x) - 1:
            arrow((x + 0.028, 0.158), (chip_x[idx + 1], 0.158), width=0.7, scale=6)

    # (c) Cross-station interaction, showing tensor structure instead of a text-only box.
    panel_title(0.520, 0.565, "c", "Cross-station interaction and gated reconstruction")
    cross_panel = FancyBboxPatch(
        (0.525, 0.090),
        0.445,
        0.430,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor="#FFFFFF",
        edgecolor=COLORS["neutral_mid"],
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(cross_panel)

    station_colors = [COLORS["blue_light"], COLORS["gold_light"], COLORS["orange_light"], COLORS["neutral_light"]]
    for row in range(4):
        token_row(0.550, 0.365 - row * 0.040, 0.090, 0.026, 4, colors=[station_colors[row]], masked=[2] if row == 2 else [])
    label(0.595, 0.410, "feature tokens by station", size=7.0, weight="bold")
    label(0.595, 0.190, r"$Z\in\mathbb{R}^{B\times N\times F\times d}$", size=6.8, color=muted)
    arrow((0.640, 0.315), (0.675, 0.315), color=gold_dark)

    box(0.675, 0.350, 0.085, 0.070, "Station ID\nembedding", face=COLORS["blue_xlight"], edge=blue, size=6.8, weight="bold")
    box(0.675, 0.235, 0.085, 0.070, "Coverage /\nmissing mask", face=COLORS["neutral_xlight"], edge=COLORS["neutral_mid"], size=6.6, weight="bold")
    box(0.790, 0.285, 0.095, 0.105, "Station-level\nMulti-Head Attention\n4 heads + pair bias", face=COLORS["gold_xlight"], edge=gold_dark, size=6.8, weight="bold")
    arrow((0.760, 0.385), (0.790, 0.355), color=gold_dark, rad=-0.08)
    arrow((0.760, 0.270), (0.790, 0.315), color=COLORS["neutral_dark"], rad=0.08)
    label(0.837, 0.265, r"$\Delta Z_{cross}$", size=6.8, color=gold_dark)

    box(0.790, 0.145, 0.095, 0.070, "Cross-delta\nHead", face=COLORS["orange_xlight"], edge=orange, size=6.8, weight="bold")
    box(0.900, 0.220, 0.050, 0.105, "Fusion\nGate\n$g$", face=COLORS["orange_xlight"], edge=orange_dark, size=6.8, weight="bold")
    arrow((0.837, 0.285), (0.837, 0.215), color=orange_dark)
    arrow((0.885, 0.180), (0.900, 0.245), color=orange_dark, rad=-0.16)
    arrow((0.640, 0.300), (0.900, 0.285), color=blue_dark, rad=-0.20)
    label(0.745, 0.232, "local path", size=6.6, color=blue_dark)
    label(0.940, 0.125, r"$\hat X_{cross}=\hat X_{local}+g\odot\Delta\hat X$", size=6.8, color=orange_dark, ha="right")

    return save_figure(fig, "fig2_ptl_framework_en", dpi=500)


def make_fig2_reference_style_architecture_v2() -> list[Path]:
    """Draw a code-aligned, paper-style overview with two expanded modules."""
    fig, ax = plt.subplots(figsize=(15.8, 9.2))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.018)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = TOKENS["ink"]
    muted = TOKENS["muted"]
    blue = COLORS["blue_mid"]
    blue_dark = COLORS["blue_dark"]
    gold = COLORS["gold_mid"]
    gold_dark = COLORS["gold_dark"]
    orange = COLORS["orange_mid"]
    orange_dark = COLORS["orange_dark"]

    def text(x, y, value, *, size=8.0, color=ink, weight="normal", ha="center", va="center", z=8):
        ax.text(
            x,
            y,
            value,
            fontsize=size,
            color=color,
            fontweight=weight,
            ha=ha,
            va=va,
            linespacing=1.12,
            zorder=z,
        )

    def box(
        x,
        y,
        w,
        h,
        value,
        *,
        face="#FFFFFF",
        edge=COLORS["neutral_mid"],
        size=7.2,
        weight="bold",
        linewidth=1.0,
        rounding=0.007,
        z=3,
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={rounding}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=z,
        )
        ax.add_patch(patch)
        if value:
            text(x + w / 2, y + h / 2, value, size=size, weight=weight, z=z + 1)
        return patch

    def arrow(start, end, *, color=COLORS["neutral_dark"], width=1.1, dashed=False, rad=0.0, scale=10, z=6):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=width,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def panel_title(x, y, panel, title):
        text(x, y, panel, size=10.5, weight="bold", ha="left")
        text(x + 0.027, y, title, size=10.5, weight="bold", ha="left")

    def waveform(x, y, w, h, *, color, count=4, seed=0, frame=True):
        rng = np.random.default_rng(seed)
        if frame:
            box(x, y, w, h, "", face="#FFFFFF", edge=COLORS["neutral_mid"], linewidth=0.85, rounding=0.004)
        xs = np.linspace(x + 0.006, x + w - 0.006, 72)
        for idx in range(count):
            center = y + h * (idx + 0.5) / count
            phase = rng.uniform(0, 2 * np.pi)
            values = (
                0.11 * h * np.sin(np.linspace(0, 3.5 * np.pi, len(xs)) + phase)
                + 0.032 * h * rng.normal(size=len(xs))
            )
            ax.plot(xs, center + values, color=color, linewidth=0.85, clip_on=True, zorder=5)

    def token_grid(x, y, cols, rows, cell_w, cell_h, *, row_colors, masked_rows=()):
        gap_x = 0.002
        gap_y = 0.004
        masked_rows = set(masked_rows)
        for row in range(rows):
            for col in range(cols):
                tx = x + col * (cell_w + gap_x)
                ty = y + (rows - row - 1) * (cell_h + gap_y)
                is_masked = row in masked_rows
                patch = FancyBboxPatch(
                    (tx, ty),
                    cell_w,
                    cell_h,
                    boxstyle="round,pad=0.001,rounding_size=0.0015",
                    facecolor="#FFFFFF" if is_masked else row_colors[row % len(row_colors)],
                    edgecolor=COLORS["neutral_mid"],
                    linewidth=0.65,
                    linestyle="--" if is_masked else "-",
                    zorder=4,
                )
                ax.add_patch(patch)

    def stage_card(x, title, subtitle, loss, *, face="#FFFFFF"):
        box(x, 0.716, 0.082, 0.096, f"{title}\n{subtitle}", face=face, edge=orange_dark, size=7.1)
        text(x + 0.041, 0.696, loss, size=6.25, color=orange_dark)

    # (a) End-to-end training and transfer workflow.
    panel_title(0.022, 0.966, "a", "PTL training and transfer workflow")
    source_panel = FancyBboxPatch(
        (0.026, 0.642),
        0.488,
        0.276,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=COLORS["blue_xlight"],
        edgecolor=blue_dark,
        linewidth=1.25,
        zorder=1,
    )
    target_panel = FancyBboxPatch(
        (0.535, 0.642),
        0.439,
        0.276,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=COLORS["orange_xlight"],
        edgecolor=orange_dark,
        linewidth=1.25,
        zorder=1,
    )
    ax.add_patch(source_panel)
    ax.add_patch(target_panel)
    text(0.047, 0.892, "Source domain: 18 monitoring stations", size=8.8, weight="bold", ha="left", color=blue_dark)
    text(0.556, 0.892, "Target domain: one station at a time", size=8.8, weight="bold", ha="left", color=orange_dark)

    waveform(0.050, 0.730, 0.067, 0.095, color=blue, count=4, seed=2)
    text(0.0835, 0.711, "weekly series", size=6.4, color=muted)
    box(0.143, 0.738, 0.069, 0.078, "Station mask\n$p=0.15$", face=COLORS["gold_xlight"], edge=gold_dark, size=7.0)
    box(0.241, 0.714, 0.118, 0.126, "Cross-station\nmasked pretraining\n(see panel c)", face="#FFFFFF", edge=blue_dark, size=7.3)
    box(0.391, 0.738, 0.077, 0.078, "Best backbone\n$\\theta_0$", face=COLORS["blue_light"], edge=blue_dark, size=7.0)
    arrow((0.117, 0.777), (0.143, 0.777), color=blue_dark)
    arrow((0.212, 0.777), (0.241, 0.777), color=blue_dark)
    arrow((0.359, 0.777), (0.391, 0.777), color=blue_dark)
    text(0.1775, 0.682, "70% clean windows; masking applied to the remaining windows", size=6.2, color=muted)
    text(
        0.301,
        0.659,
        r"$\mathcal{L}_{pre}=\mathcal{L}_{local}+0.5\mathcal{L}_{cross-all}+\mathcal{L}_{cross-mask}$",
        size=6.8,
        color=blue_dark,
    )

    waveform(0.556, 0.738, 0.052, 0.078, color=orange, count=4, seed=8)
    text(0.582, 0.711, "target station", size=6.3, color=muted)
    stage_card(0.632, "Stage 1", "Weekly: 56 d -> 8", r"$\lambda_{NSE}=0.05$")
    stage_card(0.741, "Stage 2", "4-day: 32 d -> 8", r"$\lambda_{NSE}=0.10$")
    stage_card(0.850, "Stage 3", "Daily: 12 d -> 12", r"$\lambda_{NSE}=0.15$")
    arrow((0.608, 0.764), (0.632, 0.764), color=orange_dark)
    arrow((0.714, 0.764), (0.741, 0.764), color=orange_dark)
    arrow((0.823, 0.764), (0.850, 0.764), color=orange_dark)
    arrow((0.932, 0.764), (0.956, 0.764), color=orange_dark)
    waveform(0.950, 0.739, 0.020, 0.050, color=orange_dark, count=1, seed=10, frame=False)
    text(0.960, 0.711, r"$\hat{y}_{t+1}$", size=7.0, color=orange_dark)
    text(0.727, 0.665, r"stagewise handoff: $\theta_0\rightarrow\theta_1\rightarrow\theta_2\rightarrow\theta_3$", size=6.8, color=orange_dark)
    arrow((0.429, 0.816), (0.675, 0.815), color=COLORS["neutral_dark"], dashed=True, rad=-0.08, width=1.2)
    text(0.548, 0.850, "initialize the same backbone architecture", size=6.7, color=COLORS["neutral_dark"])

    # (b) Shared single-station backbone.
    panel_title(0.022, 0.595, "b", "Shared single-station Transformer backbone")
    backbone_panel = FancyBboxPatch(
        (0.026, 0.055),
        0.474,
        0.495,
        boxstyle="round,pad=0.008,rounding_size=0.011",
        facecolor="#FFFFFF",
        edgecolor=COLORS["neutral_mid"],
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(backbone_panel)
    waveform(0.047, 0.335, 0.064, 0.094, color=blue, count=4, seed=14)
    text(0.079, 0.314, r"$x\in\mathbb{R}^{B\times T\times4}$", size=6.4, color=muted)
    box(0.133, 0.344, 0.065, 0.074, "Instance\nnormalization", face=COLORS["neutral_xlight"], edge=COLORS["neutral_mid"], size=6.6)
    box(0.220, 0.332, 0.077, 0.098, "Temporal adapter\nDWConv1D ($k=5$)\n+ residual", face=COLORS["blue_xlight"], edge=blue, size=6.5)
    box(0.320, 0.332, 0.077, 0.098, "Feature-wise\nembedding\n$4$ tokens, $d=256$", face=COLORS["blue_xlight"], edge=blue, size=6.5)
    arrow((0.111, 0.381), (0.133, 0.381), color=blue_dark)
    arrow((0.198, 0.381), (0.220, 0.381), color=blue_dark)
    arrow((0.297, 0.381), (0.320, 0.381), color=blue_dark)
    token_grid(
        0.414,
        0.350,
        cols=1,
        rows=4,
        cell_w=0.030,
        cell_h=0.013,
        row_colors=[COLORS["blue_light"], COLORS["gold_light"], COLORS["orange_light"], COLORS["neutral_light"]],
    )
    arrow((0.397, 0.381), (0.414, 0.381), color=blue_dark)

    box(0.300, 0.204, 0.130, 0.077, "Transformer encoder x3\n8-head feature attention", face=COLORS["gold_xlight"], edge=gold_dark, size=7.0)
    arrow((0.429, 0.350), (0.365, 0.281), color=gold_dark, rad=0.12)
    box(0.300, 0.103, 0.075, 0.061, "Flatten +\nlinear head", face=COLORS["orange_xlight"], edge=orange, size=6.6)
    box(0.401, 0.103, 0.074, 0.061, "Inverse instance\nnormalization", face=COLORS["neutral_xlight"], edge=COLORS["neutral_mid"], size=6.3)
    arrow((0.365, 0.204), (0.338, 0.164), color=orange_dark, rad=0.10)
    arrow((0.375, 0.133), (0.401, 0.133), color=orange_dark)
    text(0.438, 0.083, r"$\hat y\in\mathbb{R}^{B\times1\times4}$", size=6.5, color=orange_dark)

    box(0.052, 0.096, 0.208, 0.146, "", face=COLORS["neutral_xlight"], edge=COLORS["neutral_mid"], linewidth=0.9)
    text(0.156, 0.222, "One encoder layer", size=6.9, weight="bold")
    chip_specs = [
        (0.064, "MHA", COLORS["gold_light"]),
        (0.108, "Add &\nNorm", COLORS["blue_light"]),
        (0.152, "FFN", COLORS["gold_light"]),
        (0.196, "Add &\nNorm", COLORS["blue_light"]),
    ]
    for idx, (x, value, face) in enumerate(chip_specs):
        box(x, 0.132, 0.037, 0.055, value, face=face, edge=COLORS["neutral_mid"], size=5.5, linewidth=0.7, rounding=0.003)
        if idx < len(chip_specs) - 1:
            arrow((x + 0.037, 0.159), (chip_specs[idx + 1][0], 0.159), width=0.65, scale=6)
    text(0.156, 0.113, "residual connections around MHA and FFN", size=5.7, color=muted)

    # (c) Detailed masked pretraining branch.
    panel_title(0.522, 0.595, "c", "Cross-station masked pretraining")
    cross_panel = FancyBboxPatch(
        (0.526, 0.055),
        0.448,
        0.495,
        boxstyle="round,pad=0.008,rounding_size=0.011",
        facecolor="#FFFFFF",
        edgecolor=COLORS["neutral_mid"],
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(cross_panel)

    station_colors = [COLORS["blue_light"], COLORS["gold_light"], COLORS["orange_light"], COLORS["neutral_light"]]
    token_grid(0.547, 0.348, cols=5, rows=4, cell_w=0.017, cell_h=0.017, row_colors=station_colors, masked_rows=[2])
    text(0.594, 0.443, "station windows", size=6.8, weight="bold")
    text(0.594, 0.326, r"$X\in\mathbb{R}^{B\times18\times168\times4}$", size=6.1, color=muted)
    text(0.594, 0.303, "entire station window masked", size=5.9, color=orange_dark)
    box(0.659, 0.348, 0.083, 0.082, "Shared backbone\nencoder $\\theta_0$\nfor every station", face=COLORS["blue_xlight"], edge=blue_dark, size=6.5)
    arrow((0.636, 0.389), (0.659, 0.389), color=blue_dark)
    text(0.7005, 0.326, r"$Z\in\mathbb{R}^{B\times18\times4\times256}$", size=5.9, color=muted)

    box(0.770, 0.420, 0.077, 0.059, "Shared decoder\n(local path)", face=COLORS["blue_xlight"], edge=blue, size=6.2)
    arrow((0.742, 0.405), (0.770, 0.449), color=blue_dark, rad=-0.10)
    box(0.875, 0.420, 0.074, 0.059, r"$\hat X_{local}$", face="#FFFFFF", edge=blue, size=7.0)
    arrow((0.847, 0.449), (0.875, 0.449), color=blue_dark)

    box(0.770, 0.326, 0.077, 0.059, "Station ID\nembedding", face=COLORS["blue_xlight"], edge=blue, size=6.2)
    arrow((0.742, 0.376), (0.770, 0.355), color=blue_dark, rad=0.10)
    box(
        0.870,
        0.311,
        0.079,
        0.088,
        "Reshape by feature\n" + r"$B\cdot4\times18\times256$",
        face=COLORS["neutral_xlight"],
        edge=COLORS["neutral_mid"],
        size=6.1,
    )
    arrow((0.847, 0.355), (0.870, 0.355), color=COLORS["neutral_dark"])

    box(0.659, 0.203, 0.083, 0.070, "Station-availability\nkey-padding mask", face=COLORS["neutral_xlight"], edge=COLORS["neutral_mid"], size=6.1)
    box(0.770, 0.190, 0.100, 0.096, "Cross-station block\n4-head MHA + AddNorm\nFFN + AddNorm", face=COLORS["gold_xlight"], edge=gold_dark, size=6.2)
    arrow((0.909, 0.311), (0.837, 0.286), color=gold_dark, rad=-0.12)
    arrow((0.742, 0.238), (0.770, 0.238), color=COLORS["neutral_dark"])
    box(0.896, 0.203, 0.053, 0.070, "Context\n$C$", face=COLORS["gold_xlight"], edge=gold_dark, size=6.7)
    arrow((0.870, 0.238), (0.896, 0.238), color=gold_dark)

    box(0.770, 0.093, 0.100, 0.061, r"Fuse: $Z+\alpha C$" "\n" r"$\alpha=\tanh(s)$", face=COLORS["orange_xlight"], edge=orange_dark, size=6.3)
    arrow((0.922, 0.203), (0.852, 0.154), color=orange_dark, rad=0.10)
    arrow((0.700, 0.348), (0.790, 0.154), color=blue_dark, rad=-0.24)
    box(0.896, 0.093, 0.053, 0.061, r"$\hat X_{cross}$", face="#FFFFFF", edge=orange_dark, size=6.8)
    arrow((0.870, 0.124), (0.896, 0.124), color=orange_dark)

    text(0.550, 0.238, "Two reconstruction paths", size=6.8, weight="bold", ha="left")
    text(0.550, 0.207, r"$\mathcal{L}_{local}$: local output, all valid positions", size=5.9, color=blue_dark, ha="left")
    text(0.550, 0.180, r"$\mathcal{L}_{cross-all}$: fused output, all valid positions", size=5.9, color=gold_dark, ha="left")
    text(0.550, 0.153, r"$\mathcal{L}_{cross-mask}$: fused output, masked positions", size=5.9, color=orange_dark, ha="left")
    text(0.550, 0.106, "Only the shared backbone is transferred to target stations", size=6.1, weight="bold", color=blue_dark, ha="left")

    return save_figure(fig, "fig2_ptl_framework_en_v2", dpi=500)


def make_fig2_reference_style_architecture_v3() -> list[Path]:
    """Draw a compact three-band PTL architecture with orthogonal data flow."""
    fig, ax = plt.subplots(figsize=(15.8, 8.6))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.018)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = TOKENS["ink"]
    muted = TOKENS["muted"]
    blue = COLORS["blue_mid"]
    blue_dark = COLORS["blue_dark"]
    gold_dark = COLORS["gold_dark"]
    orange = COLORS["orange_mid"]
    orange_dark = COLORS["orange_dark"]
    neutral = COLORS["neutral_mid"]

    def text(x, y, value, *, size=7.4, color=ink, weight="normal", ha="center", va="center", z=8):
        ax.text(
            x,
            y,
            value,
            fontsize=size,
            color=color,
            fontweight=weight,
            ha=ha,
            va=va,
            linespacing=1.08,
            zorder=z,
        )

    def box(
        x,
        y,
        w,
        h,
        value,
        *,
        face="#FFFFFF",
        edge=neutral,
        size=6.8,
        weight="bold",
        linewidth=1.0,
        rounding=0.006,
        z=3,
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.003,rounding_size={rounding}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=z,
        )
        ax.add_patch(patch)
        if value:
            text(x + w / 2, y + h / 2, value, size=size, weight=weight, z=z + 1)
        return patch

    def arrow(start, end, *, color=COLORS["neutral_dark"], width=1.05, dashed=False, scale=9, z=6):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=width,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle="arc3,rad=0",
            zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def elbow_arrow(start, end, *, direction="vertical-first", color=COLORS["neutral_dark"], width=1.05, dashed=False):
        x1, y1 = start
        x2, y2 = end
        linestyle = "--" if dashed else "-"
        if direction == "vertical-first":
            turn = (x1, y2)
            ax.plot([x1, turn[0]], [y1, turn[1]], color=color, linewidth=width, linestyle=linestyle, zorder=5)
            arrow(turn, end, color=color, width=width, dashed=dashed, scale=9)
        else:
            turn = (x2, y1)
            ax.plot([x1, turn[0]], [y1, turn[1]], color=color, linewidth=width, linestyle=linestyle, zorder=5)
            arrow(turn, end, color=color, width=width, dashed=dashed, scale=9)

    def panel(x, y, w, h, *, face="#FFFFFF", edge=neutral):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.05,
            zorder=1,
        )
        ax.add_patch(patch)
        return patch

    def panel_title(y, tag, title):
        text(0.025, y, tag, size=10.2, weight="bold", ha="left")
        text(0.052, y, title, size=10.2, weight="bold", ha="left")

    def waveform(x, y, w, h, *, color, count=4, seed=0):
        rng = np.random.default_rng(seed)
        box(x, y, w, h, "", face="#FFFFFF", edge=neutral, linewidth=0.8, rounding=0.003)
        xs = np.linspace(x + 0.005, x + w - 0.005, 64)
        for idx in range(count):
            center = y + h * (idx + 0.5) / count
            phase = rng.uniform(0, 2 * np.pi)
            values = (
                0.105 * h * np.sin(np.linspace(0, 3.2 * np.pi, len(xs)) + phase)
                + 0.030 * h * rng.normal(size=len(xs))
            )
            ax.plot(xs, center + values, color=color, linewidth=0.82, clip_on=True, zorder=5)

    def station_matrix(x, y, *, masked_row=2):
        colors = [COLORS["blue_light"], COLORS["gold_light"], COLORS["orange_light"], COLORS["neutral_light"]]
        rows = 4
        cols = 5
        cell_w = 0.012
        cell_h = 0.013
        for row in range(rows):
            for col in range(cols):
                tx = x + col * 0.0135
                ty = y + (rows - row - 1) * 0.017
                is_masked = row == masked_row
                patch = FancyBboxPatch(
                    (tx, ty),
                    cell_w,
                    cell_h,
                    boxstyle="round,pad=0.001,rounding_size=0.001",
                    facecolor="#FFFFFF" if is_masked else colors[row],
                    edgecolor=neutral,
                    linewidth=0.6,
                    linestyle="--" if is_masked else "-",
                    zorder=4,
                )
                ax.add_patch(patch)

    # (a) Compact end-to-end workflow.
    panel_title(0.968, "a", "PTL training and transfer workflow")
    panel(0.025, 0.690, 0.465, 0.235, face=COLORS["blue_xlight"], edge=blue_dark)
    panel(0.510, 0.690, 0.465, 0.235, face=COLORS["orange_xlight"], edge=orange_dark)
    text(0.047, 0.900, "Source-domain masked pretraining", size=8.5, color=blue_dark, weight="bold", ha="left")
    text(0.532, 0.900, "Target-domain progressive adaptation", size=8.5, color=orange_dark, weight="bold", ha="left")

    waveform(0.047, 0.775, 0.057, 0.074, color=blue, count=4, seed=2)
    box(0.128, 0.778, 0.067, 0.068, "Station mask\n$p=0.15$", face=COLORS["gold_xlight"], edge=gold_dark, size=6.8)
    box(0.220, 0.765, 0.105, 0.094, "Cross-station\nmasked pretraining\n(panel c)", face="#FFFFFF", edge=blue_dark, size=6.9)
    box(0.357, 0.778, 0.075, 0.068, "Backbone\n$\\theta_0$", face=COLORS["blue_light"], edge=blue_dark, size=6.8)
    arrow((0.104, 0.812), (0.128, 0.812), color=blue_dark)
    arrow((0.195, 0.812), (0.220, 0.812), color=blue_dark)
    arrow((0.325, 0.812), (0.357, 0.812), color=blue_dark)
    text(0.075, 0.752, "18 stations", size=6.1, color=muted)
    text(0.1615, 0.752, "station-level", size=6.1, color=muted)
    text(0.273, 0.721, r"$\mathcal{L}_{pre}=\mathcal{L}_{local}+0.5\mathcal{L}_{cross-all}+\mathcal{L}_{cross-mask}$", size=6.4, color=blue_dark)

    waveform(0.532, 0.779, 0.050, 0.068, color=orange, count=4, seed=8)
    stage_specs = [
        (0.610, "Stage 1", "Weekly\n56 d -> 8", r"$\lambda=0.05$"),
        (0.715, "Stage 2", "4-day\n32 d -> 8", r"$\lambda=0.10$"),
        (0.820, "Stage 3", "Daily\n12 d -> 12", r"$\lambda=0.15$"),
    ]
    for x, title, subtitle, loss in stage_specs:
        box(x, 0.765, 0.080, 0.094, f"{title}\n{subtitle}", face="#FFFFFF", edge=orange_dark, size=6.8)
        text(x + 0.040, 0.742, loss, size=6.0, color=orange_dark)
    box(0.925, 0.782, 0.035, 0.060, r"$\hat y_{t+1}$", face="#FFFFFF", edge=orange_dark, size=7.0)
    arrow((0.582, 0.812), (0.610, 0.812), color=orange_dark)
    arrow((0.690, 0.812), (0.715, 0.812), color=orange_dark)
    arrow((0.795, 0.812), (0.820, 0.812), color=orange_dark)
    arrow((0.900, 0.812), (0.925, 0.812), color=orange_dark)
    text(0.557, 0.752, "target station", size=6.1, color=muted)
    text(0.767, 0.713, r"stagewise handoff: $\theta_0\rightarrow\theta_1\rightarrow\theta_2\rightarrow\theta_3$", size=6.2, color=orange_dark)

    # Backbone initialization travels right, then down into Stage 1.
    ax.plot([0.395, 0.650], [0.870, 0.870], color=COLORS["neutral_dark"], linewidth=1.0, linestyle="--", zorder=5)
    arrow((0.650, 0.870), (0.650, 0.859), color=COLORS["neutral_dark"], width=1.0, dashed=True, scale=8)
    text(0.520, 0.882, "initialize compatible backbone weights", size=6.2, color=COLORS["neutral_dark"])

    # (b) One compact left-to-right backbone pipeline.
    panel_title(0.650, "b", "Shared single-station Transformer backbone")
    panel(0.025, 0.405, 0.950, 0.205, face="#FFFFFF", edge=neutral)
    waveform(0.045, 0.490, 0.060, 0.072, color=blue, count=4, seed=14)
    modules = [
        (0.128, 0.490, 0.082, 0.072, "Instance\nnorm", COLORS["neutral_xlight"], neutral),
        (0.233, 0.480, 0.100, 0.092, "Temporal adapter\nDWConv1D $k=5$\n+ residual", COLORS["blue_xlight"], blue),
        (0.356, 0.480, 0.100, 0.092, "Feature embedding\n4 indicator tokens\n$d=256$", COLORS["blue_xlight"], blue),
        (0.489, 0.480, 0.125, 0.092, "Transformer encoder x3\n8-head feature attention", COLORS["gold_xlight"], gold_dark),
        (0.648, 0.490, 0.078, 0.072, "Flatten +\nlinear head", COLORS["orange_xlight"], orange),
        (0.758, 0.490, 0.100, 0.072, "Inverse instance\nnormalization", COLORS["neutral_xlight"], neutral),
        (0.891, 0.490, 0.060, 0.072, r"$\hat y$" + "\n" + r"$B\times1\times4$", "#FFFFFF", orange_dark),
    ]
    for x, y, w, h, value, face, edge in modules:
        box(x, y, w, h, value, face=face, edge=edge, size=6.5)
    arrow((0.105, 0.526), (0.128, 0.526), color=blue_dark)
    arrow((0.210, 0.526), (0.233, 0.526), color=blue_dark)
    arrow((0.333, 0.526), (0.356, 0.526), color=blue_dark)
    arrow((0.456, 0.526), (0.489, 0.526), color=blue_dark)
    arrow((0.614, 0.526), (0.648, 0.526), color=orange_dark)
    arrow((0.726, 0.526), (0.758, 0.526), color=orange_dark)
    arrow((0.858, 0.526), (0.891, 0.526), color=orange_dark)
    text(0.075, 0.466, r"$x\in\mathbb{R}^{B\times T\times4}$", size=6.1, color=muted)

    # Encoder detail sits directly below the encoder, without introducing a second flow direction.
    chip_y = 0.426
    chip_xs = [0.489, 0.521, 0.553, 0.585]
    chip_labels = ["MHA", "Add &\nNorm", "FFN", "Add &\nNorm"]
    for idx, (x, value) in enumerate(zip(chip_xs, chip_labels)):
        face = COLORS["gold_light"] if idx in {0, 2} else COLORS["blue_light"]
        box(x, chip_y, 0.029, 0.035, value, face=face, edge=neutral, size=4.9, linewidth=0.65, rounding=0.002)
        if idx < 3:
            arrow((x + 0.029, chip_y + 0.0175), (chip_xs[idx + 1], chip_y + 0.0175), width=0.6, scale=5)
    arrow((0.5515, 0.480), (0.5515, 0.463), color=gold_dark, dashed=True, width=0.8, scale=7)
    text(0.628, 0.444, "one encoder layer", size=5.8, color=muted, ha="left")

    # (c) Cross-station pretraining with a local row and a cross-station row.
    panel_title(0.365, "c", "Cross-station masked pretraining")
    panel(0.025, 0.040, 0.950, 0.285, face="#FFFFFF", edge=neutral)

    station_matrix(0.047, 0.220, masked_row=2)
    text(0.080, 0.297, "multi-station windows", size=6.4, weight="bold")
    text(0.080, 0.200, r"$B\times18\times168\times4$", size=5.9, color=muted)
    box(0.137, 0.220, 0.072, 0.064, "Station-level\nmask", face=COLORS["gold_xlight"], edge=gold_dark, size=6.4)
    box(0.237, 0.210, 0.100, 0.084, "Shared backbone\nencoder $\\theta_0$\nfor each station", face=COLORS["blue_xlight"], edge=blue_dark, size=6.5)
    box(0.365, 0.220, 0.065, 0.064, "Encoded $Z$\n" + r"$B\times18\times4\times256$", face=COLORS["blue_light"], edge=blue_dark, size=6.0)
    arrow((0.114, 0.252), (0.137, 0.252), color=blue_dark)
    arrow((0.209, 0.252), (0.237, 0.252), color=blue_dark)
    arrow((0.337, 0.252), (0.365, 0.252), color=blue_dark)
    text(0.173, 0.198, "mask entire station window", size=5.8, color=orange_dark)

    # Local reconstruction row.
    box(0.474, 0.220, 0.082, 0.064, "Shared decoder\nlocal path", face=COLORS["blue_xlight"], edge=blue, size=6.3)
    box(0.585, 0.220, 0.063, 0.064, r"$\hat X_{local}$", face="#FFFFFF", edge=blue, size=6.9)
    box(0.678, 0.220, 0.092, 0.064, r"$\mathcal{L}_{local}$" + "\nall valid positions", face=COLORS["neutral_xlight"], edge=neutral, size=6.0)
    arrow((0.430, 0.252), (0.474, 0.252), color=blue_dark)
    arrow((0.556, 0.252), (0.585, 0.252), color=blue_dark)
    arrow((0.648, 0.252), (0.678, 0.252), color=blue_dark)
    text(0.452, 0.277, "local branch", size=5.8, color=blue_dark)

    # Cross-station row starts by moving down from Z, then proceeds only left to right.
    box(0.455, 0.105, 0.080, 0.064, "+ Station ID\nembedding", face=COLORS["blue_xlight"], edge=blue, size=6.1)
    box(0.558, 0.105, 0.092, 0.064, "Reshape by feature\n" + r"$B\cdot4\times18\times256$", face=COLORS["neutral_xlight"], edge=neutral, size=5.9)
    box(0.678, 0.095, 0.112, 0.084, "Cross-station block\n4-head MHA + AddNorm\nFFN + AddNorm", face=COLORS["gold_xlight"], edge=gold_dark, size=6.0)
    box(0.814, 0.105, 0.075, 0.064, r"Fuse $Z+\alpha C$" "\n" r"$\alpha=\tanh(s)$", face=COLORS["orange_xlight"], edge=orange_dark, size=6.0)
    box(0.912, 0.105, 0.047, 0.064, "Decode\n" + r"$\hat X_{cross}$", face="#FFFFFF", edge=orange_dark, size=5.9)
    elbow_arrow((0.3975, 0.220), (0.455, 0.137), direction="vertical-first", color=blue_dark)
    arrow((0.535, 0.137), (0.558, 0.137), color=COLORS["neutral_dark"])
    arrow((0.650, 0.137), (0.678, 0.137), color=gold_dark)
    arrow((0.790, 0.137), (0.814, 0.137), color=gold_dark)
    arrow((0.889, 0.137), (0.912, 0.137), color=orange_dark)

    box(0.685, 0.184, 0.098, 0.025, "Station-availability mask", face=COLORS["neutral_xlight"], edge=neutral, size=5.1)
    arrow((0.734, 0.184), (0.734, 0.179), color=COLORS["neutral_dark"], width=0.9, scale=7)
    text(0.734, 0.081, "context C", size=5.8, color=gold_dark)
    box(0.870, 0.048, 0.089, 0.038, r"$0.5\mathcal{L}_{cross-all}+\mathcal{L}_{cross-mask}$", face=COLORS["orange_xlight"], edge=orange_dark, size=5.5)
    arrow((0.9355, 0.105), (0.9355, 0.086), color=orange_dark, width=0.9, scale=8)
    text(0.047, 0.078, "70% clean windows; station masking is sampled in the remaining windows", size=5.9, color=muted, ha="left")
    text(0.455, 0.064, "Only the shared backbone is transferred; cross-station modules are pretraining-only", size=6.0, color=blue_dark, weight="bold", ha="left")

    return save_figure(fig, "fig2_ptl_framework_en_v3", dpi=500)


def make_fig2_ptl_workflow_split_v4() -> list[Path]:
    """Draw the PTL training workflow as a standalone transfer-process figure."""
    fig, ax = plt.subplots(figsize=(15.8, 5.2))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.975, bottom=0.035)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = TOKENS["ink"]
    muted = TOKENS["muted"]
    blue = COLORS["blue_mid"]
    blue_dark = COLORS["blue_dark"]
    gold_dark = COLORS["gold_dark"]
    orange = COLORS["orange_mid"]
    orange_dark = COLORS["orange_dark"]
    neutral = COLORS["neutral_mid"]

    def text(x, y, value, *, size=7.6, color=ink, weight="normal", ha="center", va="center", z=8):
        ax.text(
            x,
            y,
            value,
            fontsize=size,
            color=color,
            fontweight=weight,
            ha=ha,
            va=va,
            linespacing=1.10,
            zorder=z,
        )

    def box(x, y, w, h, value, *, face="#FFFFFF", edge=neutral, size=7.0, linewidth=1.0, z=3):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.009",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=z,
        )
        ax.add_patch(patch)
        text(x + w / 2, y + h / 2, value, size=size, weight="bold", z=z + 1)
        return patch

    def arrow(start, end, *, color=COLORS["neutral_dark"], width=1.15, dashed=False, scale=10, z=6):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=width,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle="arc3,rad=0",
            zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def panel(x, y, w, h, *, face, edge):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.014",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.25,
            zorder=1,
        )
        ax.add_patch(patch)

    def waveform(x, y, w, h, *, color, count=4, seed=0):
        rng = np.random.default_rng(seed)
        xs = np.linspace(x + 0.004, x + w - 0.004, 72)
        for idx in range(count):
            center = y + h * (idx + 0.5) / count
            phase = rng.uniform(0, 2 * np.pi)
            values = (
                0.10 * h * np.sin(np.linspace(0, 3.4 * np.pi, len(xs)) + phase)
                + 0.025 * h * rng.normal(size=len(xs))
            )
            ax.plot(xs, center + values, color=color, linewidth=0.85, clip_on=True, zorder=5)

    def dataset_icon(x, y, w, h, *, color, label, seed):
        box(x, y, w, h, "", face="#FFFFFF", edge=color, linewidth=1.0)
        waveform(x + 0.008, y + 0.035, w - 0.016, h - 0.070, color=color, count=4, seed=seed)
        text(x + w / 2, y + 0.015, label, size=6.1, color=color, weight="bold")

    # Two adjacent phases keep the transfer boundary explicit without breaking the left-to-right flow.
    panel(0.020, 0.150, 0.430, 0.745, face=COLORS["blue_xlight"], edge=blue_dark)
    panel(0.470, 0.150, 0.510, 0.745, face=COLORS["orange_xlight"], edge=orange_dark)
    text(0.040, 0.855, "Source-domain cross-station pretraining", size=10.5, color=blue_dark, weight="bold", ha="left")
    text(0.490, 0.855, "Target-domain progressive adaptation", size=10.5, color=orange_dark, weight="bold", ha="left")

    # Source-domain pretraining.
    dataset_icon(0.040, 0.485, 0.075, 0.205, color=blue, label="N stations", seed=2)
    box(0.140, 0.505, 0.072, 0.165, "Aligned\n168-step\nwindows", face="#FFFFFF", edge=blue_dark, size=7.0)
    box(0.237, 0.505, 0.070, 0.165, "Station\nmask\n$r=0.15$", face=COLORS["gold_xlight"], edge=gold_dark, size=7.0)
    box(0.332, 0.485, 0.090, 0.205, "Cross-station\nmasked\npretraining", face="#FFFFFF", edge=blue_dark, size=7.4)
    arrow((0.115, 0.588), (0.140, 0.588), color=blue_dark)
    arrow((0.212, 0.588), (0.237, 0.588), color=blue_dark)
    arrow((0.307, 0.588), (0.332, 0.588), color=blue_dark)
    text(0.230, 0.410, "Gap-aware window filtering and train-only scaling", size=6.6, color=muted)
    text(
        0.230,
        0.315,
        r"$\mathcal{L}_{pre}=\mathcal{L}_{local}+0.5\mathcal{L}_{cross-all}+\mathcal{L}_{cross-mask}$",
        size=7.0,
        color=blue_dark,
    )
    box(0.340, 0.205, 0.075, 0.090, "Export\n$\theta_0$", face=COLORS["blue_light"], edge=blue_dark, size=7.2)
    arrow((0.377, 0.485), (0.377, 0.295), color=blue_dark)

    # Target-domain progressive transfer.
    dataset_icon(0.490, 0.485, 0.070, 0.205, color=orange, label="Target station", seed=8)
    stage_specs = [
        (0.585, "Stage 1", "Weekly", "56 d -> 8", r"$\lambda_{NSE}=0.05$"),
        (0.700, "Stage 2", "4-day", "32 d -> 8", r"$\lambda_{NSE}=0.10$"),
        (0.815, "Stage 3", "Daily", "12 d -> 12", r"$\lambda_{NSE}=0.15$"),
    ]
    for x, title, resolution, window, loss in stage_specs:
        box(x, 0.485, 0.090, 0.205, f"{title}\n{resolution}\n{window}", face="#FFFFFF", edge=orange_dark, size=7.1)
        text(x + 0.045, 0.440, loss, size=6.2, color=orange_dark)
    box(0.930, 0.515, 0.035, 0.145, r"$\hat y_{t+1}$", face="#FFFFFF", edge=orange_dark, size=8.4)
    arrow((0.560, 0.588), (0.585, 0.588), color=orange_dark)
    arrow((0.675, 0.588), (0.700, 0.588), color=orange_dark)
    arrow((0.790, 0.588), (0.815, 0.588), color=orange_dark)
    arrow((0.905, 0.588), (0.930, 0.588), color=orange_dark)
    text(0.755, 0.330, r"Progressive parameter handoff: $\theta_0\rightarrow\theta_1\rightarrow\theta_2\rightarrow\theta_3$", size=7.1, color=orange_dark)
    text(0.755, 0.245, "Each stage is fine-tuned and validated at the corresponding temporal resolution", size=6.6, color=muted)

    # Knowledge transfer crosses the phase boundary once, then enters Stage 1 from above.
    ax.plot([0.415, 0.630], [0.250, 0.250], color=COLORS["neutral_dark"], linewidth=1.15, linestyle="--", zorder=5)
    ax.plot([0.630, 0.630], [0.250, 0.485], color=COLORS["neutral_dark"], linewidth=1.15, linestyle="--", zorder=5)
    arrow((0.630, 0.485), (0.630, 0.500), color=COLORS["neutral_dark"], width=1.15, dashed=True, scale=9)
    text(0.515, 0.270, "backbone initialization", size=6.4, color=COLORS["neutral_dark"])

    text(0.500, 0.075, "Solid arrows: data/training flow    Dashed arrow: transferred backbone weights", size=6.5, color=muted)
    return save_figure(fig, "fig2_ptl_workflow_en_v4", dpi=500)


def make_fig3_ptl_model_architecture_split_v4() -> list[Path]:
    """Draw the pretraining model and shared forecasting backbone as a standalone figure."""
    fig, ax = plt.subplots(figsize=(15.8, 7.2))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.980, bottom=0.025)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = TOKENS["ink"]
    muted = TOKENS["muted"]
    blue = COLORS["blue_mid"]
    blue_dark = COLORS["blue_dark"]
    gold_dark = COLORS["gold_dark"]
    orange = COLORS["orange_mid"]
    orange_dark = COLORS["orange_dark"]
    neutral = COLORS["neutral_mid"]

    def text(x, y, value, *, size=7.3, color=ink, weight="normal", ha="center", va="center", z=8):
        ax.text(
            x,
            y,
            value,
            fontsize=size,
            color=color,
            fontweight=weight,
            ha=ha,
            va=va,
            linespacing=1.08,
            zorder=z,
        )

    def box(x, y, w, h, value, *, face="#FFFFFF", edge=neutral, size=6.7, linewidth=1.0, z=3):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.003,rounding_size=0.007",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=z,
        )
        ax.add_patch(patch)
        text(x + w / 2, y + h / 2, value, size=size, weight="bold", z=z + 1)
        return patch

    def arrow(start, end, *, color=COLORS["neutral_dark"], width=1.05, dashed=False, scale=9, z=6):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=width,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle="arc3,rad=0",
            zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def elbow(start, end, *, color=COLORS["neutral_dark"], width=1.05, dashed=False):
        x1, y1 = start
        x2, y2 = end
        style = "--" if dashed else "-"
        ax.plot([x1, x1], [y1, y2], color=color, linewidth=width, linestyle=style, zorder=5)
        arrow((x1, y2), (x2, y2), color=color, width=width, dashed=dashed)

    def panel(x, y, w, h, *, face="#FFFFFF", edge=neutral):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.011",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.05,
            zorder=1,
        )
        ax.add_patch(patch)

    def station_matrix(x, y, *, rows=5, cols=6, masked_row=2):
        row_colors = [COLORS["blue_light"], COLORS["gold_light"], COLORS["orange_light"], COLORS["neutral_light"], COLORS["olive_base"]]
        for row in range(rows):
            for col in range(cols):
                tx = x + col * 0.0120
                ty = y + (rows - row - 1) * 0.0170
                masked = row == masked_row
                patch = FancyBboxPatch(
                    (tx, ty),
                    0.0105,
                    0.0130,
                    boxstyle="round,pad=0.001,rounding_size=0.001",
                    facecolor="#FFFFFF" if masked else row_colors[row],
                    edgecolor=neutral,
                    linewidth=0.55,
                    linestyle="--" if masked else "-",
                    zorder=4,
                )
                ax.add_patch(patch)

    # (a) The complete cross-station pretraining model.
    text(0.025, 0.965, "a", size=10.2, weight="bold", ha="left")
    text(0.052, 0.965, "Cross-station masked pretraining model", size=10.2, weight="bold", ha="left")
    panel(0.025, 0.405, 0.950, 0.510, face="#FFFFFF", edge=neutral)

    station_matrix(0.045, 0.690)
    text(0.081, 0.790, "Multi-station windows", size=6.7, weight="bold")
    text(0.081, 0.665, r"$X\in\mathbb{R}^{B\times N\times168\times4}$", size=6.0, color=muted)
    box(0.140, 0.690, 0.075, 0.085, "Station-level\nmask", face=COLORS["gold_xlight"], edge=gold_dark, size=6.7)
    box(0.245, 0.675, 0.110, 0.115, "Shared station\nencoder $\theta_0$\n(applied to each station)", face=COLORS["blue_xlight"], edge=blue_dark, size=6.7)
    box(0.385, 0.690, 0.075, 0.085, "Encoded $Z$\n" + r"$B\times N\times4\times d$", face=COLORS["blue_light"], edge=blue_dark, size=6.2)
    arrow((0.117, 0.733), (0.140, 0.733), color=blue_dark)
    arrow((0.215, 0.733), (0.245, 0.733), color=blue_dark)
    arrow((0.355, 0.733), (0.385, 0.733), color=blue_dark)
    text(0.177, 0.660, "mask complete station windows", size=5.9, color=orange_dark)

    # Local reconstruction path.
    text(0.500, 0.850, "Local reconstruction", size=7.0, color=blue_dark, weight="bold", ha="left")
    box(0.500, 0.745, 0.090, 0.075, "Shared\nprediction head", face=COLORS["blue_xlight"], edge=blue, size=6.4)
    box(0.620, 0.745, 0.070, 0.075, r"$\hat X_{local}$", face="#FFFFFF", edge=blue, size=7.2)
    box(0.720, 0.745, 0.105, 0.075, r"$\mathcal{L}_{local}$" + "\nvalid positions", face=COLORS["neutral_xlight"], edge=neutral, size=6.2)
    arrow((0.460, 0.783), (0.500, 0.783), color=blue_dark)
    arrow((0.590, 0.783), (0.620, 0.783), color=blue_dark)
    arrow((0.690, 0.783), (0.720, 0.783), color=blue_dark)

    # Cross-station reconstruction path follows the same left-to-right direction one level below.
    text(0.500, 0.670, "Cross-station reconstruction", size=7.0, color=gold_dark, weight="bold", ha="left")
    box(0.500, 0.555, 0.085, 0.085, "+ Station ID\nembedding", face=COLORS["blue_xlight"], edge=blue, size=6.2)
    box(0.610, 0.555, 0.095, 0.085, "Reshape by\nindicator\n" + r"$B\cdot4\times N\times d$", face=COLORS["neutral_xlight"], edge=neutral, size=5.9)
    box(0.730, 0.540, 0.105, 0.115, "Cross-station block\n4-head MHA\nAddNorm - FFN - AddNorm", face=COLORS["gold_xlight"], edge=gold_dark, size=6.1)
    box(0.860, 0.555, 0.080, 0.085, r"Fuse" + "\n" + r"$Z+\alpha C$" + "\n" + r"$\alpha=\tanh(s)$", face=COLORS["orange_xlight"], edge=orange_dark, size=6.2)
    elbow((0.4225, 0.690), (0.500, 0.598), color=blue_dark)
    arrow((0.585, 0.598), (0.610, 0.598), color=COLORS["neutral_dark"])
    arrow((0.705, 0.598), (0.730, 0.598), color=gold_dark)
    arrow((0.835, 0.598), (0.860, 0.598), color=gold_dark)
    box(0.742, 0.665, 0.080, 0.035, "availability mask", face=COLORS["neutral_xlight"], edge=neutral, size=5.2)
    arrow((0.782, 0.665), (0.782, 0.655), color=COLORS["neutral_dark"], width=0.8, scale=7)

    box(0.860, 0.445, 0.080, 0.065, "Shared head\n" + r"$\hat X_{cross}$", face="#FFFFFF", edge=orange_dark, size=6.1)
    box(0.720, 0.445, 0.115, 0.065, r"$0.5\mathcal{L}_{cross-all}$" + "\n" + r"$+\mathcal{L}_{cross-mask}$", face=COLORS["orange_xlight"], edge=orange_dark, size=5.9)
    arrow((0.900, 0.555), (0.900, 0.510), color=orange_dark)
    arrow((0.860, 0.478), (0.835, 0.478), color=orange_dark)
    text(0.500, 0.445, "Cross-station modules are used only during pretraining; the compatible backbone is exported for transfer.", size=6.2, color=blue_dark, weight="bold", ha="left")

    # (b) Expand the shared station encoder used by both pretraining and fine-tuning.
    text(0.025, 0.355, "b", size=10.2, weight="bold", ha="left")
    text(0.052, 0.355, "Shared single-station Transformer backbone", size=10.2, weight="bold", ha="left")
    panel(0.025, 0.055, 0.950, 0.255, face=COLORS["blue_xlight"], edge=blue_dark)

    modules = [
        (0.045, 0.155, 0.075, 0.070, r"Input $x$" + "\n" + r"$B\times T\times4$", "#FFFFFF", blue_dark),
        (0.145, 0.155, 0.075, 0.070, "Instance\nnormalization", COLORS["neutral_xlight"], neutral),
        (0.245, 0.145, 0.095, 0.090, "Temporal adapter\nDWConv1D $k=5$\n+ residual", "#FFFFFF", blue),
        (0.365, 0.145, 0.095, 0.090, "Time-feature\nembedding\n4 tokens, $d=256$", "#FFFFFF", blue),
        (0.485, 0.135, 0.130, 0.110, "Transformer encoder x3\n8-head indicator attention\nMHA - AddNorm - FFN - AddNorm", COLORS["gold_xlight"], gold_dark),
        (0.640, 0.155, 0.085, 0.070, "Flatten +\nlinear head", COLORS["orange_xlight"], orange),
        (0.750, 0.155, 0.100, 0.070, "Inverse instance\nnormalization", COLORS["neutral_xlight"], neutral),
        (0.875, 0.155, 0.075, 0.070, r"Output $\hat y$" + "\n" + r"$B\times1\times4$", "#FFFFFF", orange_dark),
    ]
    for x, y, w, h, value, face, edge in modules:
        box(x, y, w, h, value, face=face, edge=edge, size=6.2)
    centers = [0.190, 0.190, 0.190, 0.190, 0.190, 0.190, 0.190, 0.190]
    for idx in range(len(modules) - 1):
        x, _, w, _, _, _, _ = modules[idx]
        nx, _, _, _, _, _, _ = modules[idx + 1]
        arrow((x + w, centers[idx]), (nx, centers[idx + 1]), color=blue_dark if idx < 4 else orange_dark)
    text(0.550, 0.105, "The same backbone architecture is initialized and fine-tuned at weekly, 4-day, and daily stages.", size=6.3, color=muted)

    return save_figure(fig, "fig3_ptl_model_architecture_en_v4", dpi=500)


def make_fig2_ptl_overall_workflow_v5() -> list[Path]:
    """Draw a compact PTL overview with explicit Transformers inside every stage."""
    fig, ax = plt.subplots(figsize=(15.8, 5.6))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.975, bottom=0.035)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = TOKENS["ink"]
    muted = TOKENS["muted"]
    blue = COLORS["blue_mid"]
    blue_dark = COLORS["blue_dark"]
    gold_dark = COLORS["gold_dark"]
    orange = COLORS["orange_mid"]
    orange_dark = COLORS["orange_dark"]
    neutral = COLORS["neutral_mid"]

    def text(x, y, value, *, size=7.2, color=ink, weight="normal", ha="center", va="center", z=8):
        ax.text(
            x,
            y,
            value,
            fontsize=size,
            color=color,
            fontweight=weight,
            ha=ha,
            va=va,
            linespacing=1.08,
            zorder=z,
        )

    def box(x, y, w, h, value, *, face="#FFFFFF", edge=neutral, size=6.7, linewidth=1.0, z=3):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=z,
        )
        ax.add_patch(patch)
        if value:
            text(x + w / 2, y + h / 2, value, size=size, weight="bold", z=z + 1)
        return patch

    def arrow(start, end, *, color=COLORS["neutral_dark"], width=1.15, dashed=False, scale=10, z=6):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=width,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle="arc3,rad=0",
            zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def panel(x, y, w, h, *, face, edge):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.007,rounding_size=0.014",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
            zorder=1,
        )
        ax.add_patch(patch)

    def waveform(x, y, w, h, *, color, count=4, seed=0):
        rng = np.random.default_rng(seed)
        xs = np.linspace(x + 0.004, x + w - 0.004, 64)
        for idx in range(count):
            center = y + h * (idx + 0.5) / count
            phase = rng.uniform(0, 2 * np.pi)
            values = (
                0.10 * h * np.sin(np.linspace(0, 3.2 * np.pi, len(xs)) + phase)
                + 0.022 * h * rng.normal(size=len(xs))
            )
            ax.plot(xs, center + values, color=color, linewidth=0.8, clip_on=True, zorder=5)

    def series_card(x, y, w, h, *, color, label, seed):
        box(x, y, w, h, "", face="#FFFFFF", edge=color, linewidth=1.0)
        waveform(x + 0.006, y + 0.040, w - 0.012, h - 0.072, color=color, count=4, seed=seed)
        text(x + w / 2, y + 0.018, label, size=5.8, color=color, weight="bold")

    def stage_card(x, title, resolution, window, loss):
        box(x, 0.350, 0.098, 0.260, "", face="#FFFFFF", edge=orange_dark, linewidth=1.1)
        text(x + 0.049, 0.575, title, size=7.2, color=orange_dark, weight="bold")
        text(x + 0.049, 0.530, resolution, size=6.4, weight="bold")
        box(x + 0.013, 0.420, 0.072, 0.075, "Transformer\nbackbone", face=COLORS["gold_xlight"], edge=gold_dark, size=6.0)
        text(x + 0.049, 0.385, window, size=5.9, color=muted)
        text(x + 0.049, 0.315, loss, size=5.9, color=orange_dark)

    # Two tightly coupled phases, matching the two domains in the manuscript.
    panel(0.018, 0.110, 0.440, 0.800, face=COLORS["blue_xlight"], edge=blue_dark)
    panel(0.478, 0.110, 0.504, 0.800, face=COLORS["orange_xlight"], edge=orange_dark)
    text(0.038, 0.868, "Source domain: cross-station masked pretraining", size=10.0, color=blue_dark, weight="bold", ha="left")
    text(0.498, 0.868, "Target domain: progressive adaptation", size=10.0, color=orange_dark, weight="bold", ha="left")

    # Source-domain flow: all arrows move left to right.
    series_card(0.038, 0.520, 0.058, 0.190, color=blue, label="18 stations", seed=2)
    box(0.115, 0.535, 0.065, 0.160, "Aligned\nweekly windows\n" + r"$168\times4$", face="#FFFFFF", edge=blue_dark, size=6.2)
    box(0.199, 0.535, 0.060, 0.160, "Station\nmask\n$r=0.15$", face=COLORS["gold_xlight"], edge=gold_dark, size=6.3)
    box(0.278, 0.515, 0.095, 0.200, "Shared Transformer\nbackbone\n+ cross-station\nattention", face="#FFFFFF", edge=blue_dark, size=6.4)
    box(0.392, 0.550, 0.048, 0.130, "Optimized\nbackbone\n" + r"$\theta_0$", face=COLORS["blue_light"], edge=blue_dark, size=6.0)
    arrow((0.096, 0.615), (0.115, 0.615), color=blue_dark)
    arrow((0.180, 0.615), (0.199, 0.615), color=blue_dark)
    arrow((0.259, 0.615), (0.278, 0.615), color=blue_dark)
    arrow((0.373, 0.615), (0.392, 0.615), color=blue_dark)
    text(0.239, 0.440, "Local reconstruction + cross-station reconstruction", size=6.4, color=blue_dark)
    text(
        0.239,
        0.360,
        r"$\mathcal{L}_{pre}=\mathcal{L}_{local}+0.5\mathcal{L}_{cross-all}+\mathcal{L}_{cross-mask}$",
        size=6.6,
        color=blue_dark,
    )
    text(0.239, 0.245, "Gap-aware windows; training-only scaling", size=6.1, color=muted)

    # Target-domain flow. Each stage contains its own Transformer backbone.
    series_card(0.498, 0.385, 0.060, 0.190, color=orange, label="Target station", seed=8)
    stage_card(0.584, "Stage 1", "Weekly", "56 d -> 8 steps -> next week", r"$\lambda_{NSE}=0.05$")
    stage_card(0.704, "Stage 2", "4-day", "32 d -> 8 steps -> next 4 d", r"$\lambda_{NSE}=0.10$")
    stage_card(0.824, "Stage 3", "Daily", "12 d -> 12 steps -> next day", r"$\lambda_{NSE}=0.15$")
    box(0.944, 0.405, 0.025, 0.150, r"$\hat y_{t+1}$", face="#FFFFFF", edge=orange_dark, size=7.4)
    arrow((0.558, 0.480), (0.584, 0.480), color=orange_dark)
    arrow((0.682, 0.480), (0.704, 0.480), color=orange_dark)
    arrow((0.802, 0.480), (0.824, 0.480), color=orange_dark)
    arrow((0.922, 0.480), (0.944, 0.480), color=orange_dark)
    text(0.754, 0.215, r"Compatible-weight handoff: $\theta_0\rightarrow\theta_1\rightarrow\theta_2\rightarrow\theta_3$", size=6.8, color=orange_dark)

    # Weight transfer first moves right, then down into Stage 1.
    ax.plot([0.440, 0.633], [0.745, 0.745], color=COLORS["neutral_dark"], linewidth=1.1, linestyle="--", zorder=5)
    ax.plot([0.633, 0.633], [0.745, 0.610], color=COLORS["neutral_dark"], linewidth=1.1, linestyle="--", zorder=5)
    arrow((0.633, 0.610), (0.633, 0.595), color=COLORS["neutral_dark"], width=1.1, dashed=True, scale=9)
    text(0.530, 0.770, r"transfer $\theta_0$", size=6.2, color=COLORS["neutral_dark"])

    text(0.500, 0.055, "Solid arrows: data and training flow    Dashed arrow: pretrained parameter transfer", size=6.2, color=muted)
    return save_figure(fig, "fig2_ptl_overall_workflow_en_v5", dpi=500)


def draw_workflow_box(ax, xy, wh, title, body, *, face, edge):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.012",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + h - 0.062,
        title,
        fontsize=9.8,
        fontweight="bold",
        va="top",
        ha="left",
        linespacing=1.12,
        color=TOKENS["ink"],
    )
    ax.text(
        x + 0.018,
        y + h - 0.145,
        body,
        fontsize=8.3,
        va="top",
        ha="left",
        linespacing=1.34,
        color=TOKENS["muted"],
    )


def make_method_workflow() -> list[Path]:
    fig = plt.figure(figsize=(13.2, 3.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (
            (0.030, 0.12),
            (0.135, 0.76),
            "Data sources",
            "Yangtze weekly\nrecords, 2007-2018\nPearl River daily\nrecords, 2021-2024\nCOD$_{\\mathrm{Mn}}$, DO,\nNH$_4$-N, pH",
            COLORS["blue_xlight"],
            COLORS["blue_dark"],
        ),
        (
            (0.195, 0.12),
            (0.135, 0.76),
            "Harmonization",
            "Indicator alignment\nmissingness screening\nstation grouping\nfixed data split",
            "#FFFFFF",
            COLORS["neutral_mid"],
        ),
        (
            (0.360, 0.12),
            (0.135, 0.76),
            "Source-domain\npretraining",
            "Masked reconstruction\ncross-station\nTransformer encoder\nshared representation",
            COLORS["gold_xlight"],
            COLORS["gold_dark"],
        ),
        (
            (0.525, 0.12),
            (0.135, 0.76),
            "Progressive\ntransfer",
            "Stage 1: weekly\nStage 2: 4-day\nStage 3: daily\nfine-tuning",
            COLORS["orange_xlight"],
            COLORS["orange_dark"],
        ),
        (
            (0.690, 0.12),
            (0.135, 0.76),
            "Evaluation",
            "Seven-model\nbenchmark\nOverall and Focus NSE\nRMSE, MAE",
            COLORS["blue_xlight"],
            COLORS["blue_dark"],
        ),
        (
            (0.855, 0.12),
            (0.135, 0.76),
            "Interpretation\nand reporting",
            "SHAP attribution\nstation/reach summaries\nrobustness analysis\npaper figures/tables",
            "#FFFFFF",
            COLORS["neutral_dark"],
        ),
    ]

    for xy, wh, title, body, face, edge in boxes:
        draw_workflow_box(ax, xy, wh, title, body, face=face, edge=edge)

    arrow_y = 0.51
    for (x0, _), (w0, _), *_ in boxes[:-1]:
        start = (x0 + w0 + 0.006, arrow_y)
        end = (x0 + w0 + 0.024, arrow_y)
        draw_arrow(ax, start, end, color=COLORS["neutral_dark"])

    return save_figure(fig, "fig_method_workflow_en", dpi=500)


def make_fig3_model_comparison(model_df: pd.DataFrame) -> list[Path]:
    summary = (
        model_df.groupby("model", as_index=False)
        .agg(
            overall_mean=("overall_nse", "mean"),
            overall_sem=("overall_nse", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
            focus_mean=("focus_mean_nse", "mean"),
            focus_sem=("focus_mean_nse", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        )
        .set_index("model")
        .loc[MODEL_ORDER]
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharey=True)
    fig.subplots_adjust(top=0.78, wspace=0.14)
    add_header(
        fig,
        "PTL outperforms direct-training baselines across overall and focus metrics",
        "Bars show station-level mean NSE across 17 target stations; error bars show standard error. Focus NSE averages COD$_{\\mathrm{Mn}}$, DO, and pH.",
    )

    metrics = [
        ("overall_mean", "overall_sem", "Overall NSE", "a"),
        ("focus_mean", "focus_sem", r"Focus NSE (COD$_{\mathrm{Mn}}$, DO, pH)", "b"),
    ]
    y = np.arange(len(MODEL_ORDER))
    x_min, x_max = -0.86, 0.90
    label_pad = 0.030
    fig3_faces = {
        "MLP": "#EAF1FE",
        "CNN": "#DEE9FE",
        "LSTM": "#D2E1FD",
        "Bi-LSTM": "#C5D8FB",
        "CNN-LSTM": "#B8CFF8",
        "Transformer": "#A3BEFA",
        "PTL": "#7F9EE4",
    }
    fig3_edges = {
        "MLP": "#7E91B8",
        "CNN": "#7389B5",
        "LSTM": "#687FAF",
        "Bi-LSTM": "#5E75A8",
        "CNN-LSTM": "#526AA0",
        "Transformer": "#3F5A93",
        "PTL": "#2E4780",
    }
    for ax, (mean_col, sem_col, title, label) in zip(axes, metrics):
        vals = summary[mean_col].to_numpy()
        sems = summary[sem_col].to_numpy()
        colors = [fig3_faces[m] for m in summary["model"]]
        edges = [fig3_edges[m] for m in summary["model"]]
        ax.barh(y, vals, xerr=sems, color=colors, edgecolor=edges, linewidth=1.0, height=0.68)
        ax.axvline(0, color=TOKENS["ink"], linewidth=0.9)
        for yi, val, sem in zip(y, vals, sems):
            if val >= 0:
                text_x = min(val + sem + label_pad, x_max - 0.025)
                ha = "left"
            else:
                text_x = max(val - sem - label_pad, x_min + 0.025)
                ha = "right"
            ax.text(
                text_x,
                yi,
                f"{val:.3f}",
                va="center",
                ha=ha,
                fontsize=8.5,
                color=TOKENS["ink"],
                bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "boxstyle": "round,pad=0.12", "alpha": 1.0},
                zorder=5,
                clip_on=False,
            )
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlabel("Mean NSE")
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(np.arange(-0.6, 0.81, 0.2))
        ax.set_yticks(y, labels=MODEL_ORDER)
        ax.invert_yaxis()
        clean_axes(ax, grid_axis=None)
        emphasize_axes(ax)
        panel_label(ax, label, parentheses=True, fontsize=12, x=-0.12, y=1.03)

    return save_figure(fig, "fig3_model_comparison_overall_focus_en")


def make_heatmap(ax, data, *, title, cmap, norm=None, annot=False, fmt=".2f", cbar=False, annot_fontsize=9):
    image = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    if title:
        ax.set_title(title, loc="left", pad=9)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = TOKENS["ink"] if abs(val) < 0.65 else "#FFFFFF"
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=annot_fontsize, color=color)
    if cbar:
        return image
    return image


def make_fig4_indicator_heatmap(indicator_df: pd.DataFrame) -> list[Path]:
    sub = indicator_df[indicator_df["indicator"].isin(INDICATOR_ORDER)].copy()
    mean = sub.groupby(["indicator", "model"])["nse"].mean().unstack("model").loc[INDICATOR_ORDER, MODEL_ORDER]

    cmap = LinearSegmentedColormap.from_list(
        "nse_diverging",
        [COLORS["orange_mid"], COLORS["orange_xlight"], "#FFFFFF", COLORS["blue_light"], COLORS["blue_mid"]],
    )
    norm = TwoSlopeNorm(vmin=-1.8, vcenter=0.0, vmax=0.9)

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    fig.subplots_adjust(top=0.78, right=0.86, bottom=0.24, left=0.12)
    add_header(
        fig,
        "PTL improves the difficult NH$_4$-N task while retaining strong COD$_{\\mathrm{Mn}}$, DO, and pH performance",
        "Cells show mean station-level NSE by model and indicator across 17 target stations.",
    )
    image = make_heatmap(ax, mean.to_numpy(), title="", cmap=cmap, norm=norm, annot=True, fmt=".3f", annot_fontsize=11)
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right")
    ax.set_yticklabels([CHEM_LABELS[i] for i in INDICATOR_ORDER])
    ax.tick_params(axis="both", labelsize=12, colors="#000000")
    ax.set_xlabel("Model", fontsize=12, color="#000000")
    ax.set_ylabel("Indicator", fontsize=12, color="#000000")
    cax = fig.add_axes([0.89, 0.24, 0.018, 0.48])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Mean NSE", color="#000000", fontsize=12)
    cbar.ax.tick_params(colors="#000000", labelsize=12)
    return save_figure(fig, "fig4_indicator_nse_heatmap_en")


def format_group_label(row) -> str:
    reach = REACH_LABELS.get(row["river_reach"], row["river_reach"])
    river_type = TYPE_LABELS.get(row["river_type"], row["river_type"])
    return f"{reach}\n{river_type.split('/')[0]}"


def make_fig5_group_heatmap(overall_group: pd.DataFrame, focus_group: pd.DataFrame) -> list[Path]:
    order_key = {"上游": 0, "中游": 1, "下游": 2}
    type_key = {"干流/主要水道": 0, "支流/区域河流": 1}
    for frame in (overall_group, focus_group):
        frame["_sort"] = frame["river_reach"].map(order_key) * 10 + frame["river_type"].map(type_key)
        frame.sort_values("_sort", inplace=True)
        frame["group_label"] = frame.apply(format_group_label, axis=1)

    overall = overall_group.set_index("group_label")[MODEL_ORDER]
    focus = focus_group.set_index("group_label")[MODEL_ORDER]
    cmap = LinearSegmentedColormap.from_list(
        "group_nse",
        [COLORS["orange_mid"], COLORS["orange_xlight"], "#FFFFFF", COLORS["blue_light"], COLORS["blue_mid"]],
    )
    norm = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=0.9)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7), sharey=False)
    fig.subplots_adjust(top=0.76, right=0.89, bottom=0.25, left=0.11, wspace=0.34)
    add_header(
        fig,
        "PTL remains strong across river reaches and river-system types",
        "Cells show group-level mean NSE; groups combine upper/middle/lower reaches with mainstem or tributary/regional stations.",
    )
    for ax, matrix, title, label in [
        (axes[0], overall, "Overall NSE", "a"),
        (axes[1], focus, r"Focus NSE (COD$_{\mathrm{Mn}}$, DO, pH)", "b"),
    ]:
        image = make_heatmap(
            ax,
            matrix.to_numpy(),
            title=title,
            cmap=cmap,
            norm=norm,
            annot=True,
            fmt=".3f",
            annot_fontsize=10,
        )
        ax.set_xticklabels(MODEL_ORDER, rotation=35, ha="right")
        ax.set_yticklabels(matrix.index)
        ax.tick_params(axis="both", colors="#000000", labelsize=10)
        panel_label(ax, label, parentheses=True, fontsize=12, x=-0.12)
    cax = fig.add_axes([0.92, 0.25, 0.014, 0.44])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Mean NSE", color="#000000", fontsize=10)
    cbar.ax.tick_params(colors="#000000", labelsize=10)
    return save_figure(fig, "fig5_group_performance_heatmap_en")


def make_fig6_training_availability(avail_df: pd.DataFrame) -> list[Path]:
    frame = avail_df[avail_df["model"].isin(MODEL_ORDER)].copy()
    frame["model"] = pd.Categorical(frame["model"], categories=MODEL_ORDER, ordered=True)
    frame.sort_values(["model", "training_availability_pct"], inplace=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.78, bottom=0.16, wspace=0.12)
    add_header(
        fig,
        "PTL remains stable as target-station training history is reduced",
        "Training availability masks the target-station training tail while keeping validation and test periods fixed; points show 17-station mean NSE.",
    )
    metrics = [
        ("mean_overall_nse", "overall_nse_sem", "Overall NSE", "a"),
        ("mean_focus_nse", "focus_nse_sem", r"Focus NSE (COD$_{\mathrm{Mn}}$, DO, pH)", "b"),
    ]
    for ax, (metric, sem, title, label) in zip(axes, metrics):
        for model in MODEL_ORDER:
            sub = frame[frame["model"] == model]
            linewidth = 1.5
            alpha = 1.0 if model in {"PTL", "Transformer"} else 0.68
            x = sub["training_availability_pct"].astype(float).to_numpy()
            y = sub[metric].astype(float).to_numpy()
            yerr = sub[sem].astype(float).to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=MODEL_EDGES[model],
                elinewidth=0.9,
                capsize=3.5,
                capthick=0.9,
                alpha=alpha,
                zorder=1.5,
            )
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4.6,
                linewidth=linewidth,
                linestyle="--",
                color=MODEL_EDGES[model],
                alpha=alpha,
                label=model,
                zorder=2.5,
            )
        ax.set_title(title, loc="left", pad=10, fontsize=12)
        ax.set_xlabel("Available training history (%)", fontsize=12)
        ax.set_ylabel("Mean NSE", fontsize=12)
        ax.set_xticks([40, 60, 80, 100])
        ax.set_ylim(-0.75, 0.90)
        ax.axhline(0, color=TOKENS["ink"], linewidth=0.9, linestyle="--")
        clean_axes(ax, grid_axis=None)
        emphasize_axes(ax, labelsize=12)
        add_axis_arrowheads(ax)
        panel_label(ax, label, parentheses=True, fontsize=12, x=-0.12, y=1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = [0, 4, 1, 5, 2, 6, 3]
    legend = fig.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        ncol=4,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.97),
        frameon=True,
        fancybox=False,
        facecolor="#FFFFFF",
        edgecolor="#000000",
        framealpha=1.0,
        fontsize=12,
        columnspacing=1.15,
        labelspacing=0.55,
        handlelength=1.45,
        handletextpad=0.45,
        borderaxespad=0.0,
        borderpad=0.55,
    )
    legend.get_frame().set_linewidth(0.8)
    return save_figure(fig, "fig6_training_availability_en")


def set_heatmap_ticks(ax, xlabels, ylabels, *, xrot=0):
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=xrot, ha="right" if xrot else "center")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)


def set_heatmap_text_size(ax, size: float) -> None:
    ax.tick_params(length=0, labelsize=size, colors="#000000")
    ax.xaxis.label.set_size(size)
    ax.yaxis.label.set_size(size)
    ax.xaxis.label.set_color("#000000")
    ax.yaxis.label.set_color("#000000")


def make_fig7_shap(shap_feature: pd.DataFrame, shap_lag: pd.DataFrame, shap_feature_lag: pd.DataFrame) -> list[Path]:
    feature_matrix = (
        shap_feature.pivot(index="target", columns="input_feature", values="station_equal_mean_abs_shap_scaled")
        .loc[TARGET_ORDER, INDICATOR_ORDER]
    )
    lag_matrix = (
        shap_lag.pivot(index="target", columns="lag_day", values="station_equal_mean_abs_shap_scaled")
        .loc[TARGET_ORDER, LAG_ORDER]
    )

    feature_lag = shap_feature_lag.copy()
    feature_lag = feature_lag[feature_lag["target"].isin(TARGET_ORDER) & feature_lag["input_feature"].isin(INDICATOR_ORDER)]

    all_values = feature_lag["station_equal_mean_abs_shap_scaled"].to_numpy()
    vmax_feature_lag = float(np.nanpercentile(all_values, 98))
    cmap = LinearSegmentedColormap.from_list("shap_blue", ["#FFFFFF", COLORS["blue_xlight"], COLORS["blue_base"], COLORS["blue_dark"]])

    fig = plt.figure(figsize=(14.2, 10.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 1.0], hspace=0.62, wspace=0.36)
    add_header(
        fig,
        "SHAP analysis shows strong self-history dependence and dominant recent lags",
        "Values are station-equal mean absolute SHAP contributions on the scaled input space across 17 target stations.",
        x=0.025,
        y=0.99,
    )
    fig.subplots_adjust(top=0.84, right=0.91, left=0.06, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0:2])
    im_a = ax_a.imshow(feature_matrix.to_numpy(), cmap=cmap, vmin=0, vmax=float(feature_matrix.to_numpy().max()), aspect="auto")
    ax_a.set_title("Input-feature importance by target", loc="left", pad=8, fontsize=10)
    set_heatmap_ticks(ax_a, [CHEM_LABELS[i] for i in INDICATOR_ORDER], [CHEM_LABELS[i] for i in TARGET_ORDER])
    ax_a.set_xlabel("Input feature", fontsize=10)
    ax_a.set_ylabel("Prediction target", fontsize=10)
    for i in range(feature_matrix.shape[0]):
        for j in range(feature_matrix.shape[1]):
            ax_a.text(j, i, f"{feature_matrix.to_numpy()[i, j]:.3f}", ha="center", va="center", fontsize=10)
    panel_label(ax_a, "a", parentheses=True, fontsize=10, x=-0.12)
    set_heatmap_text_size(ax_a, 10)
    ax_a.spines[:].set_visible(False)

    ax_b = fig.add_subplot(gs[0, 2:4])
    im_b = ax_b.imshow(lag_matrix.to_numpy(), cmap=cmap, vmin=0, vmax=float(lag_matrix.to_numpy().max()), aspect="auto")
    ax_b.set_title("Lag importance by target", loc="left", pad=8, fontsize=10)
    set_heatmap_ticks(ax_b, [f"t-{i}" for i in LAG_ORDER], [CHEM_LABELS[i] for i in TARGET_ORDER], xrot=45)
    ax_b.set_xlabel("Input lag", fontsize=10)
    ax_b.set_ylabel("")
    panel_label(ax_b, "b", parentheses=True, fontsize=10, x=-0.12)
    set_heatmap_text_size(ax_b, 10)
    ax_b.spines[:].set_visible(False)

    target_axes = []
    for idx, target in enumerate(TARGET_ORDER):
        row = 1 + idx // 2
        col = (idx % 2) * 2
        ax = fig.add_subplot(gs[row, col : col + 2])
        mat = (
            feature_lag[feature_lag["target"] == target]
            .pivot(index="input_feature", columns="lag_day", values="station_equal_mean_abs_shap_scaled")
            .loc[INDICATOR_ORDER, LAG_ORDER]
        )
        im = ax.imshow(mat.to_numpy(), cmap=cmap, vmin=0, vmax=vmax_feature_lag, aspect="auto")
        ax.set_title(f"Feature-lag importance for {CHEM_LABELS[target]} target", loc="left", pad=8, fontsize=10)
        set_heatmap_ticks(ax, [f"t-{i}" for i in LAG_ORDER], [CHEM_LABELS[i] for i in INDICATOR_ORDER], xrot=45)
        ax.set_xlabel("Input lag", fontsize=10)
        ax.set_ylabel("Input feature", fontsize=10)
        panel_label(ax, chr(ord("c") + idx), parentheses=True, fontsize=10, x=-0.12)
        set_heatmap_text_size(ax, 10)
        ax.spines[:].set_visible(False)
        target_axes.append(ax)

    cax = fig.add_axes([0.935, 0.18, 0.014, 0.55])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean absolute SHAP", color="#000000", fontsize=10)
    cbar.ax.tick_params(colors="#000000", labelsize=10)
    return save_figure(fig, "fig7_shap_summary_en", dpi=500)


def make_supplemental_model_heatmap(model_df: pd.DataFrame, metric: str, stem: str, title: str) -> list[Path]:
    abbr = pd.read_csv(CASE_FIG_DIR / "experiment_17_station_abbreviations_for_map.csv", encoding="utf-8-sig")
    station_to_code = dict(zip(abbr["station_cn"], abbr["station_code"]))
    order = abbr["station_cn"].tolist()
    matrix = model_df.pivot(index="station", columns="model", values=metric).reindex(order)[MODEL_ORDER]
    matrix.index = [station_to_code.get(s, s) for s in matrix.index]
    cmap = LinearSegmentedColormap.from_list(
        "nse_station",
        [COLORS["orange_mid"], COLORS["orange_xlight"], "#FFFFFF", COLORS["blue_light"], COLORS["blue_mid"]],
    )
    norm = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=0.95)
    fig, ax = plt.subplots(figsize=(9.8, 8.5))
    fig.subplots_adjust(top=0.83, right=0.88)
    add_header(fig, title, "Rows are 17 target stations shown by map abbreviation; columns are models.")
    image = ax.imshow(matrix.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(MODEL_ORDER)), labels=MODEL_ORDER, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), labels=matrix.index)
    ax.set_xlabel("Model")
    ax.set_ylabel("Target station")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    cax = fig.add_axes([0.91, 0.22, 0.018, 0.48])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("NSE", color=TOKENS["muted"])
    cbar.ax.tick_params(colors=TOKENS["muted"], labelsize=8)
    return save_figure(fig, stem)


def make_missing_heatmap() -> list[Path]:
    missing = pd.read_csv(CASE_DIR / "缺失情况统计" / "17站每日是否存在任一指标缺失_2023_2024.csv", encoding="utf-8-sig")
    abbr = pd.read_csv(CASE_FIG_DIR / "experiment_17_station_abbreviations_for_map.csv", encoding="utf-8-sig")
    station_order = abbr["station_cn"].tolist()
    codes = abbr["station_code"].tolist()
    dates = pd.to_datetime(missing["日期"])
    data = missing[station_order].T.to_numpy(dtype=float)

    cmap = ListedColormap(["#FFFFFF", COLORS["orange_base"]])
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    fig.subplots_adjust(top=0.78)
    add_header(
        fig,
        "Daily missingness pattern across 17 target stations",
        "Orange cells mark dates where at least one of COD$_{\\mathrm{Mn}}$, DO, NH$_4$-N, or pH is missing for a station.",
    )
    ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(codes)), labels=codes)
    ax.set_ylabel("Target station")
    ax.set_xlabel("Date")
    month_starts = pd.date_range(dates.min(), dates.max(), freq="MS")
    tick_positions = [int(np.searchsorted(dates.values, np.datetime64(d))) for d in month_starts]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in month_starts], rotation=45, ha="right")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.text(1.0, -0.18, "White = observed, orange = missing", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=TOKENS["muted"])
    return save_figure(fig, "figS_missing_heatmap_17stations_en")


def make_ablation_figure() -> list[Path]:
    path = SUMMARY_ROOT / "PTL_ablation" / "ptl_ablation_model_average_summary.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, encoding="utf-8-sig")
    labels = {
        "ptl_full": "Full PTL",
        "scratch_direct_daily": "Scratch daily",
        "pretrain_direct_daily": "Pretrain + daily",
        "no_progressive_handoff": "No stage handoff",
    }
    order = ["ptl_full", "scratch_direct_daily", "pretrain_direct_daily", "no_progressive_handoff"]
    frame = frame.set_index("ablation_key").loc[order].reset_index()
    x = np.arange(len(frame))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    fig.subplots_adjust(top=0.76)
    add_header(
        fig,
        "PTL ablation variants show similar average performance in the independent ablation batch",
        "Bars show 17-station mean NSE for same-backbone PTL variants; use as supplemental evidence rather than the main performance claim.",
    )
    ax.bar(x - width / 2, frame["mean_overall_nse"], width=width, color=COLORS["blue_base"], edgecolor=COLORS["blue_dark"], label="Overall NSE")
    ax.bar(x + width / 2, frame["mean_focus_nse"], width=width, color=COLORS["orange_base"], edgecolor=COLORS["orange_dark"], label="Focus NSE")
    ax.set_xticks(x, [labels[k] for k in frame["ablation_key"]], rotation=20, ha="right")
    ax.set_ylabel("Mean NSE")
    ax.set_ylim(0.68, 0.79)
    clean_axes(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper left")
    return save_figure(fig, "figS_ptl_ablation_en")


def write_manifest(paths_by_figure: dict[str, list[Path]]) -> Path:
    manifest = OUT_DIR / "figure_manifest.md"
    main_items = [(name, paths) for name, paths in paths_by_figure.items() if name.startswith("Figure ")]
    supplemental_items = [(name, paths) for name, paths in paths_by_figure.items() if name.startswith("Supplement")]
    source_assets = [
        ("Figure 1 editable source", OUT_DIR / "fig1_study_area_17stations_en.drawio"),
        ("PTL framework editable source", OUT_DIR / "PTL_framework.drawio"),
        ("Pearl basin SVG asset", OUT_DIR / "Pearl.svg"),
        ("GraphPad Prism - Figure 3", OUT_DIR / "prism_fig3_model_comparison"),
        ("GraphPad Prism - Figure 6", OUT_DIR / "prism_fig6_training_availability"),
    ]
    lines = [
        "# English Paper Figure Manifest",
        "",
        "Generated by `src/Base/analysis/make_paper_english_figures_17stations.py`.",
        "",
        "This manifest follows the currently retained journal figure set in `paper_figures_en_journal`.",
        "",
        "Chemical labels use math notation: `COD$_{\\mathrm{Mn}}$` and `NH$_4$-N`.",
        "",
        "## Main figures",
        "",
        "| Figure | Files |",
        "| --- | --- |",
    ]
    for name, paths in main_items:
        file_list = "<br>".join(str(path) for path in paths)
        lines.append(f"| {name} | {file_list} |")
    if supplemental_items:
        lines.extend(["", "## Supplemental figures", "", "| Figure | Files |", "| --- | --- |"])
        for name, paths in supplemental_items:
            file_list = "<br>".join(str(path) for path in paths)
            lines.append(f"| {name} | {file_list} |")
    existing_assets = [(name, path) for name, path in source_assets if path.exists()]
    if existing_assets:
        lines.extend(["", "## Source and support assets", "", "| Asset | Path |", "| --- | --- |"])
        for name, path in existing_assets:
            lines.append(f"| {name} | {path} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- SVG files are the current editable figure targets.",
            "- PDF and PNG exports should be regenerated after final SVG-only touch-ups if those formats are submitted.",
            "- Hidden macOS metadata and editor backup files are intentionally omitted from this manifest.",
        ]
    )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_df = pd.read_csv(CASE_DIR / "模型对比长表.csv", encoding="utf-8-sig")
    indicator_df = pd.read_csv(CASE_DIR / "各指标_站点_模型_NSE长表.csv", encoding="utf-8-sig")
    overall_group = pd.read_csv(CASE_DIR / "分组均值_Overall_NSE.csv", encoding="utf-8-sig")
    focus_group = pd.read_csv(CASE_DIR / "分组均值_Focus_NSE.csv", encoding="utf-8-sig")
    avail_df = pd.read_csv(TRAIN_AVAIL_DIR / "training_availability_17stations_all_models_model_average_summary.csv", encoding="utf-8-sig")
    shap_feature = pd.read_csv(SHAP_DIR / "shap_global_feature_importance.csv", encoding="utf-8-sig")
    shap_lag = pd.read_csv(SHAP_DIR / "shap_global_lag_importance.csv", encoding="utf-8-sig")
    shap_feature_lag = pd.read_csv(SHAP_DIR / "shap_global_feature_lag_heatmap.csv", encoding="utf-8-sig")

    outputs: dict[str, list[Path]] = {}
    outputs["Figure 1 - Study area"] = export_existing_study_area()
    outputs["Figure 2 - PTL framework"] = make_fig2_reference_style_architecture()
    outputs["Figure 3 - Overall and focus NSE"] = make_fig3_model_comparison(model_df)
    outputs["Figure 4 - Indicator NSE"] = make_fig4_indicator_heatmap(indicator_df)
    outputs["Figure 5 - Group performance"] = make_fig5_group_heatmap(overall_group, focus_group)
    outputs["Figure 6 - Training availability"] = make_fig6_training_availability(avail_df)
    outputs["Figure 7 - SHAP summary"] = make_fig7_shap(shap_feature, shap_lag, shap_feature_lag)
    fig8_paths = [
        OUT_DIR / f"fig8_six_model_rmse_mae_ptl_comparison_en.{extension}"
        for extension in ("pdf", "png", "svg")
        if (OUT_DIR / f"fig8_six_model_rmse_mae_ptl_comparison_en.{extension}").exists()
    ]
    if fig8_paths:
        outputs["Figure 8 - RMSE and MAE with and without PTL"] = fig8_paths
    outputs["Supplement - Overall NSE heatmap"] = make_supplemental_model_heatmap(
        model_df,
        "overall_nse",
        "figS_overall_nse_heatmap_en",
        "Overall NSE heatmap across 17 target stations",
    )
    outputs["Supplement - Focus NSE heatmap"] = make_supplemental_model_heatmap(
        model_df,
        "focus_mean_nse",
        "figS_focus_nse_heatmap_en",
        r"Focus NSE heatmap across 17 target stations",
    )
    outputs["Supplement - Missingness heatmap"] = make_missing_heatmap()
    ablation_paths = make_ablation_figure()
    if ablation_paths:
        outputs["Supplement - PTL ablation"] = ablation_paths

    manifest = write_manifest(outputs)
    print(f"Wrote figures to: {OUT_DIR}")
    print(f"Wrote manifest: {manifest}")
    for name, paths in outputs.items():
        print(f"{name}: {len(paths)} files")


if __name__ == "__main__":
    main()
