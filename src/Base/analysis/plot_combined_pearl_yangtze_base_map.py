from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

import plot_pearl_17_station_map_trimmed as pearl
import plot_yangtze_18_station_map as yangtze


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "combined_pearl_yangtze"

RIVER_COLOR = "#176fa8"
STATION_EDGE_COLOR = "#2b2f33"
YANGTZE_STATION_COLOR = "#e6ab02"
PEARL_STATION_COLOR = "#d1495b"
RIVER_STYLES = {
    "5": dict(color=RIVER_COLOR, linewidth=0.45, alpha=0.70, zorder=2),
    "4": dict(color=RIVER_COLOR, linewidth=0.65, alpha=0.78, zorder=3),
    "23": dict(color=RIVER_COLOR, linewidth=1.05, alpha=0.88, zorder=4),
    "1": dict(color=RIVER_COLOR, linewidth=2.00, alpha=0.98, zorder=5),
}
MARKER_BY_TYPE = {
    "Mainstream/major waterway": "o",
    "Tributary/regional river": "^",
}
STATION_MARKER_SIZE = 112 * 1.15


def configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax: plt.Axes, bbox: tuple[float, float, float, float]) -> None:
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_xlabel("Longitude", fontsize=14, labelpad=5)
    ax.set_ylabel("Latitude", fontsize=14, labelpad=5)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis="both", labelsize=11, width=0.75, length=3.5)
    ax.grid(True, color="#d5e2eb", linewidth=0.55, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#333333")


def draw_background(ax: plt.Axes, provinces: gpd.GeoDataFrame) -> None:
    ax.set_facecolor("#f7fbff")
    provinces.plot(
        ax=ax,
        facecolor="#f4f6f7",
        edgecolor="#bfc5c9",
        linewidth=0.55,
        zorder=0,
    )
    provinces.boundary.plot(ax=ax, color="#b9bec2", linewidth=0.55, zorder=1)


def draw_rivers(ax: plt.Axes, rivers: gpd.GeoDataFrame) -> None:
    for level_key in ["5", "4", "23", "1"]:
        part = rivers[rivers["level_key"].eq(level_key)]
        if not part.empty:
            part.plot(ax=ax, **RIVER_STYLES[level_key])


def draw_station(
    ax: plt.Axes,
    x: float,
    y: float,
    marker: str,
    color: str,
) -> None:
    ax.scatter(
        x,
        y,
        s=STATION_MARKER_SIZE * 1.75,
        marker=marker,
        color="white",
        edgecolor="white",
        linewidth=0,
        alpha=0.96,
        zorder=7.8,
    )
    ax.scatter(
        x,
        y,
        s=STATION_MARKER_SIZE,
        marker=marker,
        color=color,
        edgecolor=STATION_EDGE_COLOR,
        linewidth=0.9,
        zorder=8,
    )


def add_scale_bar(
    ax: plt.Axes,
    *,
    kilometers: int,
    x0: float,
    y0: float,
    latitude_factor: float,
) -> None:
    length_deg = kilometers / (111.32 * latitude_factor)
    x1 = x0 + length_deg
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    cap = y_span * 0.0095
    label_offset = y_span * 0.023
    ax.plot([x0, x1], [y0, y0], color="#222222", lw=2.2, zorder=10)
    ax.plot([x0, x0], [y0 - cap, y0 + cap], color="#222222", lw=2.2, zorder=10)
    ax.plot([x1, x1], [y0 - cap, y0 + cap], color="#222222", lw=2.2, zorder=10)
    ax.text(
        (x0 + x1) / 2,
        y0 + label_offset,
        f"{kilometers} km",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#222222",
        family="Arial",
        zorder=10,
    )


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "",
        xy=(0.975, 0.905),
        xytext=(0.975, 0.775),
        xycoords="axes fraction",
        arrowprops=dict(
            facecolor="#111111",
            edgecolor="#111111",
            width=4,
            headwidth=18,
        ),
        zorder=10,
    )
    ax.text(
        0.975,
        0.925,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        family="Arial",
        color="#111111",
        zorder=10,
    )


def draw_yangtze(ax: plt.Axes) -> None:
    rivers = yangtze.filter_rivers(yangtze.load_rivers())
    lakes = yangtze.load_lakes()
    stations = yangtze.load_stations()
    provinces = gpd.read_file(yangtze.PROVINCE_SHP).to_crs("EPSG:4326")
    provinces = provinces.cx[
        yangtze.BBOX[0] : yangtze.BBOX[2],
        yangtze.BBOX[1] : yangtze.BBOX[3],
    ].copy()

    draw_background(ax, provinces)
    lakes.plot(
        ax=ax,
        facecolor=RIVER_COLOR,
        edgecolor=RIVER_COLOR,
        linewidth=0.65,
        alpha=0.13,
        zorder=1.5,
    )
    draw_rivers(ax, rivers)
    for _, row in stations.iterrows():
        marker = MARKER_BY_TYPE[row["river_type"]]
        draw_station(
            ax,
            row["LongitudeMeasure"],
            row["LatitudeMeasure"],
            marker,
            YANGTZE_STATION_COLOR,
        )

    style_axes(ax, yangtze.BBOX)
    add_scale_bar(
        ax,
        kilometers=300,
        x0=116.85,
        y0=27.86,
        latitude_factor=0.875,
    )
    add_north_arrow(ax)


def draw_pearl(ax: plt.Axes) -> None:
    rivers = pearl.filter_rivers(pearl.load_rivers())
    stations = pearl.load_stations()
    provinces = gpd.read_file(pearl.PROVINCE_SHP).to_crs("EPSG:4326")
    provinces = provinces.cx[
        pearl.BBOX[0] : pearl.BBOX[2],
        pearl.BBOX[1] : pearl.BBOX[3],
    ].copy()

    draw_background(ax, provinces)
    draw_rivers(ax, rivers)
    pearl_type_map = {
        "干流/主要水道": "o",
        "支流/区域河流": "^",
    }
    for _, row in stations.iterrows():
        draw_station(
            ax,
            row["经度"],
            row["纬度"],
            pearl_type_map[row["river_type"]],
            PEARL_STATION_COLOR,
        )

    style_axes(ax, pearl.BBOX)
    add_scale_bar(
        ax,
        kilometers=200,
        x0=114.55,
        y0=21.98,
        latitude_factor=0.928,
    )
    add_north_arrow(ax)


def add_shared_legend(fig: plt.Figure) -> None:
    color_handles = [
        Line2D([0], [0], color=RIVER_COLOR, lw=3.0, label="Rivers"),
        Patch(
            facecolor=RIVER_COLOR,
            edgecolor=RIVER_COLOR,
            alpha=0.13,
            label="Major lakes",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=YANGTZE_STATION_COLOR,
            markeredgecolor=STATION_EDGE_COLOR,
            markersize=10.5,
            label="Yangtze stations",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PEARL_STATION_COLOR,
            markeredgecolor=STATION_EDGE_COLOR,
            markersize=10.5,
            label="Pearl stations",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#ffffff",
            markeredgecolor=STATION_EDGE_COLOR,
            markersize=10.5,
            label="Mainstream/major waterway",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#ffffff",
            markeredgecolor=STATION_EDGE_COLOR,
            markersize=10.5,
            label="Tributary/regional river",
        ),
    ]
    fig.legend(
        handles=color_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.012),
        ncol=3,
        frameon=False,
        fontsize=11.5,
        handlelength=2.2,
        handletextpad=0.7,
        columnspacing=1.7,
        labelspacing=0.7,
    )


def export_drawio_from_svg(svg_path: Path, drawio_path: Path) -> None:
    encoded_svg = quote(svg_path.read_text(encoding="utf-8"), safe="")
    style = (
        "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
        "imageAspect=1;aspect=fixed;image=data:image/svg+xml,"
        f"{encoded_svg};"
    )
    xml = f'''<mxfile host="app.diagrams.net" modified="2026-07-06T00:00:00.000Z" agent="Codex" version="24.7.17" type="device">
  <diagram id="combined-map" name="Page-1">
    <mxGraphModel dx="1600" dy="1320" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="1320" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="map-svg" value="" style="{style}" vertex="1" parent="1">
          <mxGeometry x="0" y="0" width="1600" height="1320" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
    drawio_path.write_text(xml, encoding="utf-8")


def main() -> None:
    configure_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 14.5), dpi=300, facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.965,
        top=0.975,
        bottom=0.090,
        width_ratios=(3.05, 1.0),
        height_ratios=(1.10, 1.00),
        hspace=0.10,
        wspace=0.055,
    )
    yangtze_ax = fig.add_subplot(grid[0, 0])
    inset_ax = fig.add_subplot(grid[0, 1])
    pearl_ax = fig.add_subplot(grid[1, :])

    draw_yangtze(yangtze_ax)
    draw_pearl(pearl_ax)

    inset_ax.set_axis_off()

    add_shared_legend(fig)

    stem = "pearl_yangtze_combined_base_map"
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    drawio_path = OUTPUT_DIR / f"{stem}.drawio"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)
    export_drawio_from_svg(svg_path, drawio_path)

    for path in [png_path, pdf_path, svg_path, drawio_path]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
