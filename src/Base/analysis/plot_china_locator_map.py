from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import geopandas as gpd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
PROVINCE_SHP = (
    REPO_ROOT
    / "data"
    / "ChinaAdminDivisonSHP-24.02.06"
    / "2. Province"
    / "province.shp"
)
COUNTRY_SHP = (
    REPO_ROOT
    / "data"
    / "ChinaAdminDivisonSHP-24.02.06"
    / "1. Country"
    / "country.shp"
)
NATIONAL_BOUNDARY_SHP = (
    REPO_ROOT
    / "data"
    / "我国河流、湖泊数据集"
    / "我国国界线"
    / "国界线.shp"
)
OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "combined_pearl_yangtze"

SEA_COLOR = "#f7fbff"
LAND_COLOR = "#f4f6f7"
COAST_COLOR = "#aeb5ba"
PROVINCE_LINE_COLOR = "#b9bec2"
FRAME_COLOR = "#333333"


def load_map_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    provinces = gpd.read_file(PROVINCE_SHP).to_crs("EPSG:4326")
    country = gpd.read_file(COUNTRY_SHP).to_crs("EPSG:4326")
    national_boundary = gpd.read_file(NATIONAL_BOUNDARY_SHP).to_crs("EPSG:4326")
    return provinces, country, national_boundary


def style_map_axes(ax: plt.Axes, *, frame: bool = False) -> None:
    ax.set_facecolor(SEA_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(frame)
        spine.set_linewidth(0.75)
        spine.set_color(FRAME_COLOR)


def draw_china_locator(ax: plt.Axes, *, outer_frame: bool = True) -> None:
    provinces, country, national_boundary = load_map_data()
    style_map_axes(ax, frame=outer_frame)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    main_ax = ax.inset_axes([0.035, 0.535, 0.93, 0.42])
    style_map_axes(main_ax)
    country.plot(
        ax=main_ax,
        facecolor=LAND_COLOR,
        edgecolor=COAST_COLOR,
        linewidth=0.65,
        zorder=1,
    )
    provinces.boundary.plot(
        ax=main_ax,
        color=PROVINCE_LINE_COLOR,
        linewidth=0.38,
        zorder=2,
    )
    main_ax.set_xlim(73.0, 135.7)
    main_ax.set_ylim(17.0, 54.3)

    south_sea_ax = ax.inset_axes([0.20, 0.055, 0.60, 0.42])
    style_map_axes(south_sea_ax, frame=True)
    south_country = country.cx[105.0:123.5, 3.0:24.0]
    south_boundary = national_boundary.cx[105.0:123.5, 3.0:24.0]
    south_country.plot(
        ax=south_sea_ax,
        facecolor=LAND_COLOR,
        edgecolor=COAST_COLOR,
        linewidth=0.50,
        zorder=1,
    )
    south_boundary.plot(
        ax=south_sea_ax,
        color=PROVINCE_LINE_COLOR,
        linewidth=0.28,
        alpha=0.95,
        zorder=2,
    )
    south_sea_ax.set_xlim(105.0, 123.5)
    south_sea_ax.set_ylim(3.0, 24.0)


def export_drawio_from_svg(svg_path: Path, drawio_path: Path) -> None:
    encoded_svg = quote(svg_path.read_text(encoding="utf-8"), safe="")
    style = (
        "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
        "imageAspect=1;aspect=fixed;image=data:image/svg+xml,"
        f"{encoded_svg};"
    )
    xml = f'''<mxfile host="app.diagrams.net" modified="2026-07-06T00:00:00.000Z" agent="Codex" version="24.7.17" type="device">
  <diagram id="china-map" name="Page-1">
    <mxGraphModel dx="800" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="800" pageHeight="1000" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="china-svg" value="" style="{style}" vertex="1" parent="1">
          <mxGeometry x="0" y="0" width="800" height="1000" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
    drawio_path.write_text(xml, encoding="utf-8")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 10), dpi=300, facecolor="white")
    ax = fig.add_axes([0.04, 0.035, 0.92, 0.93])
    draw_china_locator(ax)

    stem = "china_locator_with_taiwan_south_china_sea"
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
