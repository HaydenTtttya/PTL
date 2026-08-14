from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.ops import unary_union


REPO_ROOT = Path(__file__).resolve().parents[3]
RIVER_DIR = REPO_ROOT / "data" / "我国河流、湖泊数据集" / "我国河流数据"
LAKE_SHP = (
    REPO_ROOT
    / "data"
    / "我国河流、湖泊数据集"
    / "我国湖泊数据"
    / "一级大河、湖泊.shp"
)
PROVINCE_SHP = (
    REPO_ROOT
    / "data"
    / "ChinaAdminDivisonSHP-24.02.06"
    / "2. Province"
    / "province.shp"
)
SITE_COORD_PATH = REPO_ROOT / "data" / "sites149.xlsx"
METADATA_XLS_PATH = REPO_ROOT / "data" / "1980-2022 年水质数据" / "metadata_and_statistics.xls"
OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "yangtze_river_basin_stations"

BBOX = (110.45, 27.65, 120.45, 32.80)
CONFLUENCE_BUFFER_M = 300
STATION_IDS = [2, 5, 8, 17, 20, 32, 33, 40, 48, 52, 60, 69, 79, 85, 107, 123, 131, 141]

MAINSTREAM_IDS = {8, 17, 20, 33, 48, 52, 69, 85, 107, 123, 131, 141}

STATION_NAMES = {
    2: "Qixing",
    5: "Wanjiazui",
    8: "Sanjiangying",
    17: "Nanzui",
    20: "Nanjinguan",
    32: "Potou",
    33: "Chenglingji",
    40: "Zongguan",
    48: "Yueyanglou",
    52: "Kangshan",
    60: "Xingang",
    69: "Songshan",
    79: "Shahekou",
    85: "Hexishuicun",
    107: "Wanhekou",
    123: "Hamashi",
    131: "Duchang",
    141: "Lujiao",
}

STATION_ABBREVIATIONS = {
    2: "QX",
    5: "WJZ",
    8: "SJY",
    17: "NZ",
    20: "NJG",
    32: "PT",
    33: "CLJ",
    40: "ZG",
    48: "YYL",
    52: "KS",
    60: "XG",
    69: "SS",
    79: "SHK",
    85: "HXSC",
    107: "WHK",
    123: "HMS",
    131: "DC",
    141: "LJ",
}

STATION_WATERBODIES = {
    2: "Liangzi Lake",
    5: "Zi River",
    8: "Grand Canal (China)",
    17: "Dongting Lake",
    20: "Yangtze River",
    32: "Xiang River",
    33: "Yangtze River",
    40: "Han River",
    48: "Dongting Lake",
    52: "Boyang Lake",
    60: "Xiang River",
    69: "Yangtze River",
    79: "Li River",
    85: "Yangtze River",
    107: "Yangtze River",
    123: "Boyang Lake",
    131: "Boyang Lake",
    141: "Dongting Lake",
}

STATION_RIVER_NAMES = {
    "长江",
    "湘江",
    "沅江",
    "汉江(汉水)",
    "赣江",
    "资水",
    "澧水",
    "松滋河",
    "虎渡河",
    "黄柏河",
    "沩水",
    "皖河(长河)",
    "金水",
}

PROVINCE_LABEL_POSITIONS = {
    "Hunan": (111.55, 28.10),
    "Hubei": (112.35, 31.05),
    "Jiangxi": (115.10, 28.05),
    "Anhui": (117.15, 31.20),
    "Jiangsu": (119.10, 32.25),
    "Shanghai": (120.18, 31.10),
    "Henan": (112.55, 32.48),
    "Zhejiang": (119.30, 29.15),
}

PROVINCE_NAMES = {
    "湖南省": "Hunan",
    "湖北省": "Hubei",
    "江西省": "Jiangxi",
    "安徽省": "Anhui",
    "江苏省": "Jiangsu",
    "上海市": "Shanghai",
    "河南省": "Henan",
    "浙江省": "Zhejiang",
}

LABEL_OFFSETS = {
    "QX": (-0.28, -0.24),
    "WJZ": (-0.26, -0.22),
    "SJY": (-0.30, 0.18),
    "NZ": (-0.24, 0.22),
    "NJG": (-0.28, 0.22),
    "PT": (-0.25, -0.22),
    "CLJ": (0.25, 0.22),
    "ZG": (-0.25, 0.22),
    "YYL": (-0.28, -0.23),
    "KS": (0.25, -0.23),
    "XG": (0.24, 0.20),
    "SS": (-0.25, 0.22),
    "SHK": (-0.27, 0.22),
    "HXSC": (-0.32, 0.22),
    "WHK": (0.30, 0.20),
    "HMS": (0.32, 0.22),
    "DC": (0.28, -0.24),
    "LJ": (0.25, -0.24),
}

RIVER_LABELS = [
    ("Yangtze River", 111.45, 30.35),
    ("Yangtze River", 115.25, 30.38),
    ("Yangtze River", 118.35, 31.55),
    ("Han River", 113.40, 30.55),
    ("Xiang River", 113.30, 28.16),
    ("Zi River", 111.78, 28.38),
    ("Gan River", 116.12, 28.33),
    ("Dongting Lake", 112.62, 29.78),
    ("Poyang Lake", 115.62, 29.22),
]


def load_rivers() -> gpd.GeoDataFrame:
    parts: list[gpd.GeoDataFrame] = []
    specs = [
        ("5", "我国五级河流.shp"),
        ("4", "我国四级河流.shp"),
        ("23", "我国二三级河流.shp"),
        ("1", "我国一级河流.shp"),
    ]
    for level_key, file_name in specs:
        gdf = gpd.read_file(RIVER_DIR / file_name).to_crs("EPSG:4326")
        if level_key == "1":
            gdf = gdf[gdf["LEVEL_RIVE"].eq(1)].copy()
        elif level_key == "23":
            gdf = gdf[gdf["LEVEL_RIVE"].isin([2, 3])].copy()
        gdf = gdf.cx[BBOX[0] : BBOX[2], BBOX[1] : BBOX[3]].copy()
        gdf["level_key"] = level_key
        parts.append(gdf[["NAME", "level_key", "geometry"]])
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")


def filter_rivers(rivers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    river_names = rivers["NAME"].fillna("").astype(str)
    is_first_order = rivers["level_key"].eq("1")
    is_station_river = river_names.isin(STATION_RIVER_NAMES)
    metric = rivers.to_crs("EPSG:3857")
    station_union = unary_union(metric.loc[is_station_river, "geometry"].to_list())
    intersects_station_rivers = metric.geometry.intersects(
        station_union.buffer(CONFLUENCE_BUFFER_M)
    )
    return rivers.loc[is_first_order | is_station_river | intersects_station_rivers].copy()


def load_lakes() -> gpd.GeoDataFrame:
    lakes = gpd.read_file(LAKE_SHP).to_crs("EPSG:4326")
    lakes = lakes[lakes["NAME"].isin(["洞庭湖", "鄱阳湖"])].copy()
    return lakes.cx[BBOX[0] : BBOX[2], BBOX[1] : BBOX[3]].copy()


def load_stations() -> gpd.GeoDataFrame:
    sites = pd.read_excel(SITE_COORD_PATH)
    sites = sites[sites["MonitoringLocationIdentifier"].isin(STATION_IDS)].copy()
    sites["station_id"] = sites["MonitoringLocationIdentifier"].astype(int)
    sites["station_name"] = sites["station_id"].map(STATION_NAMES)
    sites["station_code"] = sites["station_id"].map(STATION_ABBREVIATIONS)
    sites["waterbody"] = sites["station_id"].map(STATION_WATERBODIES)
    sites["river_type"] = sites["station_id"].apply(
        lambda station_id: "Mainstream/major waterway"
        if station_id in MAINSTREAM_IDS
        else "Tributary/regional river"
    )
    return gpd.GeoDataFrame(
        sites.sort_values("station_id"),
        geometry=gpd.points_from_xy(sites["LongitudeMeasure"], sites["LatitudeMeasure"]),
        crs="EPSG:4326",
    )


def save_station_metadata_table(stations: gpd.GeoDataFrame) -> None:
    table = stations[
        [
            "station_id",
            "station_name",
            "station_code",
            "waterbody",
            "river_type",
            "LongitudeMeasure",
            "LatitudeMeasure",
        ]
    ].copy()
    table["metadata_source"] = str(METADATA_XLS_PATH)
    table.rename(
        columns={"LongitudeMeasure": "longitude", "LatitudeMeasure": "latitude"}
    ).to_csv(
        OUTPUT_DIR / "yangtze_18_station_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )


def add_scale_bar(ax: plt.Axes, x0: float = 116.85, y0: float = 27.86) -> None:
    length_deg = 300 / (111.32 * 0.875)
    x1 = x0 + length_deg
    ax.plot([x0, x1], [y0, y0], color="#222222", lw=2.2, zorder=10)
    ax.plot([x0, x0], [y0 - 0.045, y0 + 0.045], color="#222222", lw=2.2, zorder=10)
    ax.plot([x1, x1], [y0 - 0.045, y0 + 0.045], color="#222222", lw=2.2, zorder=10)
    ax.text(
        (x0 + x1) / 2,
        y0 + 0.10,
        "300 km",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#222222",
    )


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "",
        xy=(119.95, 32.42),
        xytext=(119.95, 31.80),
        arrowprops=dict(facecolor="#111111", edgecolor="#111111", width=4, headwidth=18),
        zorder=10,
    )
    ax.text(119.95, 32.47, "N", ha="center", va="bottom", fontsize=13, fontweight="bold")


def export_drawio_from_svg(svg_path: Path, drawio_path: Path) -> None:
    encoded_svg = quote(svg_path.read_text(encoding="utf-8"), safe="")
    style = (
        "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
        "imageAspect=1;aspect=fixed;image=data:image/svg+xml,"
        f"{encoded_svg};"
    )
    xml = f'''<mxfile host="app.diagrams.net" modified="2026-07-06T00:00:00.000Z" agent="Codex" version="24.7.17" type="device">
  <diagram id="yangtze-river-map" name="Yangtze River Map">
    <mxGraphModel dx="1600" dy="946" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="946" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="map-svg" value="" style="{style}" vertex="1" parent="1">
          <mxGeometry x="0" y="0" width="1600" height="946" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
    drawio_path.write_text(xml, encoding="utf-8")


def plot_map(show_map_text: bool = True, output_suffix: str = "") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rivers = filter_rivers(load_rivers())
    lakes = load_lakes()
    stations = load_stations()
    save_station_metadata_table(stations)
    provinces = gpd.read_file(PROVINCE_SHP).to_crs("EPSG:4326")
    provinces = provinces.cx[BBOX[0] : BBOX[2], BBOX[1] : BBOX[3]].copy()

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
    fig = plt.figure(figsize=(16, 9.458333), dpi=288)
    ax = fig.add_axes([0.055, 0.24 if not show_map_text else 0.21, 0.90, 0.64 if not show_map_text else 0.67])
    ax.set_facecolor("#f7fbff")
    provinces.plot(ax=ax, facecolor="#f4f6f7", edgecolor="#bfc5c9", linewidth=0.55, zorder=0)
    provinces.boundary.plot(ax=ax, color="#b9bec2", linewidth=0.55, zorder=1)

    unified_river_color = "#176fa8"
    lakes.plot(
        ax=ax,
        facecolor=unified_river_color,
        edgecolor=unified_river_color,
        linewidth=0.65,
        alpha=0.13,
        zorder=1.5,
    )

    if show_map_text:
        for province_cn, label in PROVINCE_NAMES.items():
            if label not in PROVINCE_LABEL_POSITIONS:
                continue
            x, y = PROVINCE_LABEL_POSITIONS[label]
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=16,
                color="#8f9599",
                alpha=0.72,
                zorder=1.8,
            )

    river_styles = {
        "5": dict(color=unified_river_color, linewidth=0.45, alpha=0.70, zorder=2),
        "4": dict(color=unified_river_color, linewidth=0.65, alpha=0.78, zorder=3),
        "23": dict(color=unified_river_color, linewidth=1.05, alpha=0.88, zorder=4),
        "1": dict(color=unified_river_color, linewidth=2.00, alpha=0.98, zorder=5),
    }
    for level_key in ["5", "4", "23", "1"]:
        part = rivers[rivers["level_key"].eq(level_key)]
        if not part.empty:
            part.plot(ax=ax, **river_styles[level_key])

    if show_map_text:
        for text, x, y in RIVER_LABELS:
            ax.text(
                x,
                y,
                text,
                fontsize=14,
                color="#5b3f27",
                alpha=0.95,
                fontstyle="italic",
                ha="center",
                va="center",
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="#f7fbff", alpha=0.95)],
            )

    station_color = "#e6ab02"
    marker_by_type = {"Mainstream/major waterway": "o", "Tributary/regional river": "^"}
    station_marker_size = 112 * (1.15 if not show_map_text else 1.0)
    for _, row in stations.iterrows():
        marker = marker_by_type[row["river_type"]]
        color = station_color
        ax.scatter(
            row["LongitudeMeasure"],
            row["LatitudeMeasure"],
            s=station_marker_size * 1.75,
            marker=marker,
            color="white",
            edgecolor="white",
            linewidth=0,
            alpha=0.96,
            zorder=7.8,
        )
        ax.scatter(
            row["LongitudeMeasure"],
            row["LatitudeMeasure"],
            s=station_marker_size,
            marker=marker,
            color=color,
            edgecolor="#2b2f33",
            linewidth=0.9,
            zorder=8,
        )
        if show_map_text:
            dx, dy = LABEL_OFFSETS[row["station_code"]]
            ax.annotate(
                row["station_code"],
                xy=(row["LongitudeMeasure"], row["LatitudeMeasure"]),
                xytext=(row["LongitudeMeasure"] + dx, row["LatitudeMeasure"] + dy),
                fontsize=16,
                color="#343434",
                fontweight="bold",
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="-", color="#8a9298", lw=1.25, shrinkA=4, shrinkB=4),
                zorder=9,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="#f7fbff", alpha=0.95)],
            )

    ax.set_xlim(BBOX[0], BBOX[2])
    ax.set_ylim(BBOX[1], BBOX[3])
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.tick_params(labelsize=10)
    ax.grid(True, color="#d5e2eb", linewidth=0.55, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#333333")
    add_north_arrow(ax)
    add_scale_bar(ax)

    fig.suptitle("Selected Yangtze River Basin Stations", x=0.505, y=0.945, fontsize=22, fontweight="bold")

    legend_fontsize = 9 if show_map_text else 14
    legend_marker_size = 7 if show_map_text else 12
    handles = [
        Line2D([0], [0], color=unified_river_color, lw=2.0 if show_map_text else 3.0, label="Rivers"),
        Patch(facecolor=unified_river_color, edgecolor=unified_river_color, alpha=0.13, label="Major lakes"),
    ]
    for river_type, marker in marker_by_type.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=station_color,
                markeredgecolor="#2b2f33",
                markersize=legend_marker_size,
                label=river_type,
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.045 if show_map_text else 0.03),
        ncol=4,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.8 if show_map_text else 2.3,
        columnspacing=2.2 if show_map_text else 1.5,
        labelspacing=0.75,
    )

    stem = f"yangtze_18_selected_stations{output_suffix}"
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    drawio_path = OUTPUT_DIR / f"{stem}.drawio"
    fig.savefig(png_path, dpi=288)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)
    export_drawio_from_svg(svg_path, drawio_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")
    print(f"Saved {drawio_path}")


if __name__ == "__main__":
    plot_map()
    plot_map(show_map_text=False, output_suffix="_no_map_text")
