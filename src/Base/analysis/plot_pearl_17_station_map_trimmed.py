from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from shapely.geometry import box
from shapely.ops import unary_union


REPO_ROOT = Path(__file__).resolve().parents[3]
RIVER_DIR = REPO_ROOT / "data" / "我国河流、湖泊数据集" / "我国河流数据"
PROVINCE_SHP = (
    REPO_ROOT
    / "data"
    / "ChinaAdminDivisonSHP-24.02.06"
    / "2. Province"
    / "province.shp"
)
COORD_PATH = (
    REPO_ROOT
    / "results"
    / "summary"
    / "current_all_tested_stations_overall_nse"
    / "均衡十五站方案"
    / "experiment_17_station_coordinates_with_recommended_level1.csv"
)
CLASS_PATH = (
    REPO_ROOT
    / "results"
    / "summary"
    / "current_all_tested_stations_overall_nse"
    / "均衡十五站方案_新增两站"
    / "站点分类.csv"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "figures"
    / "current_all_tested_stations_overall_nse"
    / "均衡十五站方案_新增两站"
)

BBOX = (102.7, 21.8, 116.75, 26.58)
CONFLUENCE_BUFFER_M = 300
XP_CONNECTION_BBOX = (112.8, 22.4, 116.9, 25.55)

STATION_ABBREVIATIONS = {
    "花山水库出水口": "HRO",
    "坝草": "BC",
    "蔗香南": "ZXN",
    "黄泥河": "HNR",
    "打邦": "DB",
    "三都桥": "SDB",
    "大化": "DH",
    "武林渡口": "WLF",
    "老口": "LK",
    "上中": "SZ",
    "桂花": "GH",
    "象州运江老街": "XYO",
    "鸦岗": "YG",
    "布洲": "BZ",
    "大墩": "DD",
    "新铺": "XP",
    "深圳河口": "SRE",
}

STATION_RIVER_TERMS = [
    "南盘江",
    "北盘江",
    "黄泥河",
    "小黄泥河",
    "打邦河",
    "白水河",
    "柳江",
    "都柳江",
    "红水河",
    "浔江",
    "邕江",
    "郁江",
    "左江",
    "桂江",
    "珠江",
    "磨刀门",
    "西江干流水道",
    "东江北干流",
    "石窑河",
    "石窟河",
    "深圳河",
]

XP_CONNECTION_RIVER_TERMS = [
    "石窑河",
    "石窟河",
    "梅江",
    "西枝江",
    "东江(寻邬水)",
    "东江南支流",
]

PROVINCE_NAMES = {
    "云南省": "Yunnan",
    "贵州省": "Guizhou",
    "广西壮族自治区": "Guangxi",
    "广东省": "Guangdong",
    "湖南省": "Hunan",
    "江西省": "Jiangxi",
    "福建省": "Fujian",
}

PROVINCE_LABEL_POSITIONS = {
    "Yunnan": (103.65, 23.90),
    "Guizhou": (106.75, 25.72),
    "Guangxi": (109.10, 22.70),
    "Guangdong": (114.40, 24.10),
    "Hunan": (111.65, 26.10),
    "Jiangxi": (115.35, 25.70),
    "Fujian": (116.20, 25.12),
}

LABEL_OFFSETS = {
    "HRO": (-0.25, 0.25),
    "BC": (0.24, -0.12),
    "ZXN": (0.23, 0.24),
    "HNR": (-0.25, 0.08),
    "DB": (0.22, 0.22),
    "SDB": (0.24, 0.28),
    "DH": (-0.23, -0.18),
    "WLF": (-0.24, -0.23),
    "LK": (0.28, -0.44),
    "SZ": (-0.30, 0.12),
    "GH": (0.24, 0.25),
    "XYO": (-0.16, 0.32),
    "YG": (-0.22, 0.22),
    "BZ": (-0.22, -0.24),
    "DD": (0.25, 0.22),
    "XP": (-0.22, 0.24),
    "SRE": (0.26, 0.34),
}

RIVER_LABELS = [
    ("Huangni River", 104.05, 25.12),
    ("Dabang River", 105.35, 25.70),
    ("Beipan River", 105.70, 25.48),
    ("Nanpan River", 105.35, 25.05),
    ("Duliu River", 108.45, 25.72),
    ("Liu River", 109.18, 24.34),
    ("Gui River", 110.76, 24.60),
    ("Hongshui River", 108.48, 23.82),
    ("Yong River", 108.28, 23.10),
    ("Zuo River", 107.82, 22.53),
    ("Xun River", 110.95, 23.52),
    ("Pearl River West Channel", 112.60, 23.55),
    ("Dongjiang N. Mainstream", 113.62, 23.08),
    ("Modao Men Waterway", 113.20, 22.82),
    ("Shenzhen River", 114.72, 22.56),
    ("Shiku River", 115.66, 24.45),
]


def load_rivers() -> gpd.GeoDataFrame:
    parts: list[gpd.GeoDataFrame] = []
    specs = [
        ("5th-order rivers", "5", "我国五级河流.shp"),
        ("4th-order rivers", "4", "我国四级河流.shp"),
        ("2nd-3rd-order rivers", "23", "我国二三级河流.shp"),
        ("1st-order rivers", "1", "我国一级河流.shp"),
    ]
    for label, level_key, file_name in specs:
        gdf = gpd.read_file(RIVER_DIR / file_name).to_crs("EPSG:4326")
        if level_key == "1":
            gdf = gdf[gdf["LEVEL_RIVE"].eq(1)].copy()
        elif level_key == "23":
            gdf = gdf[gdf["LEVEL_RIVE"].isin([2, 3])].copy()
        gdf = gdf.cx[BBOX[0] : BBOX[2], BBOX[1] : BBOX[3]].copy()
        gdf["legend_label"] = label
        gdf["level_key"] = level_key
        parts.append(gdf[["NAME", "legend_label", "level_key", "geometry"]])
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")


def filter_rivers(rivers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    river_names = rivers["NAME"].fillna("").astype(str)
    is_first_order = rivers["level_key"].eq("1")
    is_station_river = river_names.apply(
        lambda name: any(term in name for term in STATION_RIVER_TERMS)
    )
    is_xp_connection_name = river_names.apply(
        lambda name: any(term in name for term in XP_CONNECTION_RIVER_TERMS)
    )
    is_xp_connection = is_xp_connection_name & rivers.geometry.intersects(
        box(*XP_CONNECTION_BBOX)
    )
    metric = rivers.to_crs("EPSG:3857")
    station_river_union = unary_union(
        metric.loc[is_station_river | is_xp_connection, "geometry"].to_list()
    )
    intersects_station_rivers = metric.geometry.intersects(
        station_river_union.buffer(CONFLUENCE_BUFFER_M)
    )
    keep = is_first_order | is_station_river | is_xp_connection | intersects_station_rivers
    filtered = rivers.loc[keep].copy()
    filtered["filter_reason"] = "confluence_or_station"
    filtered.loc[is_first_order.loc[keep].to_numpy(), "filter_reason"] = "first_order"
    filtered.loc[is_station_river.loc[keep].to_numpy(), "filter_reason"] = "station_river"
    filtered.loc[
        is_xp_connection.loc[keep].to_numpy(),
        "filter_reason",
    ] = "xp_connection"
    return filtered


def load_stations() -> gpd.GeoDataFrame:
    coords = pd.read_csv(COORD_PATH, encoding="utf-8-sig")
    classes = pd.read_csv(CLASS_PATH, encoding="utf-8-sig")
    stations = coords.merge(
        classes[["station", "river_reach", "river_type", "站点顺序"]],
        left_on="实验站点",
        right_on="station",
        how="left",
    ).sort_values("站点顺序")
    stations["abbr"] = stations["实验站点"].map(STATION_ABBREVIATIONS)
    missing = stations.loc[stations["abbr"].isna(), "实验站点"].tolist()
    if missing:
        raise ValueError(f"Missing station abbreviations: {missing}")
    return gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["经度"], stations["纬度"]),
        crs="EPSG:4326",
    )


def add_scale_bar(ax: plt.Axes, x0: float = 114.55, y0: float = 21.98) -> None:
    # 200 km converted to longitude degrees at the local latitude.
    length_deg = 200 / (111.32 * 0.928)
    x1 = x0 + length_deg
    ax.plot([x0, x1], [y0, y0], color="#222222", lw=2.2, zorder=8)
    ax.plot([x0, x0], [y0 - 0.045, y0 + 0.045], color="#222222", lw=2.2, zorder=8)
    ax.plot([x1, x1], [y0 - 0.045, y0 + 0.045], color="#222222", lw=2.2, zorder=8)
    ax.text(
        (x0 + x1) / 2,
        y0 + 0.12,
        "200 km",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#222222",
        family="Arial",
    )


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "",
        xy=(116.12, 26.20),
        xytext=(116.12, 25.55),
        arrowprops=dict(facecolor="#111111", edgecolor="#111111", width=4, headwidth=18),
        zorder=9,
    )
    ax.text(
        116.12,
        26.25,
        "N",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        family="Arial",
        color="#111111",
    )


def save_station_code_table(stations: gpd.GeoDataFrame) -> None:
    table = stations[
        [
            "实验站点",
            "abbr",
            "责任省份",
            "责任城市",
            "所属河流（湖库）",
            "river_reach",
            "river_type",
            "经度",
            "纬度",
        ]
    ].rename(
        columns={
            "实验站点": "station_cn",
            "abbr": "station_code",
            "责任省份": "province_cn",
            "责任城市": "city_cn",
            "所属河流（湖库）": "river_cn",
            "river_reach": "reach_cn",
            "river_type": "river_type_cn",
            "经度": "longitude",
            "纬度": "latitude",
        }
    )
    table.to_csv(
        OUTPUT_DIR / "experiment_17_station_abbreviations_for_map.csv",
        index=False,
        encoding="utf-8-sig",
    )


def export_drawio_from_svg(svg_path: Path, drawio_path: Path) -> None:
    svg_text = svg_path.read_text(encoding="utf-8")
    encoded_svg = quote(svg_text, safe="")
    style = (
        "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
        "imageAspect=1;aspect=fixed;image=data:image/svg+xml,"
        f"{encoded_svg};"
    )
    xml = f'''<mxfile host="app.diagrams.net" modified="2026-06-04T00:00:00.000Z" agent="Codex" version="24.7.17" type="device">
  <diagram id="pearl-river-map" name="Pearl River Map">
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


def plot_map(
    show_map_text: bool = True,
    output_suffix: str = "",
    *,
    show_title: bool = True,
    journal_style: bool = False,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rivers = filter_rivers(load_rivers())
    stations = load_stations()
    save_station_code_table(stations)

    provinces = gpd.read_file(PROVINCE_SHP).to_crs("EPSG:4326")
    province_subset = provinces.cx[BBOX[0] : BBOX[2], BBOX[1] : BBOX[3]].copy()

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(16, 9.458333), dpi=288)
    if show_map_text:
        ax = fig.add_axes([0.055, 0.21, 0.90, 0.67])
    else:
        ax = fig.add_axes([0.055, 0.24, 0.90, 0.64])

    ax.set_facecolor("#f7fbff")
    province_subset.plot(
        ax=ax,
        facecolor="#f4f6f7",
        edgecolor="#bfc5c9",
        linewidth=0.55,
        zorder=0,
    )
    province_subset.boundary.plot(ax=ax, color="#b9bec2", linewidth=0.55, zorder=1)

    if show_map_text:
        for province, label in PROVINCE_NAMES.items():
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
                fontweight="normal",
                alpha=0.72,
                zorder=1.5,
            )

    unified_river_color = "#176fa8"
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
                rotation=0,
                ha="center",
                va="center",
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="#f7fbff", alpha=0.95)],
            )

    station_color = "#d1495b"
    marker_by_type = {
        "干流/主要水道": "o",
        "支流/区域河流": "^",
    }
    station_marker_size = 112 * (1.15 if not show_map_text else 1.0)
    for (_, row) in stations.iterrows():
        color = station_color
        marker = marker_by_type[row["river_type"]]
        ax.scatter(
            row["经度"],
            row["纬度"],
            s=station_marker_size * 1.75,
            marker=marker,
            color="white",
            edgecolor="white",
            linewidth=0,
            alpha=0.96,
            zorder=7.8,
        )
        ax.scatter(
            row["经度"],
            row["纬度"],
            s=station_marker_size,
            marker=marker,
            color=color,
            edgecolor="#2b2f33",
            linewidth=0.9,
            zorder=8,
        )
        if show_map_text:
            dx, dy = LABEL_OFFSETS[row["abbr"]]
            label_x = row["经度"] + dx
            label_y = row["纬度"] + dy
            ax.annotate(
                row["abbr"],
                xy=(row["经度"], row["纬度"]),
                xytext=(label_x, label_y),
                fontsize=16,
                color="#343434",
                fontweight="bold",
                ha="center",
                va="center",
                arrowprops=dict(
                    arrowstyle="-",
                    color="#8a9298",
                    lw=1.25,
                    shrinkA=4,
                    shrinkB=4,
                ),
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

    if show_title:
        fig.suptitle(
            "Selected Pearl River Basin Stations",
            x=0.505,
            y=0.945,
            fontsize=22,
            fontweight="bold",
        )

    legend_fontsize = 9 if show_map_text else 14
    legend_marker_size = 7 if show_map_text else 12
    river_handles = [
        Line2D(
            [0],
            [0],
            color=unified_river_color,
            lw=2.0 if show_map_text else 3.0,
            label="Rivers",
        ),
    ]
    station_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=station_color,
            markeredgecolor="#2b2f33",
            markersize=legend_marker_size,
            label="Mainstream/major waterway",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=station_color,
            markeredgecolor="#2b2f33",
            markersize=legend_marker_size,
            label="Tributary/regional river",
        ),
    ]
    fig.legend(
        handles=river_handles + station_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.045 if show_map_text else 0.03),
        ncol=3,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.8 if show_map_text else 2.3,
        columnspacing=2.6 if show_map_text else 1.8,
        labelspacing=0.75,
    )

    png_path = OUTPUT_DIR / f"experiment_17_selected_pearl_river_basin_stations_trimmed_abbrev_provinces{output_suffix}.png"
    pdf_path = OUTPUT_DIR / f"experiment_17_selected_pearl_river_basin_stations_trimmed_abbrev_provinces{output_suffix}.pdf"
    svg_path = OUTPUT_DIR / f"experiment_17_selected_pearl_river_basin_stations_trimmed_abbrev_provinces{output_suffix}.svg"
    drawio_path = OUTPUT_DIR / f"experiment_17_selected_pearl_river_basin_stations_trimmed_abbrev_provinces{output_suffix}.drawio"
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.03} if journal_style else {}
    fig.savefig(png_path, dpi=450 if journal_style else 288, **save_kwargs)
    fig.savefig(pdf_path, **save_kwargs)
    fig.savefig(svg_path, **save_kwargs)
    plt.close(fig)
    export_drawio_from_svg(svg_path, drawio_path)

    summary = (
        rivers.groupby(["legend_label", "filter_reason"])
        .size()
        .rename("kept_lines")
        .reset_index()
        .sort_values(["legend_label", "filter_reason"])
    )
    summary.to_csv(
        OUTPUT_DIR / "experiment_17_trimmed_river_filter_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")
    print(f"Saved {drawio_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    plot_map()
    plot_map(show_map_text=False, output_suffix="_no_map_text")
