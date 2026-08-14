import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEL = REPO_ROOT / "data" / "2023-2025_GX_water" / "阳朔交州桂花运江老街2023-2025水质数据.xlsx"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "water_quality_processed_tp_2023_2025"
FEATURE_MAPPING = {
    "permanganate_index": "CODMn",
    "DO": "DO",
    "TP": "TP",
    "pH": "pH",
}
FEATURE_COLUMNS = ["CODMn", "DO", "TP", "pH"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build cleaned 2023-2025 Guangxi water-quality datasets.")
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL, help="Source Excel file.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output dataset root.")
    return parser.parse_args()


def load_source_frame(excel_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(excel_path)
    required_columns = {"province", "basin", "section_name", "tm", "station_condition", *FEATURE_MAPPING.keys()}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Excel is missing required columns: {sorted(missing)}")

    normalized = frame.copy()
    normalized["tm"] = pd.to_datetime(normalized["tm"], errors="coerce")
    normalized["section_name"] = normalized["section_name"].fillna("").astype(str).str.strip()
    normalized["province"] = normalized["province"].fillna("").astype(str).str.strip()
    normalized["basin"] = normalized["basin"].fillna("").astype(str).str.strip()
    normalized["station_condition"] = normalized["station_condition"].fillna("").astype(str).str.strip()

    for source_column in FEATURE_MAPPING:
        normalized[source_column] = pd.to_numeric(normalized[source_column], errors="coerce")

    normalized = normalized.dropna(subset=["tm"])
    return normalized


def align_to_4h_floor(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.floor("4h")


def align_to_4h_ceil(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.ceil("4h")


def clean_station_frame(station_frame: pd.DataFrame, global_start: pd.Timestamp, global_end: pd.Timestamp):
    raw_row_count = len(station_frame)
    station_frame = station_frame.sort_values("tm").drop_duplicates("tm").copy()
    condition = station_frame["station_condition"].fillna("").astype(str).str.strip()
    normalized_condition = condition.replace("", "缺失")

    prepared = pd.DataFrame({"timestamp": station_frame["tm"]})
    for source_column, target_column in FEATURE_MAPPING.items():
        prepared[target_column] = station_frame[source_column]

    raw_missing_mask = prepared[FEATURE_COLUMNS].isna()
    raw_zero_mask = prepared[FEATURE_COLUMNS].eq(0)

    # Explicit maintenance rows should not be treated as trustworthy observations.
    maintenance_mask = normalized_condition.eq("维护")

    # Impossible or placeholder zero values show up in this export and should
    # become gaps so downstream interpolation can bridge them safely.
    prepared.loc[prepared["pH"] <= 0, "pH"] = np.nan
    prepared.loc[prepared["DO"] <= 0, "DO"] = np.nan
    prepared.loc[normalized_condition.ne("正常") & (prepared["CODMn"] <= 0), "CODMn"] = np.nan
    prepared.loc[normalized_condition.ne("正常") & (prepared["TP"] <= 0), "TP"] = np.nan
    prepared.loc[maintenance_mask, FEATURE_COLUMNS] = np.nan

    full_index = pd.date_range(start=global_start, end=global_end, freq="4h")
    reindexed = prepared.set_index("timestamp").reindex(full_index)
    reindexed.index.name = "timestamp"
    frame_4h = reindexed.reset_index()

    invalid_rows_4h = frame_4h[FEATURE_COLUMNS].isna().any(axis=1)
    return frame_4h, {
        "raw_rows": int(raw_row_count),
        "duplicate_timestamps_removed": int(raw_row_count - len(station_frame)),
        "maintenance_rows": int(maintenance_mask.sum()),
        "raw_missing_value_count": int(raw_missing_mask.sum().sum()),
        "raw_missing_row_count": int(raw_missing_mask.any(axis=1).sum()),
        "raw_zero_value_count": int(raw_zero_mask.sum().sum()),
        "raw_zero_row_count": int(raw_zero_mask.any(axis=1).sum()),
        "cleaned_missing_timestamp_count": int(len(full_index) - len(prepared)),
        "cleaned_gap_row_count_4h": int(invalid_rows_4h.sum()),
        "cleaned_gap_value_count_4h": int(frame_4h[FEATURE_COLUMNS].isna().sum().sum()),
        "condition_counts": normalized_condition.value_counts().to_dict(),
    }


def resample_frame(frame_4h: pd.DataFrame, freq: str) -> pd.DataFrame:
    indexed = frame_4h.set_index("timestamp")
    resampled = indexed[FEATURE_COLUMNS].resample(freq).mean().interpolate(limit_direction="both").dropna()
    return resampled.reset_index()


def save_station_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def build_station_meta_row(station_name: str, station_frame: pd.DataFrame, frame_4h: pd.DataFrame, daily_frame: pd.DataFrame, weekly_frame: pd.DataFrame, clean_stats):
    province = station_frame["province"].replace("", np.nan).dropna().astype(str).str.strip().iloc[0]
    basin = station_frame["basin"].replace("", np.nan).dropna().astype(str).str.strip().iloc[0]
    return {
        "断面名称": station_name,
        "省份": province,
        "流域": basin,
        "原始记录数": clean_stats["raw_rows"],
        "维护记录数": clean_stats["maintenance_rows"],
        "原始缺失值数": clean_stats["raw_missing_value_count"],
        "原始零值数": clean_stats["raw_zero_value_count"],
        "补齐缺失时间点数_4h": clean_stats["cleaned_missing_timestamp_count"],
        "Gap行数_4h": clean_stats["cleaned_gap_row_count_4h"],
        "Gap值数_4h": clean_stats["cleaned_gap_value_count_4h"],
        "记录数_4h": int(len(frame_4h)),
        "记录数_daily": int(len(daily_frame)),
        "记录数_weekly": int(len(weekly_frame)),
        "时间起": frame_4h["timestamp"].min(),
        "时间止": frame_4h["timestamp"].max(),
        "状态统计": clean_stats["condition_counts"],
    }


def main():
    args = parse_args()
    source_frame = load_source_frame(args.excel)
    output_root = args.output_root
    meta_rows = []

    global_start = align_to_4h_floor(source_frame["tm"].min())
    global_end = align_to_4h_ceil(source_frame["tm"].max())

    grouped = source_frame.groupby("section_name", sort=True)
    for station_name, station_frame in grouped:
        frame_4h, clean_stats = clean_station_frame(station_frame, global_start=global_start, global_end=global_end)
        daily_frame = resample_frame(frame_4h, "1D")
        weekly_frame = resample_frame(frame_4h, "1W")

        save_station_frame(frame_4h[["timestamp", *FEATURE_COLUMNS]], output_root / "4h" / f"{station_name}.csv")
        save_station_frame(daily_frame[["timestamp", *FEATURE_COLUMNS]], output_root / "daily" / f"{station_name}.csv")
        save_station_frame(weekly_frame[["timestamp", *FEATURE_COLUMNS]], output_root / "weekly" / f"{station_name}.csv")

        meta_rows.append(
            build_station_meta_row(
                station_name,
                station_frame,
                frame_4h,
                daily_frame,
                weekly_frame,
                clean_stats,
            )
        )
        print(
            f"{station_name}: 4h={len(frame_4h)} rows, "
            f"daily={len(daily_frame)} rows, weekly={len(weekly_frame)} rows, "
            f"gap_rows_4h={clean_stats['cleaned_gap_row_count_4h']}, "
            f"maintenance_rows={clean_stats['maintenance_rows']}, "
            f"missing_timestamps={clean_stats['cleaned_missing_timestamp_count']}"
        )

    if meta_rows:
        output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(meta_rows).to_csv(output_root / "station_meta.csv", index=False)
        print(f"Saved station_meta.csv with {len(meta_rows)} station(s) to {output_root}")


if __name__ == "__main__":
    main()
