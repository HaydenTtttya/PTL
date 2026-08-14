from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEL = REPO_ROOT / "data" / "station_mapping.xlsx"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "water_quality_processed"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "by_resolution_basin_station"
RESOLUTIONS = ("4h", "daily", "weekly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize water quality CSV files by resolution/basin/station."
    )
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL, help="Excel mapping file.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory that contains the 4h/daily/weekly folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination directory for the organized view.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output directory before writing a fresh copy.",
    )
    return parser.parse_args()


def safe_component(value: str) -> str:
    return str(value).strip().replace("/", "／")


def load_mapping(excel_path: Path) -> tuple[dict[str, list[dict[str, str]]], pd.DataFrame]:
    frame = (
        pd.read_excel(excel_path, usecols=["省份", "流域", "断面名称"])
        .dropna(subset=["省份", "流域", "断面名称"])
        .drop_duplicates()
    )
    mapping: dict[str, list[dict[str, str]]] = {}
    for record in frame.to_dict("records"):
        station = str(record["断面名称"]).strip()
        mapping.setdefault(station, []).append(
            {
                "province": str(record["省份"]).strip(),
                "basin": str(record["流域"]).strip(),
            }
        )
    return mapping, frame


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. Use --overwrite to rebuild it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def write_reports(
    output_root: Path,
    manifest_rows: list[dict[str, str]],
    unmatched_rows: list[dict[str, str]],
    ambiguous_rows: list[dict[str, str]],
) -> None:
    reports_root = output_root / "_reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(manifest_rows).to_csv(reports_root / "classification_manifest.csv", index=False)
    pd.DataFrame(unmatched_rows).drop_duplicates().to_csv(
        reports_root / "unmatched_stations.csv", index=False
    )
    pd.DataFrame(ambiguous_rows).drop_duplicates().to_csv(
        reports_root / "ambiguous_stations.csv", index=False
    )


def main() -> None:
    args = parse_args()
    mapping, frame = load_mapping(args.excel)
    prepare_output(args.output_root, args.overwrite)

    manifest_rows: list[dict[str, str]] = []
    unmatched_rows: list[dict[str, str]] = []
    ambiguous_rows: list[dict[str, str]] = []

    copied_files = 0
    matched_files = 0
    unmatched_files = 0
    ambiguous_files = 0

    for resolution in RESOLUTIONS:
        source_dir = args.data_root / resolution
        if not source_dir.exists():
            continue

        for source_file in sorted(source_dir.glob("*.csv")):
            station_name = source_file.stem.strip()
            station_records = mapping.get(station_name, [])

            if not station_records:
                destination_dir = (
                    args.output_root / resolution / "_未匹配_Excel缺失"
                )
                unmatched_rows.append(
                    {
                        "断面名称": station_name,
                        "resolution": resolution,
                        "source_file": str(source_file),
                    }
                )
                status = "unmatched"
                basin = ""
                province = ""
                unmatched_files += 1
            elif len(station_records) > 1:
                destination_dir = (
                    args.output_root / resolution / "_待确认_重名断面"
                )
                for station_record in station_records:
                    ambiguous_rows.append(
                        {
                            "断面名称": station_name,
                            "resolution": resolution,
                            "候选流域": station_record["basin"],
                            "候选省份": station_record["province"],
                            "source_file": str(source_file),
                        }
                    )
                status = "ambiguous"
                basin = ""
                province = ""
                ambiguous_files += 1
            else:
                station_record = station_records[0]
                basin = station_record["basin"]
                province = station_record["province"]
                destination_dir = (
                    args.output_root
                    / resolution
                    / safe_component(basin)
                )
                status = "matched"
                matched_files += 1

            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_file = destination_dir / f"{safe_component(station_name)}.csv"
            shutil.copy2(source_file, destination_file)

            manifest_rows.append(
                {
                    "status": status,
                    "resolution": resolution,
                    "断面名称": station_name,
                    "流域": basin,
                    "省份": province,
                    "source_file": str(source_file),
                    "destination_file": str(destination_file),
                }
            )
            copied_files += 1

    write_reports(args.output_root, manifest_rows, unmatched_rows, ambiguous_rows)

    unique_station_count = frame["断面名称"].nunique()
    print(f"Excel unique stations: {unique_station_count}")
    print(f"Copied files: {copied_files}")
    print(f"Matched files: {matched_files}")
    print(f"Ambiguous files: {ambiguous_files}")
    print(f"Unmatched files: {unmatched_files}")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
