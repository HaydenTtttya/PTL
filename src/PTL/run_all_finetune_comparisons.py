import argparse
import csv
import datetime
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

from finetune import (
    ARTIFACTS_ROOT,
    PRETRAIN_RUNS_DIR,
    build_daily_target_config,
    build_finetune_preset,
    find_latest_pretrain_run,
    main,
)


SEED = 42
COMPARISON_ROOT = Path(ARTIFACTS_ROOT) / "finetune" / "comparisons"

ROOT_FILE_RENAMES = {
    "summary.json": "运行总览_summary.json",
}
STAGE_DIR_RENAMES = {
    "stage1_weekly": "阶段1_周级训练",
    "stage2_4d": "阶段2_4天级训练",
    "stage3_daily": "阶段3_日级训练",
}
STAGE_FILE_RENAMES = {
    "history.csv": "训练历史_history.csv",
    "metrics.csv": "评估指标_metrics.csv",
    "predictions.csv": "预测明细_predictions.csv",
    "meta.json": "运行元信息_meta.json",
}
RUN_SPECS = [
    {
        "sort_index": 0,
        "key": "default_finetune",
        "kind": "default_finetune",
        "preset_name": "",
        "label": "00_默认Finetune_周4天日渐进",
        "description": "finetune.py 默认配置",
    },
    {
        "sort_index": 1,
        "key": "default_daily_target",
        "kind": "default_daily_target",
        "preset_name": "",
        "label": "01_默认DailyTarget_软缺口6步",
        "description": "run_daily_target.py 默认配置",
    },
    {
        "sort_index": 2,
        "key": "target75_v1",
        "kind": "preset",
        "preset_name": "target75_v1",
        "label": "02_目标75_v1",
        "description": "target75 首版目标窗口策略",
    },
    {
        "sort_index": 3,
        "key": "target75_v2_soft3",
        "kind": "preset",
        "preset_name": "target75_v2_soft3",
        "label": "03_目标75_v2_软缺口3步",
        "description": "target75 v2，日级阶段允许 3 步软缺口",
    },
    {
        "sort_index": 4,
        "key": "target75_v2_soft6",
        "kind": "preset",
        "preset_name": "target75_v2_soft6",
        "label": "04_目标75_v2_软缺口6步",
        "description": "target75 v2，日级阶段允许 6 步软缺口",
    },
    {
        "sort_index": 5,
        "key": "nh4n_daily_v1",
        "kind": "preset",
        "preset_name": "nh4n_daily_v1",
        "label": "05_NH4N日级强化_v1",
        "description": "NH4N 目标强化、事件采样",
    },
    {
        "sort_index": 6,
        "key": "nh4n_floor_guard_v1",
        "kind": "preset",
        "preset_name": "nh4n_floor_guard_v1",
        "label": "06_NH4N底线保护_v1",
        "description": "NH4N floor guard 后处理",
    },
    {
        "sort_index": 7,
        "key": "base_0824_v1",
        "kind": "preset",
        "preset_name": "base_0824_v1",
        "label": "07_截止0824_基础版_v1",
        "description": "截至 2025-08-24 的基础配置",
    },
    {
        "sort_index": 8,
        "key": "weather_base_0824_v1",
        "kind": "preset",
        "preset_name": "weather_base_0824_v1",
        "label": "08_截止0824_气象基础版_v1",
        "description": "截至 2025-08-24，加入气象特征",
    },
    {
        "sort_index": 9,
        "key": "nh4n_floor_guard_0824_v1",
        "kind": "preset",
        "preset_name": "nh4n_floor_guard_0824_v1",
        "label": "09_截止0824_NH4N底线保护_v1",
        "description": "截至 2025-08-24，NH4N floor guard",
    },
    {
        "sort_index": 10,
        "key": "nh4n_weather_dual_station_v1",
        "kind": "preset",
        "preset_name": "nh4n_weather_dual_station_v1",
        "label": "10_截止0824_NH4N双站气象_v1",
        "description": "截至 2025-08-24，NH4N + 双站气象 + floor guard",
    },
    {
        "sort_index": 11,
        "key": "nh4n_weather_two_stage_v1",
        "kind": "preset",
        "preset_name": "nh4n_weather_two_stage_v1",
        "label": "11_截止0824_NH4N气象双阶段_v1",
        "description": "截至 2025-08-24，NH4N + 气象 + 双阶段损失",
    },
    {
        "sort_index": 12,
        "key": "nh4n_weather_two_stage_v2",
        "kind": "preset",
        "preset_name": "nh4n_weather_two_stage_v2",
        "label": "12_截止0824_NH4N气象双阶段_v2",
        "description": "截至 2025-08-24，NH4N + 气象 + 双阶段损失 v2",
    },
    {
        "sort_index": 13,
        "key": "nh4n_weather_two_stage_v3",
        "kind": "preset",
        "preset_name": "nh4n_weather_two_stage_v3",
        "label": "13_截止0824_NH4N气象双阶段_v3",
        "description": "截至 2025-08-24，NH4N + 气象 + 双阶段损失 v3",
    },
    {
        "sort_index": 14,
        "key": "nh4n_two_stage_v1",
        "kind": "preset",
        "preset_name": "nh4n_two_stage_v1",
        "label": "14_NH4N双阶段_v1",
        "description": "NH4N 双阶段损失，不含气象特征",
    },
    {
        "sort_index": 15,
        "key": "tp_overall_v1",
        "kind": "preset",
        "preset_name": "tp_overall_v1",
        "label": "15_TP整体_v1",
        "description": "TP 线轻度 mse_nse，保持整体优化目标",
    },
    {
        "sort_index": 16,
        "key": "tp_overall_v2",
        "kind": "preset",
        "preset_name": "tp_overall_v2",
        "label": "16_TP整体_v2_42天",
        "description": "TP 整体版 v2，在 v1 基础上将日级窗口扩到 42 天",
    },
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def log_print(*args, file_handle=None, **kwargs):
    text = " ".join(str(arg) for arg in args)
    print(text, **kwargs)
    if file_handle is not None:
        print(text, file=file_handle, **kwargs)
        file_handle.flush()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def try_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_metrics_csv(path):
    if not path.exists():
        return {}

    rows = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        first_column = reader.fieldnames[0] if reader.fieldnames else ""
        for row in reader:
            row_name = row.pop(first_column, None)
            if row_name is None:
                continue
            rows[row_name] = {key: try_float(value) for key, value in row.items()}
    return rows


def build_custom_config(spec, save_dir):
    if spec["kind"] == "default_finetune":
        config = {}
    elif spec["kind"] == "default_daily_target":
        config = build_daily_target_config()
    elif spec["kind"] == "preset":
        config = build_finetune_preset(spec["preset_name"])
    else:
        raise ValueError(f"未知 run kind: {spec['kind']}")

    config = dict(config or {})
    config["save_dir"] = str(save_dir)
    return config


def move_and_rename_run_outputs(raw_run_dirs, target_dir):
    if target_dir.exists():
        raise FileExistsError(f"目标目录已存在: {target_dir}")

    if len(raw_run_dirs) == 1:
        shutil.move(str(raw_run_dirs[0]), str(target_dir))
    else:
        target_dir.mkdir(parents=True, exist_ok=False)
        for raw_dir in raw_run_dirs:
            station_target = target_dir / raw_dir.name
            shutil.move(str(raw_dir), str(station_target))


def rename_run_files(target_dir):
    summary_path = target_dir / "summary.json"
    summary_data = read_json(summary_path) if summary_path.exists() else None

    stage_dir_map = {}
    for original_name, renamed_name in STAGE_DIR_RENAMES.items():
        source_dir = target_dir / original_name
        if not source_dir.exists():
            continue
        destination_dir = target_dir / renamed_name
        source_dir.rename(destination_dir)
        stage_dir_map[original_name] = destination_dir

    for stage_dir in list(target_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        for original_name, renamed_name in STAGE_FILE_RENAMES.items():
            source_file = stage_dir / original_name
            if source_file.exists():
                source_file.rename(stage_dir / renamed_name)

    if summary_data is not None:
        summary_data["save_dir"] = str(target_dir)
        for stage in summary_data.get("stages", []):
            stage_name = stage.get("stage_name")
            stage_dir = stage_dir_map.get(stage_name)
            if stage_dir is not None:
                stage["save_dir"] = str(stage_dir)

    for original_name, renamed_name in ROOT_FILE_RENAMES.items():
        source_file = target_dir / original_name
        if source_file.exists():
            source_file.rename(target_dir / renamed_name)

    if summary_data is not None:
        write_json(target_dir / ROOT_FILE_RENAMES["summary.json"], summary_data)

    for stage_name, stage_dir in stage_dir_map.items():
        meta_path = stage_dir / STAGE_FILE_RENAMES["meta.json"]
        if not meta_path.exists():
            continue
        meta_data = read_json(meta_path)
        meta_data["save_dir"] = str(stage_dir)
        write_json(meta_path, meta_data)

    return stage_dir_map


def collect_stage_meta(target_dir, stage_dir_map):
    metadata = {}
    for stage_name, stage_dir in stage_dir_map.items():
        meta_path = stage_dir / STAGE_FILE_RENAMES["meta.json"]
        if meta_path.exists():
            metadata[stage_name] = read_json(meta_path)
    return metadata


def write_run_info(target_dir, spec, pretrain_dir, run_seconds, log_path, stage_dir_map):
    run_info = {
        "key": spec["key"],
        "label": spec["label"],
        "kind": spec["kind"],
        "preset_name": spec["preset_name"],
        "description": spec["description"],
        "seed": SEED,
        "pretrain_model_dir": str(pretrain_dir),
        "run_seconds": run_seconds,
        "log_file": str(log_path),
        "run_dir": str(target_dir),
        "stage_directories": {stage_name: str(path) for stage_name, path in stage_dir_map.items()},
    }
    write_json(target_dir / "本次运行说明_run_info.json", run_info)


def build_result_row(spec, target_dir, log_path, stage_dir_map, error_message=None):
    row = {
        "sort_index": spec["sort_index"],
        "key": spec["key"],
        "label": spec["label"],
        "kind": spec["kind"],
        "preset_name": spec["preset_name"],
        "description": spec["description"],
        "status": "failed" if error_message else "completed",
        "error_message": error_message or "",
        "run_dir": str(target_dir) if target_dir is not None else "",
        "log_file": str(log_path),
        "final_stage_name": "",
        "final_stage_dir": "",
        "best_val_loss": None,
        "best_val_nse": None,
        "test_loss": None,
        "test_nse": None,
        "monitor_metric": "",
        "monitor_feature": "",
        "total_train_seconds": None,
        "overall_mae": None,
        "overall_rmse": None,
        "overall_nse": None,
        "overall_mape": None,
        "focus_feature_name": "",
        "focus_mae": None,
        "focus_rmse": None,
        "focus_nse": None,
        "focus_mape": None,
    }
    if target_dir is None or error_message:
        return row

    summary_path = target_dir / ROOT_FILE_RENAMES["summary.json"]
    if not summary_path.exists():
        row["status"] = "failed"
        row["error_message"] = "缺少运行总览_summary.json"
        return row

    summary_data = read_json(summary_path)
    row["status"] = summary_data.get("status", row["status"])
    stages = summary_data.get("stages", [])
    if not stages:
        return row

    final_stage = stages[-1]
    final_stage_name = final_stage.get("stage_name", "")
    final_stage_dir = stage_dir_map.get(final_stage_name)
    stage_metadata = collect_stage_meta(target_dir, stage_dir_map)
    total_train_seconds = sum(
        meta.get("train_seconds", 0.0) or 0.0 for meta in stage_metadata.values()
    )

    row.update(
        {
            "final_stage_name": final_stage_name,
            "final_stage_dir": str(final_stage_dir) if final_stage_dir is not None else "",
            "best_val_loss": final_stage.get("best_val_loss"),
            "best_val_nse": final_stage.get("best_val_nse"),
            "test_loss": final_stage.get("test_loss"),
            "test_nse": final_stage.get("test_nse"),
            "total_train_seconds": total_train_seconds,
        }
    )

    final_meta = stage_metadata.get(final_stage_name, {})
    row["monitor_metric"] = final_meta.get("monitor_metric", "")
    row["monitor_feature"] = final_meta.get("monitor_feature", "") or ""

    focus_feature_name = final_meta.get("monitor_feature", "") or ""
    if not focus_feature_name:
        feature_columns = final_meta.get("feature_columns") or []
        if len(feature_columns) >= 3:
            focus_feature_name = feature_columns[2]
    row["focus_feature_name"] = focus_feature_name

    if final_stage_dir is not None:
        metrics = read_metrics_csv(final_stage_dir / STAGE_FILE_RENAMES["metrics.csv"])
        overall_metrics = metrics.get("__overall__", {})
        focus_metrics = metrics.get(focus_feature_name, {}) if focus_feature_name else {}
        row.update(
            {
                "overall_mae": overall_metrics.get("MAE"),
                "overall_rmse": overall_metrics.get("RMSE"),
                "overall_nse": overall_metrics.get("NSE"),
                "overall_mape": overall_metrics.get("MAPE"),
                "focus_mae": focus_metrics.get("MAE"),
                "focus_rmse": focus_metrics.get("RMSE"),
                "focus_nse": focus_metrics.get("NSE"),
                "focus_mape": focus_metrics.get("MAPE"),
            }
        )

    for stage in stages:
        stage_name = stage.get("stage_name")
        if not stage_name:
            continue
        row[f"{stage_name}_best_val_nse"] = stage.get("best_val_nse")
        row[f"{stage_name}_test_nse"] = stage.get("test_nse")

    return row


def write_comparison_csv(output_path, rows):
    base_columns = [
        "sort_index",
        "key",
        "label",
        "kind",
        "preset_name",
        "description",
        "status",
        "error_message",
        "final_stage_name",
        "final_stage_dir",
        "best_val_loss",
        "best_val_nse",
        "test_loss",
        "test_nse",
        "monitor_metric",
        "monitor_feature",
        "total_train_seconds",
        "overall_mae",
        "overall_rmse",
        "overall_nse",
        "overall_mape",
        "focus_feature_name",
        "focus_mae",
        "focus_rmse",
        "focus_nse",
        "focus_mape",
        "run_dir",
        "log_file",
    ]
    extra_columns = []
    for row in rows:
        for key in row.keys():
            if key not in base_columns and key not in extra_columns:
                extra_columns.append(key)
    fieldnames = base_columns + sorted(extra_columns)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["sort_index"]):
            writer.writerow(row)


def format_metric(value):
    if value is None or value == "":
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def write_comparison_markdown(output_path, rows, comparison_root, pretrain_dir):
    completed_rows = [row for row in rows if row["status"] == "completed" and row["overall_nse"] is not None]
    ranked_rows = sorted(completed_rows, key=lambda item: item["overall_nse"], reverse=True)

    lines = [
        "# 预设与默认结果全面对比",
        "",
        f"- 结果目录：`{comparison_root}`",
        f"- 预训练模型：`{pretrain_dir}`",
        f"- 固定随机种子：`{SEED}`",
        f"- 运行组数：`{len(rows)}`",
        "",
        "## 综合排序（按最终阶段 overall NSE 从高到低）",
        "",
        "| 排名 | 名称 | 重点指标 | overall NSE | focus NSE | test_nse | 训练时长(s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for index, row in enumerate(ranked_rows, start=1):
        lines.append(
            "| {rank} | {label} | {focus_feature_name} | {overall_nse} | {focus_nse} | {test_nse} | {seconds} |".format(
                rank=index,
                label=row["label"],
                focus_feature_name=row["focus_feature_name"] or "-",
                overall_nse=format_metric(row["overall_nse"]),
                focus_nse=format_metric(row["focus_nse"]),
                test_nse=format_metric(row["test_nse"]),
                seconds=format_metric(row["total_train_seconds"]),
            )
        )

    failed_rows = [row for row in rows if row["status"] != "completed"]
    lines.extend(
        [
            "",
            "## 运行清单",
            "",
            "| 名称 | 类型 | 预设名 | 状态 | 最终阶段 | 结果目录 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: item["sort_index"]):
        lines.append(
            "| {label} | {kind} | {preset_name} | {status} | {final_stage_name} | `{run_dir}` |".format(
                label=row["label"],
                kind=row["kind"],
                preset_name=row["preset_name"] or "-",
                status=row["status"],
                final_stage_name=row["final_stage_name"] or "-",
                run_dir=row["run_dir"] or "-",
            )
        )

    if failed_rows:
        lines.extend(["", "## 失败项", ""])
        for row in failed_rows:
            lines.append(f"- {row['label']}: {row['error_message'] or '未知错误'}")

    lines.extend(
        [
            "",
            "## 文件命名说明",
            "",
            "- `运行总览_summary.json`：单次运行汇总。",
            "- `本次运行说明_run_info.json`：这次批量任务补充生成的运行说明。",
            "- `阶段1_周级训练` / `阶段2_4天级训练` / `阶段3_日级训练`：按阶段拆分后的结果目录。",
            "- `训练历史_history.csv`：训练过程曲线。",
            "- `评估指标_metrics.csv`：各指标汇总。",
            "- `预测明细_predictions.csv`：时间戳、真实值、预测值明细。",
            "- `运行元信息_meta.json`：该阶段配置、监控指标、耗时等信息。",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single_spec(spec, comparison_root, pretrain_dir, logs_dir, batch_log_handle):
    target_dir = comparison_root / spec["label"]
    run_log_path = logs_dir / f"{spec['label']}_运行日志.log"
    before_dirs = {path.name for path in comparison_root.iterdir() if path.is_dir()}
    before_dirs.discard("logs")

    start_time = time.time()
    raw_result_rows = None
    error_message = None
    stage_dir_map = {}

    log_print(
        f"[开始] {spec['label']} | kind={spec['kind']} | preset={spec['preset_name'] or '默认'}",
        file_handle=batch_log_handle,
    )

    with run_log_path.open("w", encoding="utf-8") as run_log_file:
        tee = Tee(sys.__stdout__, run_log_file, batch_log_handle)
        try:
            with redirect_output(tee):
                custom_config = build_custom_config(spec, comparison_root)
                raw_result_rows = main(pretrain_model_dir=str(pretrain_dir), custom_config=custom_config, seed=SEED)

            raw_run_dirs = [
                Path(result["save_dir"])
                for result in raw_result_rows
                if result.get("save_dir")
            ]
            move_and_rename_run_outputs(raw_run_dirs, target_dir)
            stage_dir_map = rename_run_files(target_dir)
            write_run_info(
                target_dir=target_dir,
                spec=spec,
                pretrain_dir=pretrain_dir,
                run_seconds=time.time() - start_time,
                log_path=run_log_path,
                stage_dir_map=stage_dir_map,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=tee)
            after_dirs = {path.name for path in comparison_root.iterdir() if path.is_dir()}
            after_dirs.discard("logs")
            orphan_dirs = sorted(after_dirs - before_dirs)
            if len(orphan_dirs) == 1:
                orphan_source = comparison_root / orphan_dirs[0]
                if not target_dir.exists() and orphan_source.exists():
                    shutil.move(str(orphan_source), str(target_dir))
                    try:
                        stage_dir_map = rename_run_files(target_dir)
                        write_run_info(
                            target_dir=target_dir,
                            spec=spec,
                            pretrain_dir=pretrain_dir,
                            run_seconds=time.time() - start_time,
                            log_path=run_log_path,
                            stage_dir_map=stage_dir_map,
                        )
                    except Exception:
                        traceback.print_exc(file=tee)

    run_seconds = time.time() - start_time
    if error_message:
        log_print(f"[失败] {spec['label']} | {error_message}", file_handle=batch_log_handle)
    else:
        log_print(f"[完成] {spec['label']} | 用时 {run_seconds:.2f}s", file_handle=batch_log_handle)

    return build_result_row(
        spec=spec,
        target_dir=target_dir if target_dir.exists() else None,
        log_path=run_log_path,
        stage_dir_map=stage_dir_map,
        error_message=error_message,
    )


class redirect_output:
    def __init__(self, stream):
        self.stream = stream
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self.stream
        sys.stderr = self.stream
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="批量复跑 PTL finetune 默认配置与全部 preset，并自动整理结果。")
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="可选，自定义输出目录；默认写入 results/ptl/finetune/comparisons 下的新时间戳目录。",
    )
    return parser.parse_args()


def main_cli():
    args = parse_args()
    comparison_root = Path(args.output_root) if args.output_root else None
    if comparison_root is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_root = COMPARISON_ROOT / f"{timestamp}_预设与默认全面对比_seed{SEED}"

    ensure_dir(comparison_root)
    logs_dir = ensure_dir(comparison_root / "logs")
    batch_log_path = comparison_root / "logs" / "批量运行总日志.log"

    pretrain_dir = find_latest_pretrain_run(PRETRAIN_RUNS_DIR)
    if pretrain_dir is None:
        raise FileNotFoundError("未找到可用的 pretrain run。请先生成包含 config.json 和 model.pth 的预训练结果。")
    pretrain_dir = Path(pretrain_dir)

    rows = []
    overall_start = time.time()
    with batch_log_path.open("w", encoding="utf-8") as batch_log_handle:
        log_print("批量复跑开始", file_handle=batch_log_handle)
        log_print(f"输出目录: {comparison_root}", file_handle=batch_log_handle)
        log_print(f"预训练目录: {pretrain_dir}", file_handle=batch_log_handle)
        log_print(f"总任务数: {len(RUN_SPECS)}", file_handle=batch_log_handle)

        for index, spec in enumerate(RUN_SPECS, start=1):
            log_print(
                f"进度: [{index}/{len(RUN_SPECS)}] {spec['label']}",
                file_handle=batch_log_handle,
            )
            rows.append(
                run_single_spec(
                    spec=spec,
                    comparison_root=comparison_root,
                    pretrain_dir=pretrain_dir,
                    logs_dir=logs_dir,
                    batch_log_handle=batch_log_handle,
                )
            )

        manifest = {
            "comparison_root": str(comparison_root),
            "pretrain_model_dir": str(pretrain_dir),
            "seed": SEED,
            "total_runs": len(RUN_SPECS),
            "total_seconds": time.time() - overall_start,
            "runs": rows,
        }
        write_json(comparison_root / "运行清单_manifest.json", manifest)
        write_comparison_csv(comparison_root / "综合对比_关键指标.csv", rows)
        write_comparison_markdown(
            comparison_root / "综合对比_关键指标.md",
            rows,
            comparison_root=comparison_root,
            pretrain_dir=pretrain_dir,
        )
        log_print(
            f"批量复跑完成，总耗时 {time.time() - overall_start:.2f}s",
            file_handle=batch_log_handle,
        )

    print(f"\n全部结果已整理到: {comparison_root}")


if __name__ == "__main__":
    main_cli()
