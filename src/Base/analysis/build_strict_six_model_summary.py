from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE_ROOT = REPO_ROOT / "results" / "base" / "fair_compare" / "full_six_model_strict_2023_2024"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "summary" / "full_six_model_strict_2023_2024"
FOCUS_FEATURES = ["CODMn", "DO", "pH"]
ALL_FEATURES = ["CODMn", "DO", "NH4N", "pH"]
MODEL_ORDER = ["MLP", "CNN", "LSTM", "Bi-LSTM", "CNN-LSTM", "Transformer", "PTL"]

BASELINE_MODELS = {
    "MLP": ("mlp", "mlp"),
    "CNN": ("cnn", "cnn"),
    "LSTM": ("lstm", "lstm"),
    "Bi-LSTM": ("bilstm", "bilstm"),
    "CNN-LSTM": ("cnn_lstm", "cnn_lstm"),
    "Transformer": ("basic_transformer", "basic_transformer"),
}

DEFAULT_STATIONS = [
    "盘溪大桥",
    "坝草",
    "老口",
    "鸦岗",
    "龟山塔",
    "珠海大桥",
    "新铺",
    "大墩",
    "五丰渡口",
    "上中",
    "白马",
    "阳朔",
    "交州",
    "桂花",
    "深圳河口",
    "象州运江老街",
]

CLASSIFICATION_ROWS = [
    {
        "station": "盘溪大桥",
        "verified_waterbody": "南盘江",
        "verified_basin_role": "珠江上游主干链/南盘江干流",
        "comparison_group": "珠江主干链/主要水道",
        "confidence": "高",
        "verification_note": "生态环境部国控断面设置材料列为珠江流域、南盘江、干流；作为滇中上游南盘江干流代表，不是昆明城区站。",
        "source_url": "https://www.mee.gov.cn/gkml/hbb/bwj/201603/W020160322380067193858.pdf",
    },
    {
        "station": "禄丰村",
        "verified_waterbody": "南盘江",
        "verified_basin_role": "珠江上游主干链/南盘江干流",
        "comparison_group": "珠江主干链/主要水道",
        "confidence": "中高",
        "verification_note": "作为盘溪大桥的同水系位置替代候选；公开资料将禄丰村国控断面纳入南盘江流域上下游联动治理重点，位置上对齐南盘江上游主干链。",
        "source_url": "https://kandian.sina.cn/article_5116298586_130f4855a02001fup4.html?from=news&subch=onews",
    },
    {
        "station": "花山水库出水口",
        "verified_waterbody": "南盘江",
        "verified_basin_role": "珠江上游主干链/南盘江干流",
        "comparison_group": "珠江主干链/主要水道",
        "confidence": "高",
        "verification_note": "作为盘溪大桥的同水系位置第二替代候选；国控监测网设置材料列为云南省曲靖市、南盘江、干流断面。",
        "source_url": "https://sthjt.hubei.gov.cn/fbjd/zc/zcwj/sthjbwj/201603/P020250604355532823698.pdf",
    },
    {
        "station": "坝草",
        "verified_waterbody": "北盘江",
        "verified_basin_role": "珠江上游主要支流北盘江干流",
        "comparison_group": "珠江主干链/主要水道",
        "confidence": "高",
        "verification_note": "贵州资料列为珠江流域、北盘江水系、北盘江、坝草；属于上游主要支流干流，不是贵阳城区站。",
        "source_url": "https://sthj.guizhou.gov.cn/zwgk/hjsj/cshszbg/201905/W020221026534836506537.pdf",
    },
    {
        "station": "老口",
        "verified_waterbody": "邕江",
        "verified_basin_role": "郁江-邕江支流水系干流",
        "comparison_group": "珠江支流水系干流",
        "confidence": "高",
        "verification_note": "广西水质资料列为南宁邕江断面；相对西江主干链属于郁江/邕江支流系统。",
        "source_url": "https://sthjt.gxzf.gov.cn/zfxxgk/zfxxgkgl/fdzdgknr/zdlyxx/szlxx/t3611721.shtml",
    },
    {
        "station": "鸦岗",
        "verified_waterbody": "珠江西航道",
        "verified_basin_role": "珠三角主要水道",
        "comparison_group": "珠江主干链/主要水道",
        "confidence": "高",
        "verification_note": "广州市政府材料明确鸦岗断面位于珠江西航道。",
        "source_url": "https://www.gz.gov.cn/zwfw/zxfw/gysy/content/post_7148030.html",
    },
    {
        "station": "龟山塔",
        "verified_waterbody": "榕江",
        "verified_basin_role": "韩江/粤东水系断面，非珠江干流链",
        "comparison_group": "非珠江主干链/区域支流",
        "confidence": "高",
        "verification_note": "广东省资料列出榕江干流龟山塔断面；本地 metadata 标作珠江流域，但论文中应按粤东榕江区域河流处理。",
        "source_url": "https://gdee.gd.gov.cn/attachment/0/396/396253/3030141.pdf",
    },
    {
        "station": "珠海大桥",
        "verified_waterbody": "磨刀门水道",
        "verified_basin_role": "西江入海主要水道",
        "comparison_group": "珠江主干链/主要水道",
        "confidence": "高",
        "verification_note": "中山市水环境月报列为磨刀门水道珠海大桥国考断面。",
        "source_url": "https://zsepb.zs.gov.cn/xxml/ztzl/hbzdlyxx/szhjxx/jhszyb/content/post_2500896.html",
    },
    {
        "station": "布洲",
        "verified_waterbody": "磨刀门水道",
        "verified_basin_role": "西江入海主要水道",
        "comparison_group": "珠江主干链/主要水道",
        "confidence": "高",
        "verification_note": "作为珠海大桥的同水道替代候选；中山市水环境月报将布洲和珠海大桥同时列为磨刀门水道国考断面。",
        "source_url": "https://zsepb.zs.gov.cn/xxml/ztzl/hbzdlyxx/szhjxx/jhszyb/content/post_2486215.html",
    },
    {
        "station": "新铺",
        "verified_waterbody": "石窟河",
        "verified_basin_role": "韩江上游支流石窟河断面，非珠江干流链",
        "comparison_group": "非珠江主干链/区域支流",
        "confidence": "高",
        "verification_note": "梅州市资料列为石窟河新铺/新铺（白渡沙坪）断面。",
        "source_url": "https://www.meizhou.gov.cn/zwgk/fggw/szfhj/content/mpost_2317624.html",
    },
    {
        "station": "大墩",
        "verified_waterbody": "东江北干流",
        "verified_basin_role": "东江支流水系主要水道",
        "comparison_group": "珠江支流水系干流",
        "confidence": "高",
        "verification_note": "增城区环境质量公报列为东江北干流水质断面。",
        "source_url": "https://www.zc.gov.cn/gk/zdly/hjbhxxgk/kqhjxx/content/post_9494980.html",
    },
    {
        "station": "五丰渡口",
        "verified_waterbody": "梅潭河",
        "verified_basin_role": "韩江支流梅潭河断面，非珠江干流链",
        "comparison_group": "非珠江主干链/区域支流",
        "confidence": "高",
        "verification_note": "梅州市/大埔县资料列为梅潭河五丰渡口断面。",
        "source_url": "https://www.meizhou.gov.cn/zwgk/fggw/szfhj/content/mpost_2317624.html",
    },
    {
        "station": "上中",
        "verified_waterbody": "左江",
        "verified_basin_role": "郁江支流左江断面",
        "comparison_group": "珠江支流水系干流",
        "confidence": "高",
        "verification_note": "广西水环境资料列出所在河流左江、断面上中、所在支流郁江支流。",
        "source_url": "https://sthjt.gxzf.gov.cn/zfxxgk/zfxxgkgl/fdzdgknr/zdlyxxgk/szlxx/t3613192.shtml",
    },
    {
        "station": "白马",
        "verified_waterbody": "乌江",
        "verified_basin_role": "长江一级支流乌江干流，非珠江流域",
        "comparison_group": "非珠江流域",
        "confidence": "高",
        "verification_note": "本地 metadata 为重庆市、长江流域；重庆武隆资料列乌江白马国考断面。注意不要与广西右江白马同名断面混淆。",
        "source_url": "https://cqwl.gov.cn/bmjz_sites/bm/sthjj/zwgk_98942/fdzdgknr_98944/zdxmhjyxpj/202307/P020230726612247015162.pdf",
    },
    {
        "station": "阳朔",
        "verified_waterbody": "漓江",
        "verified_basin_role": "桂江支流漓江干流断面",
        "comparison_group": "珠江支流水系干流",
        "confidence": "高",
        "verification_note": "广西年度水生态环境保护工作计划列为桂江支流、漓江、阳朔断面。",
        "source_url": "https://sthjt.gxzf.gov.cn/zfxxgk/zfxxgkgl/fdzdgknr/ghjg/hbgh/P020251103651237230738.pdf",
    },
    {
        "station": "交州",
        "verified_waterbody": "寻江",
        "verified_basin_role": "柳江支流寻江断面",
        "comparison_group": "珠江支流水系干流",
        "confidence": "高",
        "verification_note": "广西年度水生态环境保护工作计划列为柳江支流、寻江、交州断面。",
        "source_url": "https://sthjt.gxzf.gov.cn/zfxxgk/zfxxgkgl/fdzdgknr/ghjg/hbgh/P020251103651237230738.pdf",
    },
    {
        "station": "龙溪",
        "verified_waterbody": "洛清江",
        "verified_basin_role": "柳江支流洛清江断面",
        "comparison_group": "珠江支流水系干流",
        "confidence": "中",
        "verification_note": "作为交州的同区域同级水系替代候选；广西年度计划列为桂林市、柳江支流、洛清江、龙溪断面。它不是寻江同一水体，只能作为柳江支流位置近似替代。",
        "source_url": "https://sthjt.gxzf.gov.cn/zfxxgk/zfxxgkgl/fdzdgknr/ghjg/hbgh/P020250120565430347340.pdf",
    },
    {
        "station": "桂花",
        "verified_waterbody": "桂江",
        "verified_basin_role": "桂江支流/桂江断面",
        "comparison_group": "珠江支流水系干流",
        "confidence": "高",
        "verification_note": "广西年度水生态环境保护工作计划列为桂江支流、桂江、桂花断面。",
        "source_url": "https://sthjt.gxzf.gov.cn/zfxxgk/zfxxgkgl/fdzdgknr/ghjg/hbgh/P020250120565430347340.pdf",
    },
    {
        "station": "深圳河口",
        "verified_waterbody": "深圳河",
        "verified_basin_role": "入海河口/城市边界河流，非珠江主干链",
        "comparison_group": "非珠江主干链/区域支流",
        "confidence": "高",
        "verification_note": "深圳市生态环境局资料列为深圳河河口断面。",
        "source_url": "https://meeb.sz.gov.cn/ztfw/zdlyxxgk/shjyb/content/post_11090893.html",
    },
    {
        "station": "象州运江老街",
        "verified_waterbody": "柳江",
        "verified_basin_role": "柳江支流/柳江断面",
        "comparison_group": "珠江支流水系干流",
        "confidence": "高",
        "verification_note": "广西年度水生态环境保护工作计划列为柳江支流、柳江、象州运江老街断面。",
        "source_url": "https://sthjt.gxzf.gov.cn/zfxxgk/zfxxgkgl/fdzdgknr/ghjg/hbgh/P020251103651237230738.pdf",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build six-model strict comparison summary.")
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--ptl-batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--station", action="append", default=[])
    return parser.parse_args()


def latest_run(parent: Path, prefix: str, station_name: str) -> Path | None:
    candidates = sorted(
        parent.glob(f"{prefix}_{station_name}_seed*_*/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent if candidates else None


def read_metrics(metrics_path: Path) -> dict:
    return pd.read_csv(metrics_path, index_col=0).replace({np.nan: None}).to_dict(orient="index")


def summarize_metrics(metrics: dict) -> dict:
    focus = [metrics[feature] for feature in FOCUS_FEATURES if feature in metrics]
    all_features = [metrics[feature] for feature in ALL_FEATURES if feature in metrics]
    row = {
        "overall_nse": metrics.get("__overall__", {}).get("NSE"),
        "overall_rmse": metrics.get("__overall__", {}).get("RMSE"),
        "overall_mae": metrics.get("__overall__", {}).get("MAE"),
    }
    if focus:
        row["focus_mean_nse"] = float(np.mean([float(item["NSE"]) for item in focus]))
        row["focus_mean_rmse"] = float(np.mean([float(item["RMSE"]) for item in focus]))
        row["focus_mean_mae"] = float(np.mean([float(item["MAE"]) for item in focus]))
    if all_features:
        row["all_feature_mean_nse"] = float(np.mean([float(item["NSE"]) for item in all_features]))
        row["all_feature_mean_rmse"] = float(np.mean([float(item["RMSE"]) for item in all_features]))
        row["all_feature_mean_mae"] = float(np.mean([float(item["MAE"]) for item in all_features]))
    for feature in ALL_FEATURES:
        if feature in metrics:
            row[f"{feature}_nse"] = metrics[feature].get("NSE")
            row[f"{feature}_rmse"] = metrics[feature].get("RMSE")
            row[f"{feature}_mae"] = metrics[feature].get("MAE")
    return row


def read_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def soft_gap_summary(value) -> tuple[int | None, str]:
    if value is None:
        return 0, "off"
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, str(value)
    if parsed <= 0:
        return 0, "off"
    return parsed, f"soft<= {parsed}"


def collect_baselines(baseline_root: Path, stations: list[str]) -> list[dict]:
    rows = []
    for model_name, (subdir, prefix) in BASELINE_MODELS.items():
        parent = baseline_root / subdir
        for station_name in stations:
            run_dir = latest_run(parent, prefix, station_name)
            if run_dir is None:
                rows.append({"station": station_name, "model": model_name, "status": "missing"})
                continue
            metrics = read_metrics(run_dir / "metrics.csv")
            meta = read_meta(run_dir)
            soft_gap_steps, soft_gap_mode = soft_gap_summary(meta.get("soft_gap_max_steps"))
            row = {
                "station": station_name,
                "model": model_name,
                "status": "ok",
                "run_dir": str(run_dir),
                "metrics_path": str(run_dir / "metrics.csv"),
                "parameter_count": meta.get("parameter_count"),
                "train_windows": meta.get("train_windows"),
                "val_windows": meta.get("val_windows"),
                "test_windows": meta.get("test_windows"),
                "soft_gap_max_steps": soft_gap_steps,
                "soft_gap_mode": soft_gap_mode,
                "invalid_window_policy": meta.get("invalid_window_policy"),
                "time_start": meta.get("time_start"),
                "time_end": meta.get("time_end"),
                "uses_pretraining": False,
                "uses_progressive_transfer": False,
            }
            row.update(summarize_metrics(metrics))
            rows.append(row)
    return rows


def collect_ptl(ptl_batch_dir: Path, stations: list[str]) -> list[dict]:
    rows = []
    for station_name in stations:
        candidates = sorted(
            ptl_batch_dir.glob(f"progressive_{station_name}_seed*/stage3_daily/metrics.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            rows.append({"station": station_name, "model": "PTL", "status": "missing"})
            continue
        metrics_path = candidates[0]
        run_dir = metrics_path.parent
        meta = read_meta(run_dir)
        metrics = read_metrics(metrics_path)
        soft_gap_steps, soft_gap_mode = soft_gap_summary(meta.get("soft_gap_max_steps"))
        row = {
            "station": station_name,
            "model": "PTL",
            "status": "ok",
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
            "parameter_count": meta.get("parameter_count"),
            "train_windows": meta.get("train_windows"),
            "val_windows": meta.get("val_windows"),
            "test_windows": meta.get("test_windows"),
            "soft_gap_max_steps": soft_gap_steps,
            "soft_gap_mode": soft_gap_mode,
            "invalid_window_policy": meta.get("invalid_window_policy"),
            "time_start": meta.get("time_start"),
            "time_end": meta.get("time_end"),
            "uses_pretraining": True,
            "uses_progressive_transfer": True,
        }
        row.update(summarize_metrics(metrics))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stations = list(dict.fromkeys(args.station or DEFAULT_STATIONS))

    rows = collect_baselines(args.baseline_root, stations)
    rows.extend(collect_ptl(args.ptl_batch_dir, stations))
    summary = pd.DataFrame(rows)
    summary["model_order"] = summary["model"].map({model: idx for idx, model in enumerate(MODEL_ORDER)})
    summary["station_order"] = summary["station"].map({station: idx for idx, station in enumerate(stations)})
    summary = summary.sort_values(["station_order", "model_order"]).reset_index(drop=True)

    class_frame = pd.DataFrame(CLASSIFICATION_ROWS)
    summary = summary.merge(class_frame, on="station", how="left")

    summary.to_csv(args.output_dir / "six_model_strict_model_summary.csv", index=False, encoding="utf-8-sig")

    ok = summary[summary["status"] == "ok"].copy()
    ok.pivot_table(index="station", columns="model", values="focus_mean_nse", aggfunc="first").reindex(stations).reindex(columns=MODEL_ORDER).to_csv(
        args.output_dir / "six_model_strict_focus_nse_pivot.csv",
        encoding="utf-8-sig",
    )
    ok.pivot_table(index="station", columns="model", values="overall_nse", aggfunc="first").reindex(stations).reindex(columns=MODEL_ORDER).to_csv(
        args.output_dir / "six_model_strict_overall_nse_pivot.csv",
        encoding="utf-8-sig",
    )
    ok.groupby("model", as_index=False).agg(
        mean_overall_nse=("overall_nse", "mean"),
        mean_focus_nse=("focus_mean_nse", "mean"),
        mean_focus_rmse=("focus_mean_rmse", "mean"),
        station_count=("station", "nunique"),
    ).assign(model_order=lambda frame: frame["model"].map({model: idx for idx, model in enumerate(MODEL_ORDER)})).sort_values("model_order").drop(columns="model_order").to_csv(
        args.output_dir / "six_model_strict_model_average_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_frame.to_csv(args.output_dir / "verified_station_classification.csv", index=False, encoding="utf-8-sig")

    missing = summary[summary["status"] != "ok"].copy()
    missing.to_csv(args.output_dir / "six_model_strict_missing_runs.csv", index=False, encoding="utf-8-sig")

    print(args.output_dir)
    print(f"summary_rows={len(summary)} ok_rows={len(ok)} missing_rows={len(missing)}")


if __name__ == "__main__":
    main()
