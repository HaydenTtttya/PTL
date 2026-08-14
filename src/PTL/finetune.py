import datetime
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from progressive_core import (
    FEATURE_COLUMNS,
    ForecastWindowDataset,
    StandardScaler,
    WaterQualityTransformer,
    build_gap_aware_invalid_masks,
    compute_per_feature_metrics,
    compute_split_points,
    evaluate_model,
    fit_model,
    infer_device,
    list_station_names,
    load_matching_weights,
    load_station_frame,
    normalize_resolution,
    snapshot_state_dict,
    set_seed,
)
from model_agnostic_backbones import (
    build_model_agnostic_forecaster,
    normalize_backbone_name,
    normalize_model_agnostic_interface,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "results", "ptl")
PRETRAIN_RUNS_DIR = os.path.join(ARTIFACTS_ROOT, "pretrain", "runs")
FINETUNE_RUNS_DIR = os.path.join(ARTIFACTS_ROOT, "finetune", "runs")
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY_ENV_VALUES


STEPS_PER_DAY = {
    "weekly": 1 / 7,
    "week": 1 / 7,
    "4d": 1 / 4,
    "daily": 1,
    "12h": 2,
    "4h": 6,
}
WEATHER_COVERAGE_END = "2025-08-24 23:59:59"
AGGREGATED_WEATHER_FEATURE_COLUMNS = [
    "meteo_temp_c",
    "meteo_dewp_c",
    "meteo_rh_pct",
    "meteo_slp_hpa",
    "meteo_wind_ms",
]
WEATHER_REQUIRED_BASE_COLUMNS = [
    "meteo_temp_c",
    "meteo_dewp_c",
    "meteo_rh_pct",
    "meteo_slp_hpa",
    "meteo_wind_ms",
]
STATION_PRECIP_ROLLING_WINDOWS = (3, 7)
RAIN_ONSET_THRESHOLD_MM = 1.0
GSOD_MISSING_VALUES = {
    "TEMP": (9999.9,),
    "DEWP": (9999.9,),
    "SLP": (9999.9,),
    "STP": (9999.9,),
    "VISIB": (999.9,),
    "WDSP": (999.9,),
    "MXSPD": (999.9,),
    "GUST": (999.9,),
    "MAX": (9999.9,),
    "MIN": (9999.9,),
    "PRCP": (99.99,),
    "SNDP": (999.9,),
}
YANGSHUO_WEATHER_STATIONS = [
    {
        "name": "MENGSHAN",
        "filename": "59058099999_MENGSHAN_CHINA_(110.5166666,24.2).csv",
        "distance_km": 63.40,
    },
    {
        "name": "LIANGJIANG",
        "filename": "57957099999_LIANGJIANG_CHINA_(110.039197,25.218106).csv",
        "distance_km": 68.11,
    },
]
TP_FEATURE_COLUMNS = list(FEATURE_COLUMNS)
NH4N_FEATURE_COLUMNS = ["CODMn", "DO", "NH4N", "pH"]


def build_shared_stage_time_ranges(time_start, time_end):
    return {
        "stage1_weekly": (time_start, time_end),
        "stage2_4d": (time_start, time_end),
        "stage3_daily": (time_start, time_end),
    }


def build_station_precip_feature_columns(station_specs):
    columns = []
    for station_spec in station_specs:
        prefix = f"meteo_{station_spec['name'].lower()}"
        columns.extend(
            [
                f"{prefix}_prcp_mm",
                f"{prefix}_prcp_3d_mm",
                f"{prefix}_prcp_7d_mm",
                f"{prefix}_rain_onset",
            ]
        )
    return columns


def build_weather_feature_columns(station_specs=None):
    station_specs = station_specs or YANGSHUO_WEATHER_STATIONS
    return [
        *AGGREGATED_WEATHER_FEATURE_COLUMNS,
        *build_station_precip_feature_columns(station_specs),
    ]


def build_weather_required_columns(station_specs=None):
    station_specs = station_specs or YANGSHUO_WEATHER_STATIONS
    columns = list(WEATHER_REQUIRED_BASE_COLUMNS)
    for station_spec in station_specs:
        columns.append(f"meteo_{station_spec['name'].lower()}_prcp_mm")
    return columns


def _coerce_gsod_numeric(series, missing_values):
    numeric = pd.to_numeric(series, errors="coerce")
    for missing_value in missing_values:
        numeric = numeric.mask(np.isclose(numeric, missing_value))
    return numeric.astype(np.float32)


def _fahrenheit_to_celsius(values):
    return (values - 32.0) * (5.0 / 9.0)


def _knots_to_ms(values):
    return values * 0.514444


def _inches_to_mm(values):
    return values * 25.4


def _compute_relative_humidity(temp_c, dewp_c):
    exponent = (
        (17.625 * dewp_c) / (243.04 + dewp_c)
        - (17.625 * temp_c) / (243.04 + temp_c)
    )
    rh = 100.0 * np.exp(exponent)
    return np.clip(rh, 0.0, 100.0)


def _weighted_average_from_columns(frame, columns_and_weights):
    weighted_sum = np.zeros(len(frame), dtype=np.float32)
    weight_sum = np.zeros(len(frame), dtype=np.float32)
    for column_name, weight in columns_and_weights:
        values = frame[column_name].to_numpy(dtype=np.float32, copy=False)
        valid_mask = np.isfinite(values)
        if not np.any(valid_mask):
            continue
        weighted_sum[valid_mask] += values[valid_mask] * float(weight)
        weight_sum[valid_mask] += float(weight)

    averaged = np.full(len(frame), np.nan, dtype=np.float32)
    valid_weight_mask = weight_sum > 0
    averaged[valid_weight_mask] = weighted_sum[valid_weight_mask] / weight_sum[valid_weight_mask]
    return averaged

def load_weather_station_history(weather_data_root, station_spec, time_start=None, time_end=None):
    yearly_frames = []
    for year_name in sorted(entry.name for entry in os.scandir(weather_data_root) if entry.is_dir()):
        path = os.path.join(weather_data_root, year_name, station_spec["filename"])
        if os.path.exists(path):
            yearly_frames.append(pd.read_csv(path))
    if not yearly_frames:
        return None

    frame = pd.concat(yearly_frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["DATE"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    if time_start is not None:
        frame = frame[frame["timestamp"] >= pd.Timestamp(time_start)]
    if time_end is not None:
        frame = frame[frame["timestamp"] <= pd.Timestamp(time_end)]
    if frame.empty:
        return None

    full_index = pd.date_range(
        start=frame["timestamp"].iloc[0],
        end=frame["timestamp"].iloc[-1],
        freq="1D",
    )
    frame = frame.set_index("timestamp").reindex(full_index)
    frame.index.name = "timestamp"
    frame = frame.reset_index()

    temp_c = _fahrenheit_to_celsius(_coerce_gsod_numeric(frame["TEMP"], GSOD_MISSING_VALUES["TEMP"]))
    dewp_c = _fahrenheit_to_celsius(_coerce_gsod_numeric(frame["DEWP"], GSOD_MISSING_VALUES["DEWP"]))
    slp_hpa = _coerce_gsod_numeric(frame["SLP"], GSOD_MISSING_VALUES["SLP"])
    wind_ms = _knots_to_ms(_coerce_gsod_numeric(frame["WDSP"], GSOD_MISSING_VALUES["WDSP"]))
    prcp_mm = _inches_to_mm(_coerce_gsod_numeric(frame["PRCP"], GSOD_MISSING_VALUES["PRCP"]))
    rh_pct = _compute_relative_humidity(temp_c, dewp_c)
    rh_pct[~(np.isfinite(temp_c) & np.isfinite(dewp_c))] = np.nan

    prefix = station_spec["name"].lower()
    station_frame = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            f"{prefix}_temp_c": temp_c,
            f"{prefix}_dewp_c": dewp_c,
            f"{prefix}_rh_pct": rh_pct.astype(np.float32),
            f"{prefix}_slp_hpa": slp_hpa,
            f"{prefix}_wind_ms": wind_ms,
            f"{prefix}_prcp_mm": prcp_mm,
        }
    )
    station_frame[f"{prefix}_gap"] = station_frame.drop(columns=["timestamp"]).isna().any(axis=1)
    return station_frame

def load_stage_weather_frame(config, station_name, stage_spec, time_start=None, time_end=None):
    weather_config = stage_spec.get("meteorology") or {}
    if not weather_config.get("enabled"):
        return None, None
    if stage_spec["resolution"] != "daily":
        raise ValueError("当前 meteorology 外生特征仅支持 daily 阶段。")

    station_specs = weather_config.get("stations")
    if station_specs is None:
        station_specs = config.station_weather_map.get(station_name)
    if not station_specs:
        raise ValueError(f"站点 {station_name} 未配置气象站映射。")

    if time_start is None or time_end is None:
        raise ValueError("加载气象数据时必须提供 time_start 和 time_end。")

    merged = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=pd.Timestamp(time_start),
                end=pd.Timestamp(time_end),
                freq="1D",
            )
        }
    )
    loaded_station_specs = []
    weighted_source_columns = {
        "meteo_temp_c": [],
        "meteo_dewp_c": [],
        "meteo_rh_pct": [],
        "meteo_slp_hpa": [],
        "meteo_wind_ms": [],
    }

    for station_spec in station_specs:
        station_frame = load_weather_station_history(
            config.weather_data_root,
            station_spec,
            time_start=time_start,
            time_end=time_end,
        )
        if station_frame is None:
            continue
        merged = merged.merge(station_frame, on="timestamp", how="left")
        weight = float(station_spec.get("weight", 1.0 / max(station_spec["distance_km"], 1e-6)))
        prefix = station_spec["name"].lower()
        weighted_source_columns["meteo_temp_c"].append((f"{prefix}_temp_c", weight))
        weighted_source_columns["meteo_dewp_c"].append((f"{prefix}_dewp_c", weight))
        weighted_source_columns["meteo_rh_pct"].append((f"{prefix}_rh_pct", weight))
        weighted_source_columns["meteo_slp_hpa"].append((f"{prefix}_slp_hpa", weight))
        weighted_source_columns["meteo_wind_ms"].append((f"{prefix}_wind_ms", weight))
        loaded_station_specs.append(
            {
                "name": station_spec["name"],
                "filename": station_spec["filename"],
                "distance_km": float(station_spec["distance_km"]),
                "weight": weight,
            }
        )

    if not loaded_station_specs:
        return None, None

    weather_frame = pd.DataFrame({"timestamp": merged["timestamp"]})
    for feature_name, columns_and_weights in weighted_source_columns.items():
        weather_frame[feature_name] = _weighted_average_from_columns(merged, columns_and_weights)

    for station_spec in loaded_station_specs:
        station_prefix = station_spec["name"].lower()
        feature_prefix = f"meteo_{station_prefix}"
        station_prcp = pd.to_numeric(
            merged[f"{station_prefix}_prcp_mm"],
            errors="coerce",
        ).astype(np.float32)
        station_prcp_series = pd.Series(station_prcp)
        station_prcp_filled = station_prcp_series.fillna(0.0)
        weather_frame[f"{feature_prefix}_prcp_mm"] = station_prcp_series.astype(np.float32)
        for window_size in STATION_PRECIP_ROLLING_WINDOWS:
            weather_frame[f"{feature_prefix}_prcp_{window_size}d_mm"] = (
                station_prcp_filled
                .rolling(window=window_size, min_periods=1)
                .sum()
                .astype(np.float32)
            )
        previous_day_prcp = station_prcp_filled.shift(1, fill_value=0.0)
        onset = (
            (station_prcp_filled >= RAIN_ONSET_THRESHOLD_MM)
            & (previous_day_prcp < RAIN_ONSET_THRESHOLD_MM)
        ).astype(np.float32)
        onset[station_prcp_series.isna()] = np.nan
        weather_frame[f"{feature_prefix}_rain_onset"] = onset.astype(np.float32)

    weather_feature_columns = build_weather_feature_columns(loaded_station_specs)
    weather_required_columns = build_weather_required_columns(loaded_station_specs)
    weather_frame["__weather_gap__"] = weather_frame[weather_required_columns].isna().any(axis=1)
    weather_frame[weather_feature_columns] = weather_frame[weather_feature_columns].astype(np.float32)

    coverage_mask = ~weather_frame["__weather_gap__"]
    coverage = {
        "records": int(len(weather_frame)),
        "valid_records": int(coverage_mask.sum()),
        "invalid_records": int((~coverage_mask).sum()),
        "coverage_start": str(weather_frame["timestamp"].min()),
        "coverage_end": str(weather_frame["timestamp"].max()),
    }
    metadata = {
        "stations": loaded_station_specs,
        "feature_columns": weather_feature_columns,
        "coverage": coverage,
        "rain_onset_threshold_mm": RAIN_ONSET_THRESHOLD_MM,
    }
    return weather_frame, metadata

def build_weekly_to_daily_stages(stage1_overrides=None, stage2_overrides=None, stage3_overrides=None):
    stages = [
        {
            "name": "stage1_weekly",
            "resolution": "weekly",
            "window_days": 28,
            "pred_steps": 1,
            "epochs": 50,
            "base_lr": 3e-4,
            "freeze_backbone_epochs": 6,
            "resize_mode": "nearest",
            "loss_name": "mse_nse",
            "nse_weight": 0.2,
            "monitor_metric": "nse",
        },
        {
            "name": "stage2_4d",
            "resolution": "4d",
            "window_days": 28,
            "pred_steps": 1,
            "epochs": 50,
            "base_lr": 2e-4,
            "freeze_backbone_epochs": 2,
        },
        {
            "name": "stage3_daily",
            "resolution": "daily",
            "window_days": 28,
            "pred_steps": 1,
            "epochs": 50,
            "base_lr": 8e-5,
            "weight_decay": 2e-4,
            "freeze_backbone_epochs": 2,
            "invalid_window_policy": "all",
            "soft_gap_max_steps": 6,
        },
    ]

    for stage, overrides in zip(stages, (stage1_overrides, stage2_overrides, stage3_overrides)):
        if overrides:
            stage.update(overrides)

    return stages

def build_daily_target_config(soft_gap_max_steps=6):
    stage3_overrides = {
        "invalid_window_policy": "all",
        "soft_gap_max_steps": soft_gap_max_steps,
    }
    return {
        "progressive_stages": build_weekly_to_daily_stages(stage3_overrides=stage3_overrides),
    }

def build_stage_nh4n_two_stage_config(stage_spec, scaler, feature_columns):
    config = dict(stage_spec.get("nh4n_two_stage") or {})
    if not config.get("enabled"):
        return None

    feature_columns = list(feature_columns)
    feature_name = config.get("feature", "TP")
    if feature_name not in feature_columns:
        raise ValueError(f"未知 nh4n_two_stage feature: {feature_name}")

    feature_index = feature_columns.index(feature_name)
    feature_mean = float(scaler.mean[feature_index])
    feature_std = float(scaler.std[feature_index] + 1e-8)
    floor_raw = float(config.get("floor", 0.0))
    spike_threshold_raw = float(config.get("spike_threshold", floor_raw))
    return {
        **config,
        "enabled": True,
        "feature": feature_name,
        "feature_index": feature_index,
        "floor_raw": floor_raw,
        "spike_threshold_raw": spike_threshold_raw,
        "floor_scaled": (floor_raw - feature_mean) / feature_std,
        "spike_threshold_scaled": (spike_threshold_raw - feature_mean) / feature_std,
    }

def build_stage_train_sampler(train_dataset, raw_train_values, stage_spec, feature_columns):
    sampler_feature = stage_spec.get("sampler_feature")
    if not sampler_feature:
        return None, None
    feature_columns = list(feature_columns)
    if sampler_feature not in feature_columns:
        raise ValueError(f"未知 sampler_feature: {sampler_feature}")

    sampler_threshold = stage_spec.get("sampler_threshold")
    if sampler_threshold is None:
        raise ValueError("启用 train sampler 时必须提供 sampler_threshold。")

    sampler_mode = stage_spec.get("sampler_positive_mode", "target_above_threshold")
    positive_weight = float(stage_spec.get("sampler_positive_weight", 1.0))
    feature_index = feature_columns.index(sampler_feature)
    weights = np.ones(len(train_dataset), dtype=np.float64)
    positive_window_count = 0

    for sample_index, start in enumerate(train_dataset.window_starts):
        mid = int(start) + train_dataset.raw_seq_len
        end = mid + train_dataset.raw_pred_len
        target_slice = raw_train_values[mid:end, feature_index]
        if target_slice.size == 0:
            continue

        is_positive = np.any(target_slice >= sampler_threshold)
        if sampler_mode == "target_spike_onset":
            previous_value = raw_train_values[mid - 1, feature_index] if mid > 0 else float("-inf")
            is_positive = is_positive and previous_value < sampler_threshold
        elif sampler_mode != "target_above_threshold":
            raise ValueError(f"不支持的 sampler_positive_mode: {sampler_mode}")

        if is_positive:
            weights[sample_index] = positive_weight
            positive_window_count += 1

    sampler_stats = {
        "feature": sampler_feature,
        "threshold": float(sampler_threshold),
        "positive_mode": sampler_mode,
        "positive_weight": positive_weight,
        "positive_window_count": positive_window_count,
        "positive_window_ratio": (
            positive_window_count / len(train_dataset)
            if len(train_dataset) > 0
            else 0.0
        ),
    }

    if positive_window_count == 0 or positive_weight <= 1.0:
        return None, sampler_stats

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
    )
    return sampler, sampler_stats

def build_previous_feature_values(frame, timestamps, feature_name):
    lookup = frame[["timestamp", feature_name]].copy()
    lookup["timestamp"] = pd.to_datetime(lookup["timestamp"])
    lookup["target_timestamp"] = lookup["timestamp"] + pd.Timedelta(days=1)
    flat_timestamps = pd.to_datetime(timestamps.reshape(-1))
    prev_values = (
        lookup.set_index("target_timestamp")[feature_name]
        .reindex(flat_timestamps)
        .to_numpy(dtype=np.float32)
    )
    return prev_values

def apply_floor_guard_adjustment(
    predictions,
    previous_values,
    floor_value,
    global_alpha,
    prev_spike_threshold,
    prev_spike_alpha,
):
    adjusted = floor_value + (global_alpha * (predictions - floor_value))
    if prev_spike_threshold is not None:
        spike_mask = np.isfinite(previous_values) & (previous_values >= prev_spike_threshold)
        adjusted[spike_mask] = floor_value + (
            prev_spike_alpha * (adjusted[spike_mask] - floor_value)
        )
    return adjusted


def fit_stage_postprocess(stage_spec, frame, preds, targets, timestamps, feature_columns):
    mode = stage_spec.get("postprocess_mode")
    if mode is None:
        return None
    if mode != "floor_guard":
        raise ValueError(f"不支持的 postprocess_mode: {mode}")

    feature_columns = list(feature_columns)
    feature_name = stage_spec.get("postprocess_feature")
    if feature_name not in feature_columns:
        raise ValueError(f"未知 postprocess_feature: {feature_name}")

    feature_index = feature_columns.index(feature_name)
    floor_value = float(stage_spec.get("postprocess_floor", 0.0))
    background_threshold = float(
        stage_spec.get("postprocess_background_threshold", floor_value)
    )
    prev_spike_threshold = stage_spec.get("postprocess_prev_spike_threshold")
    if prev_spike_threshold is not None:
        prev_spike_threshold = float(prev_spike_threshold)
    selection_metric = stage_spec.get("postprocess_selection", "background_rmse")
    global_alpha_grid = stage_spec.get("postprocess_global_alpha_grid")
    prev_alpha_grid = stage_spec.get("postprocess_prev_alpha_grid")
    if global_alpha_grid is None:
        global_alpha_grid = [step / 10 for step in range(11)]
    if prev_alpha_grid is None:
        prev_alpha_grid = [step / 10 for step in range(11)]

    flat_targets = targets.reshape(-1, targets.shape[-1])[:, feature_index]
    background_mask = flat_targets <= background_threshold
    previous_values = build_previous_feature_values(frame, timestamps, feature_name)
    flat_preds = preds.reshape(-1, preds.shape[-1])[:, feature_index]
    best = None

    for global_alpha in global_alpha_grid:
        for prev_alpha in prev_alpha_grid:
            adjusted = apply_floor_guard_adjustment(
                flat_preds,
                previous_values,
                floor_value=floor_value,
                global_alpha=float(global_alpha),
                prev_spike_threshold=prev_spike_threshold,
                prev_spike_alpha=float(prev_alpha),
            )
            if selection_metric == "background_rmse":
                eval_mask = background_mask
            elif selection_metric == "rmse":
                eval_mask = np.ones_like(background_mask, dtype=bool)
            else:
                raise ValueError(f"不支持的 postprocess_selection: {selection_metric}")

            if not np.any(eval_mask):
                continue

            rmse = float(
                np.sqrt(np.mean((adjusted[eval_mask] - flat_targets[eval_mask]) ** 2))
            )
            if best is None or rmse < best["selection_rmse"]:
                best = {
                    "mode": mode,
                    "feature": feature_name,
                    "floor": floor_value,
                    "background_threshold": background_threshold,
                    "prev_spike_threshold": prev_spike_threshold,
                    "global_alpha": float(global_alpha),
                    "prev_spike_alpha": float(prev_alpha),
                    "selection_metric": selection_metric,
                    "selection_rmse": rmse,
                }

    return best


def apply_stage_postprocess(postprocess_params, frame, preds, timestamps, feature_columns):
    if postprocess_params is None:
        return preds
    if postprocess_params["mode"] != "floor_guard":
        raise ValueError(f"不支持的 postprocess mode: {postprocess_params['mode']}")

    feature_columns = list(feature_columns)
    feature_name = postprocess_params["feature"]
    feature_index = feature_columns.index(feature_name)
    previous_values = build_previous_feature_values(frame, timestamps, feature_name)
    adjusted_preds = np.array(preds, copy=True)
    flat_preds = adjusted_preds.reshape(-1, adjusted_preds.shape[-1])
    flat_preds[:, feature_index] = apply_floor_guard_adjustment(
        flat_preds[:, feature_index],
        previous_values,
        floor_value=postprocess_params["floor"],
        global_alpha=postprocess_params["global_alpha"],
        prev_spike_threshold=postprocess_params["prev_spike_threshold"],
        prev_spike_alpha=postprocess_params["prev_spike_alpha"],
    )
    return adjusted_preds


class FinetuneConfig:
    def __init__(self):
        self.data_root = os.path.join(REPO_ROOT, "data", "water_quality_processed_tp_2023_2025")
        self.weather_data_root = os.path.join(REPO_ROOT, "data", "meteorology_2022-2025")
        self.save_dir = FINETUNE_RUNS_DIR
        self.save_model_weights = env_flag("PTL_FINETUNE_SAVE_MODEL_WEIGHTS")
        self.pretrain_model_dir = None
        self.optimization_profile = "default"
        self.model_agnostic_interface = "legacy"

        self.target_station_names = ["阳朔"]
        self.max_target_stations = None
        self.time_start = "2023-01-01 00:00:00"
        self.time_end = "2025-12-31 23:59:59"
        self.stage_time_ranges = {
            "stage3_daily": ("2023-01-01 00:00:00", "2025-12-31 23:59:59"),
        }
        self.filter_invalid_windows = True
        self.invalid_window_policy = "target_only"
        self.resize_mode = "linear"
        self.feature_columns = list(TP_FEATURE_COLUMNS)

        self.model_seq_len = 168
        self.model_pred_len = 1
        self.input_dim = len(self.feature_columns)
        self.backbone_name = "transformer"
        self.hidden_size = 256
        self.num_heads = 8
        self.e_layer = 3
        self.use_temporal_adapter = True
        self.temporal_adapter_kernel_size = 5

        self.cnn_channels = (64, 128, 128)
        self.cnn_kernel_sizes = (3, 3, 3)
        self.cnn_dilations = (1, 2, 4)
        self.cnn_use_batch_norm = True
        self.lstm_hidden_dim = 128
        self.lstm_num_layers = 2
        self.lstm_use_input_layer_norm = True
        self.backbone_dropout = 0.15
        self.backbone_activation = "gelu"
        self.backbone_head_hidden_dim = 128
        self.mlp_hidden_dims = (256, 128)
        self.mlp_dropout = 0.1
        self.mlp_use_layer_norm = True
        self.cnn_lstm_conv_channels = (64, 128)
        self.cnn_lstm_kernel_sizes = (3, 3)
        self.cnn_lstm_hidden_dim = 128
        self.cnn_lstm_layers = 1
        self.cnn_lstm_use_batch_norm = True
        self.cnn_lstm_use_input_layer_norm = True

        self.batch_size = 32
        self.epochs = 60
        self.base_lr = 3e-4
        self.epsilon = 1e-8
        self.weight_decay = 1e-4
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.lr_milestones = [20, 40]
        self.lr_decay_ratio = 0.5
        self.max_grad_norm = 1.0
        self.loss_name = "mse"
        self.nse_weight = 0.0
        self.monitor_metric = "loss"
        self.early_stopping_patience = 10
        self.early_stopping_min_delta = 1e-3
        self.scheduler_name = "plateau"
        self.scheduler_patience = 4
        self.scheduler_min_lr = 1e-5
        self.freeze_backbone_epochs = 0

        self.progressive_stages = build_weekly_to_daily_stages()
        self.station_weather_map = {
            "阳朔": [dict(station_spec) for station_spec in YANGSHUO_WEATHER_STATIONS],
        }

        self.device = infer_device()
        os.makedirs(self.save_dir, exist_ok=True)


def build_stage_specs(config):
    stage_specs = []
    for stage in config.progressive_stages:
        resolution = normalize_resolution(stage["resolution"])
        if resolution not in STEPS_PER_DAY:
            raise ValueError(f"不支持的分辨率: {resolution}")

        raw_seq_len_float = stage["window_days"] * STEPS_PER_DAY[resolution]
        raw_seq_len = int(round(raw_seq_len_float))
        if raw_seq_len <= 0 or not np.isclose(raw_seq_len_float, raw_seq_len):
            raise ValueError(
                f"window_days={stage['window_days']} 与 resolution={resolution} 无法得到合法步数。"
            )
        model_seq_len = stage.get("model_seq_len", config.model_seq_len)
        model_pred_len = stage.get("model_pred_len", config.model_pred_len)
        stage_specs.append(
            {
                **stage,
                "resolution": resolution,
                "raw_seq_len": raw_seq_len,
                "model_seq_len": model_seq_len,
                "model_pred_len": model_pred_len,
            }
        )
    return stage_specs


def load_stage_data(config, station_name, stage_spec):
    feature_columns = list(config.feature_columns)
    stage_time_start, stage_time_end = config.stage_time_ranges.get(
        stage_spec["name"],
        (config.time_start, config.time_end),
    )
    filter_invalid_windows = stage_spec.get("filter_invalid_windows", config.filter_invalid_windows)
    invalid_window_policy = stage_spec.get("invalid_window_policy", config.invalid_window_policy)
    frame = load_station_frame(
        config.data_root,
        station_name,
        stage_spec["resolution"],
        time_start=stage_time_start,
        time_end=stage_time_end,
        feature_columns=feature_columns,
    )
    if frame is None:
        return None

    raw_values = frame[feature_columns].to_numpy(dtype="float32", copy=True)
    timestamps = frame["timestamp"].to_numpy(dtype="datetime64[ns]")
    invalid_mask = frame["__gap__"].to_numpy(dtype=bool, copy=True) if "__gap__" in frame.columns else np.zeros(len(frame), dtype=bool)
    input_invalid_mask, target_invalid_mask, gap_stats = build_gap_aware_invalid_masks(
        invalid_mask,
        soft_gap_max_steps=stage_spec.get("soft_gap_max_steps"),
    )
    weather_frame, weather_metadata = load_stage_weather_frame(
        config,
        station_name,
        stage_spec,
        time_start=stage_time_start,
        time_end=stage_time_end,
    )
    weather_invalid_mask = np.zeros(len(frame), dtype=bool)
    input_feature_columns = list(feature_columns)
    raw_input_values = raw_values
    weather_feature_columns = list((weather_metadata or {}).get("feature_columns", []))
    if weather_frame is not None:
        merged_frame = frame[["timestamp", *feature_columns]].merge(
            weather_frame,
            on="timestamp",
            how="left",
        )
        weather_invalid_mask = merged_frame["__weather_gap__"].fillna(True).to_numpy(dtype=bool)
        raw_input_values = merged_frame[
            [*feature_columns, *weather_feature_columns]
        ].to_numpy(dtype="float32", copy=True)
        input_feature_columns = [*feature_columns, *weather_feature_columns]
        weather_input_invalid_mask, _, weather_gap_stats = build_gap_aware_invalid_masks(
            weather_invalid_mask,
            soft_gap_max_steps=stage_spec.get(
                "weather_soft_gap_max_steps",
                stage_spec.get("soft_gap_max_steps"),
            ),
        )
        input_invalid_mask = input_invalid_mask | weather_input_invalid_mask
    else:
        weather_gap_stats = {
            "soft_gap_max_steps": stage_spec.get(
                "weather_soft_gap_max_steps",
                stage_spec.get("soft_gap_max_steps"),
            ),
            "gap_segment_count": 0,
            "soft_gap_segment_count": 0,
            "hard_gap_segment_count": 0,
            "gap_steps": 0,
            "soft_gap_steps": 0,
            "hard_gap_steps": 0,
        }
    raw_seq_len = stage_spec["raw_seq_len"]
    raw_pred_len = stage_spec["pred_steps"]

    if len(raw_values) < raw_seq_len + raw_pred_len + 8:
        return None

    train_end, val_end = compute_split_points(len(raw_values), config.train_ratio, config.val_ratio)
    scaler = StandardScaler()
    scaler.fit(raw_values[:train_end])
    target_values = scaler.transform(raw_values).astype("float32")
    input_fit_values = np.array(raw_input_values, copy=True)
    if np.isnan(input_fit_values).any():
        train_feature_means = np.nanmean(input_fit_values[:train_end], axis=0)
        train_feature_means = np.where(np.isfinite(train_feature_means), train_feature_means, 0.0)
        nan_mask = np.isnan(input_fit_values)
        input_fit_values[nan_mask] = np.take(train_feature_means, np.where(nan_mask)[1])
    input_scaler = StandardScaler()
    input_scaler.fit(input_fit_values[:train_end])
    input_values = input_scaler.transform(input_fit_values).astype("float32")

    train_dataset = ForecastWindowDataset(
        input_values=input_values,
        target_values=target_values,
        timestamps=timestamps,
        raw_seq_len=raw_seq_len,
        raw_pred_len=raw_pred_len,
        model_seq_len=stage_spec["model_seq_len"],
        model_pred_len=stage_spec["model_pred_len"],
        invalid_mask=invalid_mask,
        split="train",
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        filter_invalid_windows=filter_invalid_windows,
        invalid_window_policy=invalid_window_policy,
        resize_mode=stage_spec.get("resize_mode", config.resize_mode),
        input_invalid_mask=input_invalid_mask,
        target_invalid_mask=target_invalid_mask,
    )
    val_dataset = ForecastWindowDataset(
        input_values=input_values,
        target_values=target_values,
        timestamps=timestamps,
        raw_seq_len=raw_seq_len,
        raw_pred_len=raw_pred_len,
        model_seq_len=stage_spec["model_seq_len"],
        model_pred_len=stage_spec["model_pred_len"],
        invalid_mask=invalid_mask,
        split="val",
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        filter_invalid_windows=filter_invalid_windows,
        invalid_window_policy=invalid_window_policy,
        resize_mode=stage_spec.get("resize_mode", config.resize_mode),
        input_invalid_mask=input_invalid_mask,
        target_invalid_mask=target_invalid_mask,
    )
    test_dataset = ForecastWindowDataset(
        input_values=input_values,
        target_values=target_values,
        timestamps=timestamps,
        raw_seq_len=raw_seq_len,
        raw_pred_len=raw_pred_len,
        model_seq_len=stage_spec["model_seq_len"],
        model_pred_len=stage_spec["model_pred_len"],
        invalid_mask=invalid_mask,
        split="test",
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        filter_invalid_windows=filter_invalid_windows,
        invalid_window_policy=invalid_window_policy,
        resize_mode=stage_spec.get("resize_mode", config.resize_mode),
        input_invalid_mask=input_invalid_mask,
        target_invalid_mask=target_invalid_mask,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        return None

    train_sampler, train_sampler_stats = build_stage_train_sampler(
        train_dataset,
        raw_values[:train_end],
        stage_spec,
        feature_columns,
    )
    if train_sampler is None:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)

    return {
        "frame": frame,
        "scaler": scaler,
        "input_scaler": input_scaler,
        "train_loader": train_loader,
        "val_loader": DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
        "test_loader": DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False),
        "filter_invalid_windows": filter_invalid_windows,
        "invalid_window_policy": invalid_window_policy,
        "soft_gap_max_steps": gap_stats["soft_gap_max_steps"],
        "input_feature_columns": input_feature_columns,
        "input_dim": len(input_feature_columns),
        "weather_feature_columns": weather_feature_columns,
        "weather_metadata": weather_metadata,
        "records": len(raw_values),
        "invalid_records": int(invalid_mask.sum()),
        "valid_records": int((~invalid_mask).sum()),
        "weather_invalid_records": int(weather_invalid_mask.sum()),
        "input_invalid_records": int(input_invalid_mask.sum()),
        "target_invalid_records": int(target_invalid_mask.sum()),
        "gap_segment_count": gap_stats["gap_segment_count"],
        "soft_gap_segment_count": gap_stats["soft_gap_segment_count"],
        "hard_gap_segment_count": gap_stats["hard_gap_segment_count"],
        "soft_gap_records": gap_stats["soft_gap_steps"],
        "hard_gap_records": gap_stats["hard_gap_steps"],
        "weather_gap_segment_count": weather_gap_stats["gap_segment_count"],
        "weather_soft_gap_segment_count": weather_gap_stats["soft_gap_segment_count"],
        "weather_hard_gap_segment_count": weather_gap_stats["hard_gap_segment_count"],
        "weather_soft_gap_records": weather_gap_stats["soft_gap_steps"],
        "weather_hard_gap_records": weather_gap_stats["hard_gap_steps"],
        "time_start": stage_time_start,
        "time_end": stage_time_end,
        "train_windows": len(train_dataset),
        "train_candidate_windows": train_dataset.candidate_window_count,
        "train_filtered_windows": train_dataset.filtered_window_count,
        "val_windows": len(val_dataset),
        "val_candidate_windows": val_dataset.candidate_window_count,
        "val_filtered_windows": val_dataset.filtered_window_count,
        "test_windows": len(test_dataset),
        "test_candidate_windows": test_dataset.candidate_window_count,
        "test_filtered_windows": test_dataset.filtered_window_count,
        "train_sampler_stats": train_sampler_stats,
    }


def build_model(config, stage_spec=None, nh4n_two_stage_config=None):
    seq_len = stage_spec["model_seq_len"] if stage_spec is not None else config.model_seq_len
    pred_len = stage_spec["model_pred_len"] if stage_spec is not None else config.model_pred_len
    input_dim = stage_spec.get("input_dim", config.input_dim) if stage_spec is not None else config.input_dim
    backbone_name = normalize_backbone_name(getattr(config, "backbone_name", "transformer"))
    if backbone_name in {"mlp", "cnn", "lstm", "bilstm", "cnn_lstm"}:
        if nh4n_two_stage_config and nh4n_two_stage_config.get("enabled"):
            raise ValueError(
                f"The {backbone_name} model-agnostic experiment does not support "
                "the NH4N auxiliary two-stage head."
            )
        return build_model_agnostic_forecaster(
            backbone_name=backbone_name,
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            target_dim=len(config.feature_columns),
            config=config,
            interface=config.model_agnostic_interface,
            reconstruction_len=seq_len,
        )
    if backbone_name != "transformer":
        raise ValueError(f"Unsupported forecasting backbone: {backbone_name}")
    return WaterQualityTransformer(
        num_heads=config.num_heads,
        e_layer=config.e_layer,
        hidden_size=config.hidden_size,
        input_dim=input_dim,
        target_dim=len(config.feature_columns),
        target_feature_names=config.feature_columns,
        seq_len=seq_len,
        pred_len=pred_len,
        use_temporal_adapter=config.use_temporal_adapter,
        temporal_adapter_kernel_size=config.temporal_adapter_kernel_size,
        nh4n_two_stage_config=nh4n_two_stage_config,
    )


def save_stage_outputs(stage_dir, history, preds, targets, timestamps, metrics, meta, feature_columns):
    os.makedirs(stage_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(stage_dir, "history.csv"), index=False)
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(os.path.join(stage_dir, "metrics.csv"))

    flat_times = timestamps.reshape(-1)
    prediction_frame = pd.DataFrame({"timestamp": pd.to_datetime(flat_times)})
    preds_flat = preds.reshape(-1, preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])

    for index, name in enumerate(feature_columns):
        prediction_frame[f"True_{name}"] = targets_flat[:, index]
        prediction_frame[f"Pred_{name}"] = preds_flat[:, index]

    prediction_frame.to_csv(os.path.join(stage_dir, "predictions.csv"), index=False)

    with open(os.path.join(stage_dir, "meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


def find_latest_pretrain_run(pretrain_runs_dir):
    if not os.path.isdir(pretrain_runs_dir):
        return None

    candidates = []
    for entry in os.scandir(pretrain_runs_dir):
        if not entry.is_dir():
            continue

        config_path = os.path.join(entry.path, "config.json")
        model_path = os.path.join(entry.path, "model.pth")
        if os.path.exists(config_path) and os.path.exists(model_path):
            candidates.append(entry.path)

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


def build_finetune_preset(preset_name):
    preset_name = (preset_name or "").strip()
    if not preset_name:
        return None

    if preset_name == "target75_v1":
        return {
            "invalid_window_policy": "target_only",
            "progressive_stages": build_weekly_to_daily_stages(
                stage2_overrides={
                    "epochs": 60,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                },
                stage3_overrides={
                    "window_days": 28,
                    "epochs": 70,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 2,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": None,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.2,
                    "monitor_metric": "nse",
                    "early_stopping_patience": 15,
                    "scheduler_patience": 6,
                },
            ),
        }

    if preset_name in {"target75_v2_soft3", "gap_policy_v2_soft3"}:
        return {
            "invalid_window_policy": "target_only",
            "progressive_stages": build_weekly_to_daily_stages(
                stage2_overrides={
                    "epochs": 60,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                },
                stage3_overrides={
                    "window_days": 28,
                    "epochs": 70,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 2,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 3,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.2,
                    "monitor_metric": "nse",
                    "early_stopping_patience": 15,
                    "scheduler_patience": 6,
                },
            ),
        }

    if preset_name in {"target75_v2_soft6", "gap_policy_v2_soft6"}:
        return {
            "invalid_window_policy": "target_only",
            "progressive_stages": build_weekly_to_daily_stages(
                stage2_overrides={
                    "epochs": 60,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                },
                stage3_overrides={
                    "window_days": 28,
                    "epochs": 70,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 2,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.2,
                    "monitor_metric": "nse",
                    "early_stopping_patience": 15,
                    "scheduler_patience": 6,
                },
            ),
        }

    if preset_name in {"tp_overall_v1", "daily_tp_overall_v1"}:
        return {
            "invalid_window_policy": "target_only",
            "progressive_stages": build_weekly_to_daily_stages(
                stage3_overrides={
                    "window_days": 28,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.05,
                    "monitor_metric": "nse",
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                },
            ),
        }

    if preset_name in {"tp_overall_v2", "daily_tp_overall_v2"}:
        return {
            "invalid_window_policy": "target_only",
            "progressive_stages": build_weekly_to_daily_stages(
                stage3_overrides={
                    "window_days": 42,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.05,
                    "monitor_metric": "nse",
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                },
            ),
        }

    if preset_name in {"nh4n_daily_v1", "daily_nh4n_v1"}:
        return {
            "invalid_window_policy": "target_only",
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "progressive_stages": build_weekly_to_daily_stages(
                stage3_overrides={
                    "window_days": 28,
                    "epochs": 60,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 2,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.2,
                    "monitor_metric": "nse",
                    "monitor_feature": "NH4N",
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "loss_feature_weights": {
                        "NH4N": 4.0,
                    },
                    "sampler_feature": "NH4N",
                    "sampler_threshold": 0.05,
                    "sampler_positive_weight": 10.0,
                    "sampler_positive_mode": "target_spike_onset",
                },
            ),
        }

    if preset_name in {"nh4n_floor_guard_v1", "daily_nh4n_floor_v1"}:
        return {
            "invalid_window_policy": "target_only",
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "progressive_stages": build_weekly_to_daily_stages(
                stage3_overrides={
                    "window_days": 28,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "postprocess_mode": "floor_guard",
                    "postprocess_feature": "NH4N",
                    "postprocess_floor": 0.025,
                    "postprocess_background_threshold": 0.03,
                    "postprocess_prev_spike_threshold": 0.05,
                    "postprocess_selection": "background_rmse",
                },
            ),
        }

    if preset_name in {"core3_progressive_v1", "daily_core3_progressive_v1"}:
        core3_weights = {
            "CODMn": 1.5,
            "DO": 1.5,
            "NH4N": 0.25,
            "pH": 1.25,
        }
        return {
            "invalid_window_policy": "target_only",
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "use_temporal_adapter": False,
            "progressive_stages": build_weekly_to_daily_stages(
                stage1_overrides={
                    "window_days": 56,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.05,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(core3_weights),
                    "loss_feature_weights": dict(core3_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage2_overrides={
                    "window_days": 32,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(core3_weights),
                    "loss_feature_weights": dict(core3_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage3_overrides={
                    "window_days": 8,
                    "model_seq_len": 8,
                    "epochs": 70,
                    "base_lr": 8e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 0,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.15,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(core3_weights),
                    "loss_feature_weights": dict(core3_weights),
                    "early_stopping_patience": 15,
                    "scheduler_patience": 6,
                },
            ),
        }

    if preset_name in {"core3_progressive_v2", "daily_core3_progressive_v2"}:
        stage12_weights = {
            "CODMn": 1.5,
            "DO": 1.5,
            "NH4N": 0.25,
            "pH": 1.25,
        }
        stage3_loss_weights = {
            "CODMn": 2.0,
            "DO": 1.4,
            "NH4N": 0.15,
            "pH": 1.25,
        }
        stage3_monitor_weights = {
            "CODMn": 2.25,
            "DO": 1.35,
            "NH4N": 0.10,
            "pH": 1.30,
        }
        return {
            "invalid_window_policy": "target_only",
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "use_temporal_adapter": False,
            "progressive_stages": build_weekly_to_daily_stages(
                stage1_overrides={
                    "window_days": 56,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.05,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage2_overrides={
                    "window_days": 32,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage3_overrides={
                    "window_days": 8,
                    "model_seq_len": 8,
                    "epochs": 80,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 0,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.15,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage3_monitor_weights),
                    "loss_feature_weights": dict(stage3_loss_weights),
                    "sampler_feature": "CODMn",
                    "sampler_threshold": 1.8,
                    "sampler_positive_weight": 4.0,
                    "sampler_positive_mode": "target_above_threshold",
                    "early_stopping_patience": 18,
                    "scheduler_patience": 6,
                },
            ),
        }

    if preset_name in {"core3_progressive_v3", "daily_core3_progressive_v3"}:
        stage12_weights = {
            "CODMn": 1.5,
            "DO": 1.5,
            "NH4N": 0.25,
            "pH": 1.25,
        }
        stage3_loss_weights = {
            "CODMn": 1.75,
            "DO": 1.4,
            "NH4N": 0.20,
            "pH": 1.25,
        }
        stage3_monitor_weights = {
            "CODMn": 1.9,
            "DO": 1.4,
            "NH4N": 0.15,
            "pH": 1.25,
        }
        return {
            "invalid_window_policy": "target_only",
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "use_temporal_adapter": False,
            "progressive_stages": build_weekly_to_daily_stages(
                stage1_overrides={
                    "window_days": 56,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.05,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage2_overrides={
                    "window_days": 32,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage3_overrides={
                    "window_days": 12,
                    "model_seq_len": 12,
                    "epochs": 80,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 0,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.15,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage3_monitor_weights),
                    "loss_feature_weights": dict(stage3_loss_weights),
                    "early_stopping_patience": 18,
                    "scheduler_patience": 6,
                },
            ),
        }

    if preset_name in {"core3_progressive_v4", "daily_core3_progressive_v4"}:
        stage123_weights = {
            "CODMn": 1.5,
            "DO": 1.5,
            "NH4N": 0.25,
            "pH": 1.25,
        }
        stage3_loss_weights = {
            "CODMn": 1.75,
            "DO": 1.4,
            "NH4N": 0.20,
            "pH": 1.25,
        }
        stage3_monitor_weights = {
            "CODMn": 1.9,
            "DO": 1.4,
            "NH4N": 0.15,
            "pH": 1.25,
        }
        return {
            "invalid_window_policy": "target_only",
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "use_temporal_adapter": False,
            "progressive_stages": build_weekly_to_daily_stages(
                stage1_overrides={
                    "window_days": 84,
                    "model_seq_len": 12,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.05,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage123_weights),
                    "loss_feature_weights": dict(stage123_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage2_overrides={
                    "window_days": 48,
                    "model_seq_len": 12,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage123_weights),
                    "loss_feature_weights": dict(stage123_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage3_overrides={
                    "window_days": 12,
                    "model_seq_len": 12,
                    "epochs": 80,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 0,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.15,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage3_monitor_weights),
                    "loss_feature_weights": dict(stage3_loss_weights),
                    "early_stopping_patience": 18,
                    "scheduler_patience": 6,
                },
            ),
        }

    if preset_name in {"core3_only_progressive_v1", "daily_core3_only_progressive_v1"}:
        stage12_weights = {
            "CODMn": 1.6,
            "DO": 1.45,
            "pH": 1.2,
        }
        stage3_loss_weights = {
            "CODMn": 1.9,
            "DO": 1.4,
            "pH": 1.25,
        }
        stage3_monitor_weights = {
            "CODMn": 2.0,
            "DO": 1.4,
            "pH": 1.2,
        }
        return {
            "invalid_window_policy": "target_only",
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": ["CODMn", "DO", "pH"],
            "use_temporal_adapter": False,
            "progressive_stages": build_weekly_to_daily_stages(
                stage1_overrides={
                    "window_days": 56,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.05,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage2_overrides={
                    "window_days": 32,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage3_overrides={
                    "window_days": 12,
                    "model_seq_len": 12,
                    "epochs": 80,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 0,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.15,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage3_monitor_weights),
                    "loss_feature_weights": dict(stage3_loss_weights),
                    "early_stopping_patience": 18,
                    "scheduler_patience": 6,
                },
            ),
        }

    if preset_name in {"core3_progressive_v5", "daily_core3_progressive_v5"}:
        stage12_weights = {
            "CODMn": 1.55,
            "DO": 1.5,
            "NH4N": 0.1,
            "pH": 1.25,
        }
        stage3_loss_weights = {
            "CODMn": 1.85,
            "DO": 1.45,
            "NH4N": 0.05,
            "pH": 1.3,
        }
        stage3_monitor_weights = {
            "CODMn": 1.95,
            "DO": 1.45,
            "NH4N": 0.05,
            "pH": 1.3,
        }
        return {
            "invalid_window_policy": "target_only",
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "use_temporal_adapter": False,
            "progressive_stages": build_weekly_to_daily_stages(
                stage1_overrides={
                    "window_days": 56,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.05,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage2_overrides={
                    "window_days": 32,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage3_overrides={
                    "window_days": 12,
                    "model_seq_len": 12,
                    "epochs": 80,
                    "base_lr": 5e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 0,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.15,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage3_monitor_weights),
                    "loss_feature_weights": dict(stage3_loss_weights),
                    "early_stopping_patience": 18,
                    "scheduler_patience": 6,
                },
            ),
        }

    if preset_name in {"core3_progressive_v2pretrain_v1", "daily_core3_progressive_v2pretrain_v1"}:
        stage12_weights = {
            "CODMn": 1.55,
            "DO": 1.5,
            "NH4N": 0.1,
            "pH": 1.25,
        }
        stage3_loss_weights = {
            "CODMn": 1.85,
            "DO": 1.45,
            "NH4N": 0.05,
            "pH": 1.3,
        }
        stage3_monitor_weights = {
            "CODMn": 1.95,
            "DO": 1.45,
            "NH4N": 0.05,
            "pH": 1.3,
        }
        return {
            "invalid_window_policy": "target_only",
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "use_temporal_adapter": True,
            "temporal_adapter_kernel_size": 5,
            "progressive_stages": build_weekly_to_daily_stages(
                stage1_overrides={
                    "window_days": 56,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.05,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage2_overrides={
                    "window_days": 32,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage3_overrides={
                    "window_days": 12,
                    "model_seq_len": 12,
                    "epochs": 80,
                    "base_lr": 5e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 0,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.15,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage3_monitor_weights),
                    "loss_feature_weights": dict(stage3_loss_weights),
                    "early_stopping_patience": 18,
                    "scheduler_patience": 6,
                },
            ),
        }

    if preset_name in {"core3_progressive_v2pretrain_v2", "daily_core3_progressive_v2pretrain_v2"}:
        stage12_weights = {
            "CODMn": 1.55,
            "DO": 1.5,
            "NH4N": 0.1,
            "pH": 1.25,
        }
        stage3_loss_weights = {
            "CODMn": 1.8,
            "DO": 1.4,
            "NH4N": 0.05,
            "pH": 1.45,
        }
        stage3_monitor_weights = {
            "CODMn": 1.85,
            "DO": 1.4,
            "NH4N": 0.05,
            "pH": 1.55,
        }
        return {
            "invalid_window_policy": "target_only",
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "use_temporal_adapter": True,
            "temporal_adapter_kernel_size": 5,
            "progressive_stages": build_weekly_to_daily_stages(
                stage1_overrides={
                    "window_days": 56,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.05,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage2_overrides={
                    "window_days": 32,
                    "model_seq_len": 8,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage12_weights),
                    "loss_feature_weights": dict(stage12_weights),
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "freeze_backbone_epochs": 0,
                },
                stage3_overrides={
                    "window_days": 12,
                    "model_seq_len": 12,
                    "epochs": 80,
                    "base_lr": 4e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 1,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.15,
                    "monitor_metric": "nse",
                    "monitor_feature_weights": dict(stage3_monitor_weights),
                    "loss_feature_weights": dict(stage3_loss_weights),
                    "early_stopping_patience": 18,
                    "scheduler_patience": 6,
                },
            ),
        }

    if preset_name in {"base_0824_v1", "daily_base_0824_v1"}:
        return {
            "time_end": WEATHER_COVERAGE_END,
            "stage_time_ranges": build_shared_stage_time_ranges(
                "2023-01-01 00:00:00",
                WEATHER_COVERAGE_END,
            ),
            "invalid_window_policy": "target_only",
        }

    if preset_name in {"weather_base_0824_v1", "daily_weather_base_0824_v1"}:
        return {
            "time_end": WEATHER_COVERAGE_END,
            "stage_time_ranges": build_shared_stage_time_ranges(
                "2023-01-01 00:00:00",
                WEATHER_COVERAGE_END,
            ),
            "invalid_window_policy": "target_only",
            "progressive_stages": build_weekly_to_daily_stages(
                stage3_overrides={
                    "meteorology": {
                        "enabled": True,
                    },
                },
            ),
        }

    if preset_name in {"nh4n_floor_guard_0824_v1", "daily_nh4n_floor_0824_v1"}:
        return {
            "time_end": WEATHER_COVERAGE_END,
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "stage_time_ranges": build_shared_stage_time_ranges(
                "2023-01-01 00:00:00",
                WEATHER_COVERAGE_END,
            ),
            "invalid_window_policy": "target_only",
            "progressive_stages": build_weekly_to_daily_stages(
                stage3_overrides={
                    "window_days": 28,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "postprocess_mode": "floor_guard",
                    "postprocess_feature": "NH4N",
                    "postprocess_floor": 0.025,
                    "postprocess_background_threshold": 0.03,
                    "postprocess_prev_spike_threshold": 0.05,
                    "postprocess_selection": "background_rmse",
                },
            ),
        }

    if preset_name in {"nh4n_weather_dual_station_v1", "daily_nh4n_weather_v1"}:
        return {
            "time_end": WEATHER_COVERAGE_END,
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "stage_time_ranges": build_shared_stage_time_ranges(
                "2023-01-01 00:00:00",
                WEATHER_COVERAGE_END,
            ),
            "invalid_window_policy": "target_only",
            "progressive_stages": build_weekly_to_daily_stages(
                stage3_overrides={
                    "window_days": 28,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "postprocess_mode": "floor_guard",
                    "postprocess_feature": "NH4N",
                    "postprocess_floor": 0.025,
                    "postprocess_background_threshold": 0.03,
                    "postprocess_prev_spike_threshold": 0.05,
                    "postprocess_selection": "background_rmse",
                    "meteorology": {
                        "enabled": True,
                    },
                },
            ),
        }

    if preset_name in {"nh4n_weather_two_stage_v1", "daily_nh4n_weather_two_stage_v1"}:
        return {
            "time_end": WEATHER_COVERAGE_END,
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "stage_time_ranges": build_shared_stage_time_ranges(
                "2023-01-01 00:00:00",
                WEATHER_COVERAGE_END,
            ),
            "invalid_window_policy": "target_only",
            "progressive_stages": build_weekly_to_daily_stages(
                stage3_overrides={
                    "window_days": 28,
                    "epochs": 60,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 2,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "monitor_feature": "NH4N",
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "meteorology": {
                        "enabled": True,
                    },
                    "nh4n_two_stage": {
                        "enabled": True,
                        "feature": "NH4N",
                        "floor": 0.025,
                        "spike_threshold": 0.05,
                        "event_loss_weight": 0.5,
                        "event_pos_weight": 12.0,
                        "excess_loss_weight": 1.0,
                        "excess_positive_only": True,
                    },
                },
            ),
        }

    if preset_name in {"nh4n_weather_two_stage_v2", "daily_nh4n_weather_two_stage_v2"}:
        return {
            "time_end": WEATHER_COVERAGE_END,
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "stage_time_ranges": build_shared_stage_time_ranges(
                "2023-01-01 00:00:00",
                WEATHER_COVERAGE_END,
            ),
            "invalid_window_policy": "target_only",
            "progressive_stages": build_weekly_to_daily_stages(
                stage3_overrides={
                    "window_days": 28,
                    "epochs": 60,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 2,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "loss_feature_weights": {
                        "NH4N": 2.5,
                    },
                    "monitor_metric": "nse",
                    "monitor_feature": "NH4N",
                    "early_stopping_patience": 15,
                    "scheduler_patience": 6,
                    "sampler_feature": "NH4N",
                    "sampler_threshold": 0.05,
                    "sampler_positive_weight": 8.0,
                    "sampler_positive_mode": "target_spike_onset",
                    "meteorology": {
                        "enabled": True,
                    },
                    "nh4n_two_stage": {
                        "enabled": True,
                        "feature": "NH4N",
                        "floor": 0.025,
                        "spike_threshold": 0.05,
                        "event_loss_weight": 0.75,
                        "event_pos_weight": 16.0,
                        "excess_loss_weight": 1.0,
                        "excess_positive_only": True,
                    },
                },
            ),
        }

    if preset_name in {"nh4n_weather_two_stage_v3", "daily_nh4n_weather_two_stage_v3"}:
        return {
            "time_end": WEATHER_COVERAGE_END,
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "stage_time_ranges": build_shared_stage_time_ranges(
                "2023-01-01 00:00:00",
                WEATHER_COVERAGE_END,
            ),
            "invalid_window_policy": "target_only",
            "progressive_stages": build_weekly_to_daily_stages(
                stage3_overrides={
                    "window_days": 28,
                    "epochs": 60,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 2,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "monitor_feature": "NH4N",
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "sampler_feature": "NH4N",
                    "sampler_threshold": 0.05,
                    "sampler_positive_weight": 4.0,
                    "sampler_positive_mode": "target_spike_onset",
                    "meteorology": {
                        "enabled": True,
                    },
                    "nh4n_two_stage": {
                        "enabled": True,
                        "feature": "NH4N",
                        "floor": 0.025,
                        "spike_threshold": 0.05,
                        "event_loss_weight": 0.5,
                        "event_pos_weight": 12.0,
                        "excess_loss_weight": 1.0,
                        "excess_positive_only": True,
                    },
                },
            ),
        }

    if preset_name in {"nh4n_two_stage_v1", "daily_nh4n_two_stage_v1"}:
        return {
            "invalid_window_policy": "target_only",
            "data_root": os.path.join(REPO_ROOT, "data", "water_quality_processed_2023_2025"),
            "feature_columns": list(NH4N_FEATURE_COLUMNS),
            "progressive_stages": build_weekly_to_daily_stages(
                stage3_overrides={
                    "window_days": 28,
                    "epochs": 60,
                    "base_lr": 6e-5,
                    "weight_decay": 2e-4,
                    "freeze_backbone_epochs": 2,
                    "invalid_window_policy": "all",
                    "soft_gap_max_steps": 6,
                    "loss_name": "mse_nse",
                    "nse_weight": 0.1,
                    "monitor_metric": "nse",
                    "monitor_feature": "NH4N",
                    "early_stopping_patience": 12,
                    "scheduler_patience": 5,
                    "nh4n_two_stage": {
                        "enabled": True,
                        "feature": "NH4N",
                        "floor": 0.025,
                        "spike_threshold": 0.05,
                        "event_loss_weight": 0.5,
                        "event_pos_weight": 12.0,
                        "excess_loss_weight": 1.0,
                        "excess_positive_only": True,
                    },
                },
            ),
        }

    raise ValueError(f"未知 finetune preset: {preset_name}")


def main(pretrain_model_dir=None, custom_config=None, seed=42):
    set_seed(seed)
    config = FinetuneConfig()
    config.pretrain_model_dir = pretrain_model_dir

    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    config.feature_columns = list(config.feature_columns)
    config.input_dim = len(config.feature_columns)

    print(f"\n{'=' * 70}")
    print(f"Progressive finetune 开始 | seed={seed}")
    print(f"pretrain_model_dir: {pretrain_model_dir}")
    print(f"{'=' * 70}")

    uses_pretraining = pretrain_model_dir is not None
    pretrain_config = {}
    pretrain_state = None
    if uses_pretraining:
        with open(
            os.path.join(pretrain_model_dir, "config.json"),
            "r",
            encoding="utf-8",
        ) as file:
            pretrain_config = json.load(file)

        config.backbone_name = normalize_backbone_name(
            pretrain_config.get("backbone_name", "transformer")
        )
        config.model_agnostic_interface = normalize_model_agnostic_interface(
            pretrain_config.get("model_agnostic_interface", "legacy")
        )
        config.hidden_size = int(pretrain_config.get("hidden_size", config.hidden_size))
        config.num_heads = int(pretrain_config.get("num_heads", config.num_heads))
        config.e_layer = int(pretrain_config.get("e_layer", config.e_layer))
        config.model_seq_len = pretrain_config.get("model_seq_len", config.model_seq_len)
    else:
        config.backbone_name = normalize_backbone_name(config.backbone_name)
        config.model_agnostic_interface = normalize_model_agnostic_interface(
            config.model_agnostic_interface
        )
    architecture_config_keys = (
        "cnn_channels",
        "cnn_kernel_sizes",
        "cnn_dilations",
        "cnn_use_batch_norm",
        "lstm_hidden_dim",
        "lstm_num_layers",
        "lstm_use_input_layer_norm",
        "backbone_dropout",
        "backbone_activation",
        "backbone_head_hidden_dim",
        "mlp_hidden_dims",
        "mlp_dropout",
        "mlp_use_layer_norm",
        "cnn_lstm_conv_channels",
        "cnn_lstm_kernel_sizes",
        "cnn_lstm_hidden_dim",
        "cnn_lstm_layers",
        "cnn_lstm_use_batch_norm",
        "cnn_lstm_use_input_layer_norm",
    )
    for key in architecture_config_keys:
        if key in pretrain_config:
            value = pretrain_config[key]
            if key in {
                "cnn_channels",
                "cnn_kernel_sizes",
                "cnn_dilations",
                "mlp_hidden_dims",
                "cnn_lstm_conv_channels",
                "cnn_lstm_kernel_sizes",
            }:
                value = tuple(value)
            setattr(config, key, value)

    if uses_pretraining:
        pretrain_state = torch.load(
            os.path.join(pretrain_model_dir, "model.pth"),
            map_location="cpu",
        )
    stage_specs = build_stage_specs(config)

    final_resolution = stage_specs[-1]["resolution"] if stage_specs else "daily"
    station_names = config.target_station_names or list_station_names(config.data_root, final_resolution)
    if config.max_target_stations is not None:
        station_names = station_names[:config.max_target_stations]

    print(f"目标站点数: {len(station_names)}")
    print(f"backbone: {config.backbone_name}")
    print(f"model_agnostic_interface: {config.model_agnostic_interface}")
    print(f"data_root: {config.data_root}")
    print(f"time_range: {config.time_start} -> {config.time_end}")
    if config.stage_time_ranges:
        print(f"stage_time_ranges: {config.stage_time_ranges}")
    print(f"filter_invalid_windows: {config.filter_invalid_windows}")
    print(f"invalid_window_policy: {config.invalid_window_policy}")
    print(f"save_model_weights: {config.save_model_weights}")
    print("阶段配置:")
    for stage in stage_specs:
        stage_policy = stage.get("invalid_window_policy", config.invalid_window_policy)
        soft_gap_max_steps = stage.get("soft_gap_max_steps")
        soft_gap_text = f" | soft_gap<={soft_gap_max_steps}" if soft_gap_max_steps is not None else ""
        monitor_text = (
            f" | monitor={stage.get('monitor_metric', config.monitor_metric)}"
            f":{stage.get('monitor_feature', '__overall__')}"
            if stage.get("monitor_metric") == "nse"
            else ""
        )
        if stage.get("monitor_metric") == "nse" and stage.get("monitor_feature_weights"):
            monitor_text = (
                f" | monitor=weighted_nse:{stage['monitor_feature_weights']}"
            )
        sampler_text = (
            f" | sampler={stage['sampler_feature']}@{stage['sampler_threshold']}"
            if stage.get("sampler_feature")
            else ""
        )
        postprocess_text = (
            f" | postprocess={stage.get('postprocess_mode')}:{stage.get('postprocess_feature')}"
            if stage.get("postprocess_mode")
            else ""
        )
        weather_text = ""
        if stage.get("meteorology", {}).get("enabled"):
            station_count = len(stage.get("meteorology", {}).get("stations") or [])
            weather_feature_count = len(
                build_weather_feature_columns(
                    stage.get("meteorology", {}).get("stations") or YANGSHUO_WEATHER_STATIONS
                )
            )
            weather_text = (
                f" | meteorology={station_count or 2}stations/{weather_feature_count}feat"
            )
        two_stage_text = (
            f" | two_stage={stage['nh4n_two_stage'].get('feature', config.feature_columns[2])}"
            f"@{stage['nh4n_two_stage'].get('spike_threshold', 'n/a')}"
            if stage.get("nh4n_two_stage", {}).get("enabled")
            else ""
        )
        print(
            f"  - {stage['name']}: resolution={stage['resolution']} "
            f"| raw_seq_len={stage['raw_seq_len']} | pred_steps={stage['pred_steps']} "
            f"| invalid_policy={stage_policy}{soft_gap_text}{monitor_text}{sampler_text}{postprocess_text}{weather_text}{two_stage_text}"
        )

    results_summary = []

    for index, station_name in enumerate(station_names, start=1):
        print(f"\n{'#' * 70}")
        print(f"[{index}/{len(station_names)}] 站点: {station_name}")
        print(f"{'#' * 70}")

        station_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        station_save_dir = os.path.join(
            config.save_dir,
            f"progressive_{station_name}_seed{seed}_{station_timestamp}",
        )
        os.makedirs(station_save_dir, exist_ok=True)

        current_state = pretrain_state

        station_stage_results = []
        station_failed = False

        for stage in stage_specs:
            print(
                f"\n>>> {stage['name']} | resolution={stage['resolution']} "
                f"| window_days={stage['window_days']}"
            )

            stage_data = load_stage_data(config, station_name, stage)
            if stage_data is None:
                print(f"[跳过] {station_name} 在 {stage['name']} 阶段数据不足或缺失。")
                station_failed = True
                break

            soft_gap_text = "off" if stage_data["soft_gap_max_steps"] is None else f"<={stage_data['soft_gap_max_steps']}"
            sampler_stats = stage_data["train_sampler_stats"]
            sampler_text = ""
            if sampler_stats is not None:
                sampler_text = (
                    f" | sampler={sampler_stats['feature']}:{sampler_stats['positive_mode']} "
                    f"{sampler_stats['positive_window_count']}/{stage_data['train_windows']}"
                )
            print(
                f"数据摘要 | records={stage_data['records']} "
                f"| gap_records={stage_data['invalid_records']} "
                f"| weather_gap_records={stage_data['weather_invalid_records']} "
                f"| input_invalid={stage_data['input_invalid_records']} "
                f"| target_invalid={stage_data['target_invalid_records']} "
                f"| time_range={stage_data['time_start']} -> {stage_data['time_end']} "
                f"| invalid_policy={stage_data['invalid_window_policy']} "
                f"| soft_gap={soft_gap_text} "
                f"| input_dim={stage_data['input_dim']} "
                f"| train={stage_data['train_windows']}/{stage_data['train_candidate_windows']} "
                f"| val={stage_data['val_windows']}/{stage_data['val_candidate_windows']} "
                f"| test={stage_data['test_windows']}/{stage_data['test_candidate_windows']}"
                f"{sampler_text}"
            )

            nh4n_two_stage_config = build_stage_nh4n_two_stage_config(
                stage,
                stage_data["scaler"],
                config.feature_columns,
            )
            if stage.get("reset_model_seed") is not None:
                set_seed(int(stage["reset_model_seed"]))
            stage_model = build_model(
                config,
                {**stage, "input_dim": stage_data["input_dim"]},
                nh4n_two_stage_config=nh4n_two_stage_config,
            )
            skip_weight_prefixes = stage.get("skip_weight_prefixes")
            load_weight_blend_alpha = float(
                stage.get("load_weight_blend_alpha", 1.0)
            )
            matched_keys = {}
            if current_state is not None:
                matched_keys = load_matching_weights(
                    stage_model,
                    current_state,
                    skip_prefixes=skip_weight_prefixes,
                    blend_alpha=load_weight_blend_alpha,
                )

            train_start = time.time()
            stage_model, history, best_stats = fit_model(
                model=stage_model,
                train_loader=stage_data["train_loader"],
                val_loader=stage_data["val_loader"],
                device=config.device,
                epochs=stage.get("epochs", config.epochs),
                base_lr=stage.get("base_lr", config.base_lr),
                epsilon=config.epsilon,
                weight_decay=stage.get("weight_decay", config.weight_decay),
                lr_milestones=stage.get("lr_milestones", config.lr_milestones),
                lr_decay_ratio=stage.get("lr_decay_ratio", config.lr_decay_ratio),
                max_grad_norm=config.max_grad_norm,
                log_prefix=stage["name"],
                loss_name=stage.get("loss_name", config.loss_name),
                nse_weight=stage.get("nse_weight", config.nse_weight),
                loss_feature_weights=stage.get("loss_feature_weights"),
                multitask_config=nh4n_two_stage_config,
                monitor_metric=stage.get("monitor_metric", config.monitor_metric),
                monitor_feature=stage.get("monitor_feature"),
                monitor_feature_weights=stage.get("monitor_feature_weights"),
                early_stopping_patience=stage.get(
                    "early_stopping_patience",
                    config.early_stopping_patience,
                ),
                early_stopping_min_delta=stage.get(
                    "early_stopping_min_delta",
                    config.early_stopping_min_delta,
                ),
                scheduler_name=stage.get("scheduler_name", config.scheduler_name),
                scheduler_patience=stage.get("scheduler_patience", config.scheduler_patience),
                scheduler_min_lr=stage.get("scheduler_min_lr", config.scheduler_min_lr),
                freeze_backbone_epochs=stage.get(
                    "freeze_backbone_epochs",
                    config.freeze_backbone_epochs,
                ),
                feature_names=config.feature_columns,
            )
            train_seconds = time.time() - train_start
            best_val_loss = best_stats["val_loss"]
            best_val_nse = best_stats["val_nse"]
            best_epoch = best_stats["epoch"]
            postprocess_params = None
            postprocess_val_metrics = None
            if stage.get("postprocess_mode"):
                _, val_preds, val_targets, val_timestamps = evaluate_model(
                    stage_model,
                    stage_data["val_loader"],
                    config.device,
                    scaler=stage_data["scaler"],
                )
                postprocess_params = fit_stage_postprocess(
                    stage,
                    stage_data["frame"],
                    val_preds,
                    val_targets,
                    val_timestamps,
                    config.feature_columns,
                )
                if postprocess_params is not None:
                    val_preds = apply_stage_postprocess(
                        postprocess_params,
                        stage_data["frame"],
                        val_preds,
                        val_timestamps,
                        config.feature_columns,
                    )
                    postprocess_val_metrics = compute_per_feature_metrics(
                        val_preds,
                        val_targets,
                        config.feature_columns,
                    )

            test_loss, preds, targets, timestamps = evaluate_model(
                stage_model,
                stage_data["test_loader"],
                config.device,
                scaler=stage_data["scaler"],
            )
            if postprocess_params is not None:
                preds = apply_stage_postprocess(
                    postprocess_params,
                    stage_data["frame"],
                    preds,
                    timestamps,
                    config.feature_columns,
                )
            metrics = compute_per_feature_metrics(preds, targets, config.feature_columns)
            test_nse = metrics["__overall__"]["NSE"]

            stage_dir = os.path.join(station_save_dir, stage["name"])
            os.makedirs(stage_dir, exist_ok=True)
            model_weights_path = os.path.join(stage_dir, "model.pth")
            model_weights_saved = bool(config.save_model_weights)
            if model_weights_saved:
                torch.save(stage_model.state_dict(), model_weights_path)
            else:
                print(f"[{stage['name']}] 跳过保存模型权重: {model_weights_path}")
            with open(os.path.join(stage_dir, "scaler.pkl"), "wb") as file:
                pickle.dump(stage_data["scaler"], file)

            meta = {
                "station_name": station_name,
                "backbone_name": config.backbone_name,
                "model_agnostic_interface": config.model_agnostic_interface,
                "uses_pretraining": uses_pretraining,
                "initialization_source": (
                    "cross_station_pretrain" if uses_pretraining else "random"
                ),
                "pretrain_model_dir": pretrain_model_dir,
                "optimization_profile": config.optimization_profile,
                "stage_name": stage["name"],
                "resolution": stage["resolution"],
                "window_days": stage["window_days"],
                "raw_seq_len": stage["raw_seq_len"],
                "model_seq_len": stage["model_seq_len"],
                "pred_steps": stage["pred_steps"],
                "resize_mode": stage.get("resize_mode", config.resize_mode),
                "records": stage_data["records"],
                "invalid_records": stage_data["invalid_records"],
                "valid_records": stage_data["valid_records"],
                "weather_invalid_records": stage_data["weather_invalid_records"],
                "input_dim": stage_data["input_dim"],
                "input_feature_columns": stage_data["input_feature_columns"],
                "feature_columns": list(config.feature_columns),
                "weather_feature_columns": stage_data["weather_feature_columns"],
                "weather_metadata": stage_data["weather_metadata"],
                "model_weights_saved": model_weights_saved,
                "model_weights_path": model_weights_path if model_weights_saved else None,
                "train_windows": stage_data["train_windows"],
                "train_candidate_windows": stage_data["train_candidate_windows"],
                "train_filtered_windows": stage_data["train_filtered_windows"],
                "val_windows": stage_data["val_windows"],
                "val_candidate_windows": stage_data["val_candidate_windows"],
                "val_filtered_windows": stage_data["val_filtered_windows"],
                "test_windows": stage_data["test_windows"],
                "test_candidate_windows": stage_data["test_candidate_windows"],
                "test_filtered_windows": stage_data["test_filtered_windows"],
                "time_start": stage_data["time_start"],
                "time_end": stage_data["time_end"],
                "stage_time_ranges": config.stage_time_ranges,
                "filter_invalid_windows": stage_data["filter_invalid_windows"],
                "invalid_window_policy": stage_data["invalid_window_policy"],
                "soft_gap_max_steps": stage_data["soft_gap_max_steps"],
                "skip_weight_prefixes": skip_weight_prefixes,
                "load_weight_blend_alpha": load_weight_blend_alpha,
                "reset_model_seed": stage.get("reset_model_seed"),
                "input_invalid_records": stage_data["input_invalid_records"],
                "target_invalid_records": stage_data["target_invalid_records"],
                "gap_segment_count": stage_data["gap_segment_count"],
                "soft_gap_segment_count": stage_data["soft_gap_segment_count"],
                "hard_gap_segment_count": stage_data["hard_gap_segment_count"],
                "soft_gap_records": stage_data["soft_gap_records"],
                "hard_gap_records": stage_data["hard_gap_records"],
                "weather_gap_segment_count": stage_data["weather_gap_segment_count"],
                "weather_soft_gap_segment_count": stage_data["weather_soft_gap_segment_count"],
                "weather_hard_gap_segment_count": stage_data["weather_hard_gap_segment_count"],
                "weather_soft_gap_records": stage_data["weather_soft_gap_records"],
                "weather_hard_gap_records": stage_data["weather_hard_gap_records"],
                "loss_feature_weights": stage.get("loss_feature_weights"),
                "monitor_feature_weights": stage.get("monitor_feature_weights"),
                "nh4n_two_stage": nh4n_two_stage_config,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_nse": best_val_nse,
                "best_val_monitor_nse": best_stats["val_monitor_nse"],
                "test_loss": test_loss,
                "test_nse": test_nse,
                "train_seconds": train_seconds,
                "loaded_pretrain_keys": len(matched_keys) if not station_stage_results else None,
                "loaded_stage_keys": len(matched_keys) if station_stage_results else None,
                "monitor_metric": best_stats["monitor_metric"],
                "monitor_feature": best_stats["monitor_feature"],
                "best_monitor_feature_weights": best_stats["monitor_feature_weights"],
                "monitor_value": best_stats["monitor_value"],
                "train_sampler_stats": stage_data["train_sampler_stats"],
                "postprocess_params": postprocess_params,
                "postprocess_val_metrics": (
                    postprocess_val_metrics["__overall__"]
                    if postprocess_val_metrics is not None
                    else None
                ),
                "use_temporal_adapter": config.use_temporal_adapter,
                "temporal_adapter_kernel_size": config.temporal_adapter_kernel_size,
                "metrics": metrics["__overall__"],
            }
            save_stage_outputs(
                stage_dir,
                history,
                preds,
                targets,
                timestamps,
                metrics,
                meta,
                config.feature_columns,
            )

            current_state = snapshot_state_dict(stage_model)
            station_stage_results.append(
                {
                    "stage_name": stage["name"],
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_nse": best_val_nse,
                    "test_loss": test_loss,
                    "test_nse": test_nse,
                    "save_dir": stage_dir,
                    "model_weights_saved": model_weights_saved,
                    "model_weights_path": model_weights_path if model_weights_saved else None,
                }
            )

            print(
                f">>> {stage['name']} 完成 | best_epoch={best_epoch} "
                f"| best_val={best_val_loss:.6f} | best_val_nse={best_val_nse:.6f} "
                f"| test={test_loss:.6f} | test_nse={test_nse:.6f}"
            )

        summary = {
            "station_name": station_name,
            "backbone_name": config.backbone_name,
            "model_agnostic_interface": config.model_agnostic_interface,
            "uses_pretraining": uses_pretraining,
            "initialization_source": (
                "cross_station_pretrain" if uses_pretraining else "random"
            ),
            "pretrain_model_dir": pretrain_model_dir,
            "optimization_profile": config.optimization_profile,
            "status": "failed" if station_failed else "completed",
            "stages": station_stage_results,
            "save_dir": station_save_dir,
        }
        with open(os.path.join(station_save_dir, "summary.json"), "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        results_summary.append(summary)

    print(f"\n{'=' * 70}")
    print(f"Progressive finetune 结束 | stations={len(results_summary)}")
    print(f"{'=' * 70}")

    for result in results_summary:
        if not result["stages"]:
            print(f"  - {result['station_name']}: failed")
            continue
        last_stage = result["stages"][-1]
        print(
            f"  - {result['station_name']}: "
            f"{last_stage['stage_name']} | val={last_stage['best_val_loss']:.6f} "
            f"| val_nse={last_stage['best_val_nse']:.6f} | test={last_stage['test_loss']:.6f} "
            f"| test_nse={last_stage['test_nse']:.6f}"
        )

    return results_summary


if __name__ == "__main__":
    latest_pretrain_dir = find_latest_pretrain_run(PRETRAIN_RUNS_DIR)
    if latest_pretrain_dir is None:
        print("未找到可用的 pretrain run。请先生成包含 config.json 和 model.pth 的预训练结果。")
    else:
        print(f"自动选择最新 pretrain run: {latest_pretrain_dir}")
        preset_name = os.environ.get("PTL_FINETUNE_PRESET", "").strip()
        custom_config = build_finetune_preset(preset_name) if preset_name else None
        if preset_name:
            print(f"使用 finetune preset: {preset_name}")
        main(pretrain_model_dir=latest_pretrain_dir, custom_config=custom_config, seed=42)
