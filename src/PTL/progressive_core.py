import copy
import os
import random
from math import sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


FEATURE_COLUMNS = ["CODMn", "DO", "TP", "pH"]
RESOLUTION_ALIASES = {
    "week": "weekly",
}
RESOLUTION_TO_FREQ = {
    "4h": "4h",
    "12h": "12h",
    "4d": "4D",
    "daily": "1D",
    "weekly": "1W",
}
RESOLUTION_SOURCE_PRIORITY = {
    "4h": ("4h",),
    "12h": ("12h", "4h"),
    "4d": ("4d", "daily", "4h"),
    "daily": ("daily", "4h"),
    "weekly": ("weekly", "daily", "4h"),
}
MIN_NSE_DENOMINATOR = 1e-8


def infer_device():
    if torch.backends.mps.is_available():
        print("Using Apple MPS acceleration.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("Using CPU.")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return (data * (self.std + 1e-8)) + self.mean


def snapshot_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def normalize_resolution(resolution):
    normalized = str(resolution).strip().lower()
    return RESOLUTION_ALIASES.get(normalized, normalized)


def get_resolution_source_priority(resolution):
    resolution = normalize_resolution(resolution)
    if resolution not in RESOLUTION_TO_FREQ:
        raise ValueError(f"Unsupported resolution: {resolution}")
    return RESOLUTION_SOURCE_PRIORITY.get(resolution, (resolution,))


def load_matching_weights(model, state_dict, skip_prefixes=None, blend_alpha=1.0):
    blend_alpha = float(blend_alpha)
    if not 0.0 < blend_alpha <= 1.0:
        raise ValueError(f"blend_alpha must be in (0, 1], got {blend_alpha}")
    skip_prefixes = tuple(skip_prefixes or ())
    current_state = model.state_dict()
    matched_state = {}
    for key, value in state_dict.items():
        if (
            key not in current_state
            or tuple(value.shape) != tuple(current_state[key].shape)
            or any(key.startswith(prefix) for prefix in skip_prefixes)
        ):
            continue
        if blend_alpha < 1.0 and torch.is_floating_point(value):
            matched_state[key] = (
                current_state[key] * (1.0 - blend_alpha)
                + value.to(dtype=current_state[key].dtype) * blend_alpha
            )
        else:
            matched_state[key] = value
    current_state.update(matched_state)
    model.load_state_dict(current_state, strict=False)
    return matched_state


def _normalize_feature_weights(feature_weights, feature_names=None):
    if feature_weights is None:
        return None

    feature_names = feature_names or FEATURE_COLUMNS
    if isinstance(feature_weights, dict):
        unknown_names = sorted(set(feature_weights) - set(feature_names))
        if unknown_names:
            raise ValueError(f"Unknown feature weights: {unknown_names}")
        weights = np.asarray(
            [feature_weights.get(name, 1.0) for name in feature_names],
            dtype=np.float32,
        )
    else:
        weights = np.asarray(feature_weights, dtype=np.float32)
        if weights.ndim != 1 or len(weights) != len(feature_names):
            raise ValueError(
                f"feature_weights must have length {len(feature_names)}, got shape {weights.shape}."
            )

    if np.any(weights <= 0):
        raise ValueError("feature_weights must be strictly positive.")
    return weights / float(np.mean(weights))


def _coerce_feature_weights_tensor(feature_weights, feature_count, device, dtype):
    if feature_weights is None:
        return None

    if isinstance(feature_weights, dict):
        feature_weights = _normalize_feature_weights(feature_weights)

    if torch.is_tensor(feature_weights):
        weights = feature_weights.to(device=device, dtype=dtype).flatten()
    else:
        weights = torch.as_tensor(feature_weights, device=device, dtype=dtype).flatten()

    if weights.numel() != feature_count:
        raise ValueError(
            f"Expected {feature_count} feature weights, received {weights.numel()}."
        )
    if torch.any(weights <= 0):
        raise ValueError("feature_weights must be strictly positive.")
    return weights / torch.mean(weights)


def compute_nse_torch(preds, targets, feature_weights=None, eps=MIN_NSE_DENOMINATOR):
    if preds.shape != targets.shape:
        raise ValueError("Predictions and targets must share the same shape for NSE.")

    reduce_dims = tuple(range(preds.ndim - 1))
    target_means = targets.mean(dim=reduce_dims, keepdim=True)
    numerator = torch.sum((preds - targets) ** 2, dim=reduce_dims)
    denominator = torch.sum((targets - target_means) ** 2, dim=reduce_dims)
    valid_mask = denominator > eps
    if not torch.any(valid_mask):
        return torch.full((), float("nan"), device=preds.device, dtype=preds.dtype)
    nse_per_feature = 1.0 - (numerator[valid_mask] / denominator[valid_mask])
    weights = _coerce_feature_weights_tensor(
        feature_weights,
        feature_count=preds.shape[-1],
        device=preds.device,
        dtype=preds.dtype,
    )
    if weights is None:
        return torch.mean(nse_per_feature)

    valid_weights = weights[valid_mask]
    return torch.sum(valid_weights * nse_per_feature) / torch.sum(valid_weights)


def compute_weighted_nse_from_metrics(metrics, feature_weights, feature_names=None):
    if feature_weights is None:
        return None

    feature_names = list(feature_names or FEATURE_COLUMNS)
    normalized_weights = _normalize_feature_weights(
        feature_weights,
        feature_names=feature_names,
    )

    weighted_sum = 0.0
    weight_sum = 0.0
    for feature_name, weight in zip(feature_names, normalized_weights):
        feature_metrics = metrics.get(feature_name) or {}
        nse_value = feature_metrics.get("NSE")
        if nse_value is None or not np.isfinite(nse_value):
            continue
        weighted_sum += float(weight) * float(nse_value)
        weight_sum += float(weight)

    if weight_sum <= 0.0:
        return float("nan")
    return weighted_sum / weight_sum


def compute_training_loss(
    preds,
    targets,
    loss_name="mse",
    nse_weight=0.0,
    feature_weights=None,
    aux_outputs=None,
    multitask_config=None,
):
    weights = _coerce_feature_weights_tensor(
        feature_weights,
        feature_count=preds.shape[-1],
        device=preds.device,
        dtype=preds.dtype,
    )
    squared_error = (preds - targets) ** 2
    if weights is None:
        mse_loss = torch.mean(squared_error)
    else:
        weight_shape = [1] * preds.ndim
        weight_shape[-1] = weights.shape[0]
        mse_loss = torch.mean(squared_error * weights.view(*weight_shape))

    if loss_name == "mse":
        total_loss = mse_loss
        nse_penalty = torch.zeros((), device=preds.device, dtype=preds.dtype)
    elif loss_name == "mse_nse":
        nse_score = compute_nse_torch(preds, targets, feature_weights=weights)
        if torch.isfinite(nse_score):
            nse_penalty = 1.0 - nse_score
        else:
            nse_penalty = torch.zeros((), device=preds.device, dtype=preds.dtype)
        total_loss = mse_loss + (nse_weight * nse_penalty)
    else:
        raise ValueError(f"Unsupported loss_name: {loss_name}")

    event_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)
    excess_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)
    multitask_config = multitask_config or {}
    if multitask_config.get("enabled"):
        aux_outputs = aux_outputs or {}
        event_logits = aux_outputs.get("nh4n_event_logits")
        excess_pred = aux_outputs.get("nh4n_excess")
        if event_logits is None or excess_pred is None:
            raise ValueError("Two-stage training requires event and excess auxiliary outputs.")

        event_targets, excess_targets = _build_nh4n_two_stage_targets(targets, multitask_config)
        event_pos_weight = multitask_config.get("event_pos_weight")
        pos_weight_tensor = (
            torch.as_tensor(event_pos_weight, device=preds.device, dtype=preds.dtype)
            if event_pos_weight is not None
            else None
        )
        event_loss = F.binary_cross_entropy_with_logits(
            event_logits,
            event_targets,
            pos_weight=pos_weight_tensor,
        )

        if multitask_config.get("excess_positive_only", True):
            positive_mask = event_targets > 0.5
            if torch.any(positive_mask):
                excess_loss = F.mse_loss(excess_pred[positive_mask], excess_targets[positive_mask])
        else:
            excess_loss = F.mse_loss(excess_pred, excess_targets)

        total_loss = total_loss + (
            float(multitask_config.get("event_loss_weight", 1.0)) * event_loss
        ) + (
            float(multitask_config.get("excess_loss_weight", 1.0)) * excess_loss
        )

    return total_loss, {
        "mse": float(mse_loss.detach().cpu()),
        "nse_penalty": float(nse_penalty.detach().cpu()),
        "event_bce": float(event_loss.detach().cpu()),
        "excess_mse": float(excess_loss.detach().cpu()),
    }


def _build_nh4n_two_stage_targets(targets, multitask_config):
    feature_index = int(multitask_config["feature_index"])
    floor_scaled = float(multitask_config["floor_scaled"])
    spike_threshold_scaled = float(multitask_config["spike_threshold_scaled"])
    feature_targets = targets[..., feature_index:feature_index + 1]
    event_targets = (feature_targets >= spike_threshold_scaled).to(dtype=targets.dtype)
    excess_targets = torch.clamp(feature_targets - floor_scaled, min=0.0)
    return event_targets, excess_targets


def set_backbone_trainable(model, trainable):
    for name, parameter in model.named_parameters():
        if (
            name.startswith("head.")
            or name.startswith("forecast_model.head.")
            or name.startswith("temporal_adapter.")
        ):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = trainable


def compute_split_points(length, train_ratio, val_ratio):
    train_end = max(1, int(length * train_ratio))
    val_end = max(train_end + 1, int(length * (train_ratio + val_ratio)))
    train_end = min(train_end, length)
    val_end = min(val_end, length)
    return train_end, val_end


def build_start_indices(window_count, max_windows=None):
    if window_count <= 0:
        return np.empty(0, dtype=np.int64)
    if max_windows is None or window_count <= max_windows:
        return np.arange(window_count, dtype=np.int64)
    return np.unique(np.linspace(0, window_count - 1, num=max_windows, dtype=np.int64))


def resize_sequence_length(array, target_len, mode="linear"):
    if array.shape[0] == target_len:
        return array.astype(np.float32, copy=False)
    if mode not in {"linear", "nearest"}:
        raise ValueError(f"Unsupported resize mode: {mode}")
    tensor = torch.as_tensor(array, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    if mode == "linear":
        resized = F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)
    else:
        resized = F.interpolate(tensor, size=target_len, mode="nearest")
    return resized.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32, copy=False)


def _normalize_timestamp_column(df):
    if "timestamp" in df.columns or len(df.columns) == 0:
        return df

    first_column = df.columns[0]
    normalized_name = str(first_column).strip().lower()
    if normalized_name in {"time", "date", "datetime"}:
        return df.rename(columns={first_column: "timestamp"})

    if normalized_name.startswith("unnamed") or normalized_name == "":
        sample = df[first_column].dropna().astype(str).head(5)
        if not sample.empty and pd.to_datetime(sample, errors="coerce").notna().all():
            return df.rename(columns={first_column: "timestamp"})
    return df


def _filter_time_range(frame, time_start=None, time_end=None):
    if time_start is not None:
        frame = frame[frame["timestamp"] >= pd.Timestamp(time_start)]
    if time_end is not None:
        frame = frame[frame["timestamp"] <= pd.Timestamp(time_end)]
    return frame


def _infer_timestamp_frequency(timestamps):
    diffs = pd.Series(timestamps).sort_values().drop_duplicates().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    modes = diffs.mode()
    if modes.empty:
        return None
    return modes.iloc[0]


def _reindex_expected_timestamps(frame, expected_freq=None):
    if frame.empty or len(frame) < 2:
        return frame

    freq = expected_freq or _infer_timestamp_frequency(frame["timestamp"])
    if freq is None:
        return frame

    full_index = pd.date_range(
        start=frame["timestamp"].iloc[0],
        end=frame["timestamp"].iloc[-1],
        freq=freq,
    )
    reindexed = frame.set_index("timestamp").reindex(full_index)
    reindexed.index.name = "timestamp"
    return reindexed.reset_index()


def load_water_frame(path, time_start=None, time_end=None, expected_freq=None, feature_columns=None):
    if not os.path.exists(path):
        return None

    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    df = pd.read_csv(path)
    df = _normalize_timestamp_column(df)

    required_columns = {"timestamp", *feature_columns}
    if not required_columns.issubset(df.columns):
        return None

    df = df[["timestamp", *feature_columns]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    df = _filter_time_range(df, time_start=time_start, time_end=time_end)
    if df.empty:
        return None

    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Newer exports may omit entire timestamps during maintenance instead of
    # leaving NaNs in place. Reindex first so these gaps are also flagged.
    df = _reindex_expected_timestamps(df, expected_freq=expected_freq)

    # Remaining NaNs in processed files correspond to longer gaps that we do not
    # want forecast windows to span across.
    df["__gap__"] = df[feature_columns].isna().any(axis=1)
    df[feature_columns] = df[feature_columns].interpolate(limit_direction="both")
    df = df.dropna(subset=feature_columns).reset_index(drop=True)
    return df


def resample_water_frame(frame, resolution, feature_columns=None):
    resolution = normalize_resolution(resolution)
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    if resolution == "4h":
        return frame.copy()
    if resolution not in RESOLUTION_TO_FREQ:
        raise ValueError(f"Unsupported resolution: {resolution}")

    freq = RESOLUTION_TO_FREQ[resolution]
    indexed = frame.set_index("timestamp")
    resample_kwargs = {"origin": "start_day"} if resolution == "4d" else {}
    resampled = (
        indexed[feature_columns]
        .resample(freq, **resample_kwargs)
        .mean()
        .interpolate(limit_direction="both")
        .dropna()
    )
    if "__gap__" in indexed.columns:
        gap_frame = (
            indexed["__gap__"]
            .astype(np.int8)
            .resample(freq, **resample_kwargs)
            .max()
            .reindex(resampled.index, fill_value=0)
        )
        resampled["__gap__"] = gap_frame.astype(bool)
    resampled = resampled.reset_index()
    return resampled


def list_station_names(data_root, resolution="4h"):
    station_names = set()
    for source_resolution in get_resolution_source_priority(resolution):
        folder = os.path.join(data_root, source_resolution)
        if not os.path.isdir(folder):
            continue
        station_names.update(
            os.path.splitext(filename)[0]
            for filename in os.listdir(folder)
            if filename.endswith(".csv") and not filename.startswith(".")
        )
    return sorted(station_names)


def _load_direct_station_frame(
    data_root,
    station_name,
    resolution,
    time_start=None,
    time_end=None,
    feature_columns=None,
):
    direct_path = os.path.join(data_root, resolution, f"{station_name}.csv")
    if not os.path.exists(direct_path):
        return None
    return load_water_frame(
        direct_path,
        time_start=time_start,
        time_end=time_end,
        expected_freq=RESOLUTION_TO_FREQ[resolution],
        feature_columns=feature_columns,
    )


def load_station_frame(
    data_root,
    station_name,
    resolution,
    time_start=None,
    time_end=None,
    feature_columns=None,
):
    resolution = normalize_resolution(resolution)
    for source_resolution in get_resolution_source_priority(resolution):
        frame = _load_direct_station_frame(
            data_root,
            station_name,
            source_resolution,
            time_start=time_start,
            time_end=time_end,
            feature_columns=feature_columns,
        )
        if frame is None:
            continue
        if source_resolution == resolution:
            return frame
        return resample_water_frame(frame, resolution, feature_columns=feature_columns)
    return None


def fit_scaler_on_train_slices(series_list, train_ratio):
    train_chunks = []
    for series in series_list:
        train_end = max(1, int(len(series) * train_ratio))
        train_chunks.append(series[:train_end])

    scaler = StandardScaler()
    scaler.fit(np.concatenate(train_chunks, axis=0))
    return scaler


def build_gap_aware_invalid_masks(invalid_mask, soft_gap_max_steps=None):
    raw_invalid_mask = np.asarray(invalid_mask, dtype=bool)
    input_invalid_mask = raw_invalid_mask.copy()

    gap_segment_count = 0
    soft_gap_segment_count = 0
    hard_gap_segment_count = 0
    soft_gap_steps = 0

    max_steps = None
    if soft_gap_max_steps is not None:
        max_steps = int(soft_gap_max_steps)
        if max_steps <= 0:
            max_steps = None

    index = 0
    while index < len(raw_invalid_mask):
        if not raw_invalid_mask[index]:
            index += 1
            continue

        start = index
        while index < len(raw_invalid_mask) and raw_invalid_mask[index]:
            index += 1
        end = index
        length = end - start
        gap_segment_count += 1

        if max_steps is not None and length <= max_steps:
            input_invalid_mask[start:end] = False
            soft_gap_segment_count += 1
            soft_gap_steps += length
        else:
            hard_gap_segment_count += 1

    return input_invalid_mask, raw_invalid_mask.copy(), {
        "soft_gap_max_steps": max_steps,
        "gap_segment_count": gap_segment_count,
        "soft_gap_segment_count": soft_gap_segment_count,
        "hard_gap_segment_count": hard_gap_segment_count,
        "gap_steps": int(raw_invalid_mask.sum()),
        "soft_gap_steps": soft_gap_steps,
        "hard_gap_steps": int(input_invalid_mask.sum()),
    }


def build_valid_window_starts(
    invalid_mask,
    raw_seq_len,
    raw_pred_len,
    policy="all",
    target_invalid_mask=None,
):
    window_count = len(invalid_mask) - raw_seq_len - raw_pred_len + 1
    if window_count <= 0:
        return np.empty(0, dtype=np.int64)

    if policy not in {"all", "input_only", "target_only"}:
        raise ValueError(f"Unsupported invalid window policy: {policy}")

    input_invalid_prefix = np.concatenate([[0], np.cumsum(np.asarray(invalid_mask, dtype=np.int64))])
    if target_invalid_mask is None:
        target_invalid_prefix = input_invalid_prefix
    else:
        if len(target_invalid_mask) != len(invalid_mask):
            raise ValueError("target_invalid_mask must have the same length as invalid_mask.")
        target_invalid_prefix = np.concatenate([[0], np.cumsum(np.asarray(target_invalid_mask, dtype=np.int64))])
    valid_starts = []
    for start in range(window_count):
        mid = start + raw_seq_len
        end = start + raw_seq_len + raw_pred_len
        if policy == "all":
            is_valid = (
                input_invalid_prefix[mid] == input_invalid_prefix[start]
                and target_invalid_prefix[end] == target_invalid_prefix[mid]
            )
        elif policy == "input_only":
            is_valid = input_invalid_prefix[mid] == input_invalid_prefix[start]
        else:
            is_valid = target_invalid_prefix[end] == target_invalid_prefix[mid]

        if is_valid:
            valid_starts.append(start)
    return np.asarray(valid_starts, dtype=np.int64)


class MaskedReconstructionDataset(Dataset):
    def __init__(
        self,
        series_list,
        raw_seq_len,
        model_seq_len,
        mask_ratio,
        split="train",
        train_ratio=0.8,
        max_windows_per_series=None,
        resize_mode="linear",
    ):
        self.series_list = []
        self.samples = []
        self.raw_seq_len = raw_seq_len
        self.model_seq_len = model_seq_len
        self.mask_ratio = mask_ratio
        self.resize_mode = resize_mode

        for series in series_list:
            train_end = max(1, int(len(series) * train_ratio))
            if split == "train":
                sliced = series[:train_end]
            else:
                sliced = series[max(0, train_end - raw_seq_len + 1):]

            window_count = len(sliced) - raw_seq_len + 1
            starts = build_start_indices(window_count, max_windows_per_series)
            if len(starts) == 0:
                continue

            series_idx = len(self.series_list)
            self.series_list.append(sliced.astype(np.float32, copy=False))
            self.samples.extend((series_idx, int(start)) for start in starts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        series_idx, start = self.samples[index]
        window = self.series_list[series_idx][start:start + self.raw_seq_len]
        window = resize_sequence_length(window, self.model_seq_len, mode=self.resize_mode)

        target = torch.as_tensor(window, dtype=torch.float32)
        mask = (torch.rand_like(target) > self.mask_ratio).float()
        masked_input = target * mask
        return masked_input, target


class ForecastWindowDataset(Dataset):
    def __init__(
        self,
        timestamps,
        raw_seq_len,
        raw_pred_len,
        model_seq_len,
        model_pred_len,
        values=None,
        input_values=None,
        target_values=None,
        invalid_mask=None,
        split="train",
        train_ratio=0.7,
        val_ratio=0.1,
        filter_invalid_windows=True,
        invalid_window_policy="all",
        resize_mode="linear",
        input_invalid_mask=None,
        target_invalid_mask=None,
    ):
        if input_invalid_mask is None:
            if invalid_mask is None:
                raise ValueError("Either invalid_mask or input_invalid_mask must be provided.")
            input_invalid_mask = invalid_mask
        if target_invalid_mask is None:
            if invalid_mask is None:
                raise ValueError("Either invalid_mask or target_invalid_mask must be provided.")
            target_invalid_mask = invalid_mask

        if input_values is None:
            if values is None:
                raise ValueError("Either values or input_values must be provided.")
            input_values = values
        if target_values is None:
            if values is None:
                raise ValueError("Either values or target_values must be provided.")
            target_values = values
        if len(input_values) != len(target_values) or len(target_values) != len(timestamps):
            raise ValueError("input_values, target_values, and timestamps must have the same length.")

        train_end, val_end = compute_split_points(len(target_values), train_ratio, val_ratio)

        if split == "train":
            sliced_input_values = input_values[:train_end]
            sliced_target_values = target_values[:train_end]
            sliced_timestamps = timestamps[:train_end]
            sliced_input_invalid_mask = input_invalid_mask[:train_end]
            sliced_target_invalid_mask = target_invalid_mask[:train_end]
        elif split == "val":
            start = max(0, train_end - raw_seq_len)
            sliced_input_values = input_values[start:val_end]
            sliced_target_values = target_values[start:val_end]
            sliced_timestamps = timestamps[start:val_end]
            sliced_input_invalid_mask = input_invalid_mask[start:val_end]
            sliced_target_invalid_mask = target_invalid_mask[start:val_end]
        else:
            start = max(0, val_end - raw_seq_len)
            sliced_input_values = input_values[start:]
            sliced_target_values = target_values[start:]
            sliced_timestamps = timestamps[start:]
            sliced_input_invalid_mask = input_invalid_mask[start:]
            sliced_target_invalid_mask = target_invalid_mask[start:]

        self.input_values = sliced_input_values.astype(np.float32, copy=False)
        self.target_values = sliced_target_values.astype(np.float32, copy=False)
        self.values = self.target_values
        self.timestamps = sliced_timestamps.astype("datetime64[ns]")
        self.input_invalid_mask = np.asarray(sliced_input_invalid_mask, dtype=bool)
        self.target_invalid_mask = np.asarray(sliced_target_invalid_mask, dtype=bool)
        self.invalid_mask = self.target_invalid_mask
        self.raw_seq_len = raw_seq_len
        self.raw_pred_len = raw_pred_len
        self.model_seq_len = model_seq_len
        self.model_pred_len = model_pred_len
        self.invalid_window_policy = invalid_window_policy
        self.resize_mode = resize_mode
        self.candidate_window_count = max(0, len(self.values) - raw_seq_len - raw_pred_len + 1)
        if filter_invalid_windows:
            self.window_starts = build_valid_window_starts(
                self.input_invalid_mask,
                raw_seq_len,
                raw_pred_len,
                policy=invalid_window_policy,
                target_invalid_mask=self.target_invalid_mask,
            )
        else:
            self.window_starts = np.arange(self.candidate_window_count, dtype=np.int64)
        self.window_count = len(self.window_starts)
        self.filtered_window_count = self.candidate_window_count - self.window_count

    def __len__(self):
        return self.window_count

    def __getitem__(self, index):
        start = int(self.window_starts[index])
        mid = start + self.raw_seq_len
        end = mid + self.raw_pred_len

        x = self.input_values[start:mid]
        y = self.target_values[mid:end]
        target_times = self.timestamps[mid:end].astype(np.int64)

        x = resize_sequence_length(x, self.model_seq_len, mode=self.resize_mode)
        y = resize_sequence_length(y, self.model_pred_len)

        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(target_times, dtype=torch.int64),
        )


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.embed = nn.Linear(seq_len, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-1)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class TemporalInputAdapter(nn.Module):
    def __init__(self, input_dim, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=input_dim,
        )
        nn.init.zeros_(self.depthwise.weight)
        nn.init.zeros_(self.depthwise.bias)

    def forward(self, x):
        return x + self.depthwise(x)


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        _, _, heads, embed_dim = queries.shape
        scale = self.scale or 1.0 / sqrt(embed_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", attn, values).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        batch_size, seq_len, _ = queries.shape
        heads = self.n_heads
        queries = self.query_projection(queries).view(batch_size, seq_len, heads, -1)
        keys = self.key_projection(keys).view(batch_size, seq_len, heads, -1)
        values = self.value_projection(values).view(batch_size, seq_len, heads, -1)
        out = self.inner_attention(queries, keys, values)
        out = out.view(batch_size, seq_len, -1)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x, x, x)))
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class WaterQualityTransformer(nn.Module):
    def __init__(
        self,
        num_heads,
        e_layer,
        hidden_size,
        input_dim,
        seq_len,
        pred_len,
        target_dim=None,
        target_feature_names=None,
        use_temporal_adapter=False,
        temporal_adapter_kernel_size=5,
        nh4n_two_stage_config=None,
    ):
        super().__init__()
        self.d_model = hidden_size
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.target_dim = input_dim if target_dim is None else target_dim
        if self.target_dim <= 0 or self.target_dim > self.input_dim:
            raise ValueError("target_dim must be in the range [1, input_dim].")
        self.target_feature_names = list(
            target_feature_names or FEATURE_COLUMNS[: self.target_dim]
        )
        if len(self.target_feature_names) != self.target_dim:
            raise ValueError("target_feature_names length must match target_dim.")
        self.nh4n_two_stage_config = nh4n_two_stage_config or {}
        self.nh4n_two_stage_enabled = bool(self.nh4n_two_stage_config.get("enabled"))
        default_feature_index = min(2, self.target_dim - 1)
        configured_feature_name = self.nh4n_two_stage_config.get("feature")
        if configured_feature_name in self.target_feature_names:
            default_feature_index = self.target_feature_names.index(configured_feature_name)
        elif "TP" in self.target_feature_names:
            default_feature_index = self.target_feature_names.index("TP")
        elif "NH4N" in self.target_feature_names:
            default_feature_index = self.target_feature_names.index("NH4N")
        self.nh4n_feature_index = int(
            self.nh4n_two_stage_config.get("feature_index", default_feature_index)
        )
        self.nh4n_floor_scaled = float(self.nh4n_two_stage_config.get("floor_scaled", 0.0))
        self.temporal_adapter = (
            TemporalInputAdapter(input_dim, kernel_size=temporal_adapter_kernel_size)
            if use_temporal_adapter
            else None
        )
        self.embedding = TimeFeatureEmbedding(hidden_size, seq_len)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(attention_dropout=0.1), hidden_size, num_heads),
                    hidden_size,
                    2048,
                    dropout=0.1,
                )
                for _ in range(e_layer)
            ],
            norm_layer=nn.LayerNorm(hidden_size),
        )
        self.head = FlattenHead(hidden_size, pred_len, head_dropout=0.1)
        if self.nh4n_two_stage_enabled:
            self.nh4n_event_head = nn.Linear(hidden_size, pred_len)
            self.nh4n_excess_head = nn.Linear(hidden_size, pred_len)
        else:
            self.nh4n_event_head = None
            self.nh4n_excess_head = None

    def forward_with_aux(self, x_enc):
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = centered / stdev

        normalized = normalized.permute(0, 2, 1)
        if self.temporal_adapter is not None:
            normalized = self.temporal_adapter(normalized)
        enc_out = self.embedding(normalized)
        enc_out = self.encoder(enc_out)
        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = dec_out[:, :, :self.target_dim]

        target_stdev = stdev[:, 0, :self.target_dim].unsqueeze(1).repeat(1, self.pred_len, 1)
        target_means = means[:, 0, :self.target_dim].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out * target_stdev
        dec_out = dec_out + target_means
        aux_outputs = {}
        if self.nh4n_two_stage_enabled:
            if self.nh4n_feature_index >= self.target_dim:
                raise ValueError("Two-stage feature index must be within target_dim.")
            nh4n_repr = enc_out[:, self.nh4n_feature_index, :]
            event_logits = self.nh4n_event_head(nh4n_repr).unsqueeze(-1)
            excess_pred = F.softplus(self.nh4n_excess_head(nh4n_repr)).unsqueeze(-1)
            nh4n_pred = self.nh4n_floor_scaled + (torch.sigmoid(event_logits) * excess_pred)
            dec_out = dec_out.clone()
            dec_out[:, :, self.nh4n_feature_index:self.nh4n_feature_index + 1] = nh4n_pred
            aux_outputs = {
                "nh4n_event_logits": event_logits,
                "nh4n_excess": excess_pred,
            }
        return dec_out, aux_outputs

    def forward(self, x_enc):
        preds, _ = self.forward_with_aux(x_enc)
        return preds


def _forward_model(model, x):
    if hasattr(model, "forward_with_aux"):
        return model.forward_with_aux(x)
    return model(x), {}


def _unpack_batch(batch):
    if len(batch) == 2:
        x, y = batch
        return x, y, None
    if len(batch) == 3:
        x, y, timestamps = batch
        return x, y, timestamps
    raise ValueError("Unexpected batch format.")


def evaluate_model(model, loader, device, scaler=None):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    timestamps = []

    with torch.no_grad():
        for batch in loader:
            x, y, target_times = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)

            output, _ = _forward_model(model, x)
            if output.shape != y.shape:
                y = y.view_as(output)

            loss = criterion(output, y)
            total_loss += loss.item()

            pred_np = output.cpu().numpy()
            target_np = y.cpu().numpy()

            if scaler is not None:
                pred_np = scaler.inverse_transform(pred_np)
                target_np = scaler.inverse_transform(target_np)

            preds.append(pred_np)
            targets.append(target_np)
            if target_times is not None:
                timestamps.append(target_times.cpu().numpy())

    if not preds:
        return float("inf"), np.empty((0, 0, 0)), np.empty((0, 0, 0)), np.empty((0, 0), dtype=np.int64)

    avg_loss = total_loss / max(1, len(loader))
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    timestamps = np.concatenate(timestamps, axis=0) if timestamps else np.empty((0, 0), dtype=np.int64)
    return avg_loss, preds, targets, timestamps


def fit_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    base_lr,
    epsilon,
    weight_decay,
    lr_milestones,
    lr_decay_ratio,
    max_grad_norm,
    log_prefix,
    loss_name="mse",
    nse_weight=0.0,
    loss_feature_weights=None,
    multitask_config=None,
    monitor_metric="loss",
    monitor_feature=None,
    monitor_feature_weights=None,
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
    scheduler_name="multistep",
    scheduler_patience=5,
    scheduler_min_lr=1e-6,
    freeze_backbone_epochs=0,
    feature_names=None,
):
    feature_names = list(feature_names or FEATURE_COLUMNS)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        eps=epsilon,
        weight_decay=weight_decay,
        amsgrad=True,
    )

    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if monitor_metric == "nse" else "min",
            factor=lr_decay_ratio,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )
    elif scheduler_name == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_milestones,
            gamma=lr_decay_ratio,
        )
    else:
        raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")

    history = []
    best_val_loss = float("inf")
    best_val_nse = float("-inf")
    best_val_monitor_nse = float("-inf") if (monitor_feature or monitor_feature_weights) else None
    best_epoch = 0
    best_state = snapshot_state_dict(model)
    best_monitor = float("-inf") if monitor_metric == "nse" else float("inf")
    patience_counter = 0
    backbone_frozen = freeze_backbone_epochs > 0
    if monitor_feature is not None and monitor_feature_weights is not None:
        raise ValueError("monitor_feature and monitor_feature_weights cannot be used together.")
    if monitor_feature is not None and monitor_feature not in feature_names:
        raise ValueError(f"Unknown monitor_feature: {monitor_feature}")
    normalized_feature_weights = _normalize_feature_weights(
        loss_feature_weights,
        feature_names=feature_names,
    )
    normalized_monitor_feature_weights = _normalize_feature_weights(
        monitor_feature_weights,
        feature_names=feature_names,
    )
    loss_feature_weights_tensor = (
        torch.as_tensor(normalized_feature_weights, device=device, dtype=torch.float32)
        if normalized_feature_weights is not None
        else None
    )

    if backbone_frozen:
        set_backbone_trainable(model, trainable=False)
        print(f"[{log_prefix}] Freeze backbone for the first {freeze_backbone_epochs} epochs.")

    for epoch in range(epochs):
        if backbone_frozen and epoch == freeze_backbone_epochs:
            set_backbone_trainable(model, trainable=True)
            backbone_frozen = False
            print(f"[{log_prefix}] Unfreeze backbone at epoch {epoch + 1}.")

        model.train()
        total_objective = 0.0
        total_mse = 0.0
        total_nse_penalty = 0.0
        total_event_bce = 0.0
        total_excess_mse = 0.0

        for batch in train_loader:
            x, y, _ = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output, aux_outputs = _forward_model(model, x)
            if output.shape != y.shape:
                y = y.view_as(output)

            loss, loss_components = compute_training_loss(
                output,
                y,
                loss_name=loss_name,
                nse_weight=nse_weight,
                feature_weights=loss_feature_weights_tensor,
                aux_outputs=aux_outputs,
                multitask_config=multitask_config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_objective += loss.item()
            total_mse += loss_components["mse"]
            total_nse_penalty += loss_components["nse_penalty"]
            total_event_bce += loss_components["event_bce"]
            total_excess_mse += loss_components["excess_mse"]

        avg_train_objective = total_objective / max(1, len(train_loader))
        avg_train_mse = total_mse / max(1, len(train_loader))
        avg_train_nse_penalty = total_nse_penalty / max(1, len(train_loader))
        avg_train_event_bce = total_event_bce / max(1, len(train_loader))
        avg_train_excess_mse = total_excess_mse / max(1, len(train_loader))
        val_loss, val_preds, val_targets, _ = evaluate_model(model, val_loader, device, scaler=None)
        val_metrics = compute_per_feature_metrics(val_preds, val_targets, feature_names)
        val_nse = val_metrics["__overall__"]["NSE"]
        if monitor_feature:
            val_monitor_nse = val_metrics[monitor_feature]["NSE"]
        elif normalized_monitor_feature_weights is not None:
            val_monitor_nse = compute_weighted_nse_from_metrics(
                val_metrics,
                normalized_monitor_feature_weights,
                feature_names=feature_names,
            )
        else:
            val_monitor_nse = val_nse
        current_lr = optimizer.param_groups[0]["lr"]

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_objective,
            "train_mse": avg_train_mse,
            "train_nse_penalty": avg_train_nse_penalty,
            "train_event_bce": avg_train_event_bce,
            "train_excess_mse": avg_train_excess_mse,
            "val_loss": val_loss,
            "val_nse": val_nse,
            "lr": current_lr,
        }
        if monitor_feature:
            history_entry[f"val_{monitor_feature}_nse"] = val_monitor_nse
        elif normalized_monitor_feature_weights is not None:
            history_entry["val_weighted_nse"] = val_monitor_nse
        history.append(history_entry)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            monitor_text = (
                f" | Val{monitor_feature}NSE: {val_monitor_nse:.6f}"
                if monitor_feature
                else (
                    f" | ValWeightedNSE: {val_monitor_nse:.6f}"
                    if normalized_monitor_feature_weights is not None
                    else ""
                )
            )
            print(
                f"[{log_prefix}] Epoch {epoch + 1}/{epochs} | "
                f"TrainObj: {avg_train_objective:.6f} | "
                f"ValLoss: {val_loss:.6f} | ValNSE: {val_nse:.6f}"
                f"{monitor_text} | LR: {current_lr:.2e}"
            )

        if np.isnan(avg_train_objective) or avg_train_objective > 1e6:
            print(f"[{log_prefix}] Loss diverged at epoch {epoch + 1}, stopping early.")
            break

        scheduler_value = val_monitor_nse if monitor_metric == "nse" else val_loss
        if scheduler_name == "plateau":
            if np.isfinite(scheduler_value):
                scheduler.step(scheduler_value)
        else:
            scheduler.step()

        current_monitor = scheduler_value
        if monitor_metric == "nse":
            improved = np.isfinite(current_monitor) and current_monitor > (best_monitor + early_stopping_min_delta)
        else:
            improved = current_monitor < (best_monitor - early_stopping_min_delta)

        if improved:
            best_monitor = current_monitor
            best_val_loss = val_loss
            best_val_nse = val_nse
            if monitor_feature or normalized_monitor_feature_weights is not None:
                best_val_monitor_nse = val_monitor_nse
            best_epoch = epoch + 1
            best_state = snapshot_state_dict(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            monitor_text = (
                f" | best_val_{monitor_feature}_nse={best_val_monitor_nse:.6f}"
                if monitor_feature and best_val_monitor_nse is not None
                else (
                    f" | best_val_weighted_nse={best_val_monitor_nse:.6f}"
                    if normalized_monitor_feature_weights is not None and best_val_monitor_nse is not None
                    else ""
                )
            )
            print(
                f"[{log_prefix}] Early stopping at epoch {epoch + 1} | "
                f"best_epoch={best_epoch} | best_val_loss={best_val_loss:.6f} "
                f"| best_val_nse={best_val_nse:.6f}{monitor_text}"
            )
            break

    model.load_state_dict(best_state)
    return model, history, {
        "epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_nse": best_val_nse,
        "val_monitor_nse": (
            best_val_monitor_nse
            if (monitor_feature or normalized_monitor_feature_weights is not None)
            else best_val_nse
        ),
        "monitor_metric": monitor_metric,
        "monitor_feature": monitor_feature,
        "monitor_feature_weights": (
            normalized_monitor_feature_weights.tolist()
            if normalized_monitor_feature_weights is not None
            else None
        ),
        "monitor_value": best_monitor,
    }


def _flatten_preds_targets(preds, targets):
    preds_2d = preds.reshape(-1, preds.shape[-1])
    targets_2d = targets.reshape(-1, targets.shape[-1])
    return preds_2d, targets_2d


def compute_per_feature_metrics(preds, targets, feature_names=None):
    feature_names = feature_names or FEATURE_COLUMNS
    preds_2d, targets_2d = _flatten_preds_targets(preds, targets)
    metrics = {}

    for index, name in enumerate(feature_names):
        y_pred = preds_2d[:, index]
        y_true = targets_2d[:, index]

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        numerator = np.sum((y_true - y_pred) ** 2)
        nse_defined = denominator > MIN_NSE_DENOMINATOR
        nse = float(1.0 - numerator / denominator) if nse_defined else float("nan")

        mask = y_true != 0
        mape = (
            float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            if np.any(mask)
            else float("nan")
        )

        metrics[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "NSE": nse,
            "MAPE": mape,
            "NSE_defined": bool(nse_defined),
        }

    overall_mse = float(np.mean((preds_2d - targets_2d) ** 2))
    overall_mae = float(np.mean(np.abs(preds_2d - targets_2d)))
    rmse_per_feature = [
        np.sqrt(np.mean((preds_2d[:, idx] - targets_2d[:, idx]) ** 2))
        for idx in range(preds_2d.shape[1])
    ]

    nse_per_feature = []
    for idx in range(preds_2d.shape[1]):
        y_true = targets_2d[:, idx]
        y_pred = preds_2d[:, idx]
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        numerator = np.sum((y_true - y_pred) ** 2)
        if denominator > MIN_NSE_DENOMINATOR:
            nse_per_feature.append(1.0 - numerator / denominator)

    mask = targets_2d != 0
    overall_mape = (
        float(np.mean(np.abs((targets_2d[mask] - preds_2d[mask]) / targets_2d[mask])) * 100)
        if np.any(mask)
        else float("nan")
    )

    metrics["__overall__"] = {
        "MSE": overall_mse,
        "MAE": overall_mae,
        "RMSE": float(np.mean(rmse_per_feature)),
        "NSE": float(np.mean(nse_per_feature)) if nse_per_feature else float("nan"),
        "MAPE": overall_mape,
        "valid_nse_feature_count": int(len(nse_per_feature)),
        "total_feature_count": int(preds_2d.shape[1]),
    }
    return metrics
