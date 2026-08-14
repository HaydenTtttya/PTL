import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from finetune import (
    FinetuneConfig,
    build_model,
    build_stage_nh4n_two_stage_config,
    build_stage_specs,
    load_stage_data,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_DIR = (
    REPO_ROOT
    / "results/ptl/finetune/runs/shap_weights_17stations_strict_2023_2024_gpu"
    / "batch_pearl_other_core3_progressive_v2pretrain_v2_2021_2024_20260519_165126"
)
DEFAULT_SUMMARY_DIR = (
    REPO_ROOT
    / "results/summary/current_all_tested_stations_overall_nse"
    / "均衡十五站方案_新增两站"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_DIR / "SHAP分析"
DEFAULT_TARGETS = ("CODMn", "DO", "NH4N", "pH")
FOCUS_TARGETS = ("CODMn", "DO", "pH")


class ScalarTargetWrapper(torch.nn.Module):
    def __init__(self, model, target_index):
        super().__init__()
        self.model = model
        self.target_index = int(target_index)

    def forward(self, x):
        return self.model(x)[:, 0, self.target_index]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Gradient SHAP/expected-gradients analysis for the 17-station PTL stage3 models.",
    )
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--background-size", type=int, default=64)
    parser.add_argument("--explain-size", type=int, default=100)
    parser.add_argument("--method", choices=("permutation", "expected_gradients"), default="permutation")
    parser.add_argument("--permutation-repeats", type=int, default=1)
    parser.add_argument("--permutation-batch-size", type=int, default=2048)
    parser.add_argument("--model-batch-size", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--shap-batch-size", type=int, default=8)
    parser.add_argument("--sample-chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--station", action="append", default=[])
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is false.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
    return torch.device(name)


def make_runtime_config(custom_config, device):
    config = FinetuneConfig()
    for key, value in custom_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.feature_columns = list(config.feature_columns)
    config.input_dim = len(config.feature_columns)
    config.device = device
    return config


def load_run_setup(batch_dir):
    setup_path = batch_dir / "run_setup.json"
    if not setup_path.exists():
        raise FileNotFoundError(f"Missing run_setup.json: {setup_path}")
    return json.loads(setup_path.read_text(encoding="utf-8"))


def find_stage_dir(batch_dir, station):
    matches = sorted(batch_dir.glob(f"progressive_{station}_seed*/stage3_daily"))
    if not matches:
        raise FileNotFoundError(f"Missing stage3_daily directory for station: {station}")
    if len(matches) > 1:
        return matches[-1]
    return matches[0]


def deterministic_indices(count, sample_count):
    if sample_count <= 0 or sample_count >= count:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, sample_count, dtype=np.int64))


def stack_dataset(dataset, indices):
    xs = []
    ys = []
    times = []
    for index in indices:
        x, y, target_times = dataset[int(index)]
        xs.append(x)
        ys.append(y)
        times.append(target_times)
    return torch.stack(xs), torch.stack(ys), torch.stack(times)


def tensor_outputs(wrapper, x_cpu, device, batch_size=128):
    outputs = []
    wrapper.eval()
    with torch.no_grad():
        for start in range(0, len(x_cpu), batch_size):
            out = wrapper(x_cpu[start : start + batch_size].to(device))
            outputs.append(out.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def predict_all_targets(model, flat_values, seq_len, input_dim, device, model_batch_size):
    values = np.asarray(flat_values, dtype=np.float32).reshape(-1, seq_len, input_dim)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), model_batch_size):
            x = torch.as_tensor(
                values[start : start + model_batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model(x)[:, 0, :].detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def permutation_shap_all_targets(
    model,
    inputs_cpu,
    background_cpu,
    *,
    target_indices,
    device,
    repeats,
    model_batch_size,
    seed,
):
    sample_count, seq_len, input_dim = inputs_cpu.shape
    flat_inputs = inputs_cpu.detach().cpu().numpy().reshape(sample_count, -1)
    flat_background = background_cpu.detach().cpu().numpy().reshape(len(background_cpu), -1)
    feature_count = flat_inputs.shape[1]
    target_indices = list(target_indices)
    target_count = len(target_indices)
    repeats = max(1, int(repeats))
    rng = np.random.default_rng(int(seed))

    attributions = np.zeros((target_count, sample_count, feature_count), dtype=np.float32)
    base_outputs = predict_all_targets(
        model,
        flat_background,
        seq_len,
        input_dim,
        device,
        model_batch_size,
    )[:, target_indices].mean(axis=0)
    predictions = predict_all_targets(
        model,
        flat_inputs,
        seq_len,
        input_dim,
        device,
        model_batch_size,
    )[:, target_indices]

    for repeat_index in range(repeats):
        permutation = rng.permutation(feature_count)
        current = np.broadcast_to(
            flat_background[None, :, :],
            (sample_count, len(flat_background), feature_count),
        ).copy()
        previous = predict_all_targets(
            model,
            current.reshape(-1, feature_count),
            seq_len,
            input_dim,
            device,
            model_batch_size,
        )[:, target_indices].reshape(sample_count, len(flat_background), target_count).mean(axis=1)
        for feature_index in permutation:
            current[:, :, feature_index] = flat_inputs[:, None, feature_index]
            next_output = predict_all_targets(
                model,
                current.reshape(-1, feature_count),
                seq_len,
                input_dim,
                device,
                model_batch_size,
            )[:, target_indices].reshape(sample_count, len(flat_background), target_count).mean(axis=1)
            delta = next_output - previous
            attributions[:, :, feature_index] += delta.T.astype(np.float32)
            previous = next_output

    attributions /= float(repeats)
    additivity_errors = np.abs(attributions.sum(axis=2).T + base_outputs[None, :] - predictions)
    attributions = attributions.reshape(target_count, sample_count, seq_len, input_dim)
    base_values = np.broadcast_to(base_outputs[None, :], (sample_count, target_count)).copy()
    max_evals = repeats * (feature_count + 2)
    return attributions, base_values, predictions, additivity_errors, max_evals


def expected_gradients_shap(
    wrapper,
    inputs_cpu,
    background_cpu,
    *,
    nsamples,
    batch_size,
    sample_chunk_size,
    device,
    seed,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    wrapper.eval()

    background_device = background_cpu.to(device)
    n_samples, seq_len, input_dim = inputs_cpu.shape
    background_count = background_cpu.shape[0]
    attributions = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)

    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        x = inputs_cpu[start:stop].to(device)
        current_batch = x.shape[0]
        attr_sum = torch.zeros_like(x)
        done = 0
        while done < nsamples:
            chunk = min(sample_chunk_size, nsamples - done)
            bg_idx = torch.randint(
                0,
                background_count,
                (current_batch, chunk),
                generator=generator,
                device="cpu",
            ).to(device)
            alpha = torch.rand(
                (current_batch, chunk, 1, 1),
                generator=generator,
                device="cpu",
            ).to(device)
            bg = background_device[bg_idx.reshape(-1)].reshape(
                current_batch,
                chunk,
                seq_len,
                input_dim,
            )
            x_rep = x[:, None, :, :].expand(-1, chunk, -1, -1)
            interp = bg + alpha * (x_rep - bg)
            interp = interp.reshape(current_batch * chunk, seq_len, input_dim)
            interp.requires_grad_(True)
            out = wrapper(interp).sum()
            grads = torch.autograd.grad(out, interp, retain_graph=False, create_graph=False)[0]
            contrib = (x_rep.reshape_as(grads) - bg.reshape_as(grads)) * grads
            contrib = contrib.reshape(current_batch, chunk, seq_len, input_dim).sum(dim=1)
            attr_sum += contrib.detach()
            done += chunk
        attributions[start:stop] = (attr_sum / float(nsamples)).detach().cpu().numpy()

    return attributions


def to_datetime_strings(target_times):
    values = target_times[:, 0].detach().cpu().numpy().astype("datetime64[ns]")
    return pd.to_datetime(values).astype(str).to_numpy()


def inverse_input_values(input_scaler, x_cpu):
    values = x_cpu.detach().cpu().numpy()
    flat = values.reshape(-1, values.shape[-1])
    raw = input_scaler.inverse_transform(flat).reshape(values.shape)
    return raw


def add_station_rows(
    *,
    local_rows,
    station,
    target_name,
    target_index,
    timestamps,
    x_scaled,
    x_raw,
    shap_scaled,
    scaler,
    input_features,
):
    target_std = float(scaler.std[target_index] + 1e-8)
    shap_raw = shap_scaled * target_std
    sample_count, seq_len, input_dim = shap_scaled.shape
    lag_days = np.arange(seq_len, 0, -1, dtype=np.int64)

    row_frames = []
    for feature_index, feature_name in enumerate(input_features):
        for time_index in range(seq_len):
            frame = pd.DataFrame(
                {
                    "station": station,
                    "target": target_name,
                    "sample_index": np.arange(sample_count, dtype=np.int64),
                    "target_timestamp": timestamps,
                    "input_feature": feature_name,
                    "time_index": time_index,
                    "lag_day": int(lag_days[time_index]),
                    "lag_label": f"t-{int(lag_days[time_index])}",
                    "input_value_scaled": x_scaled[:, time_index, feature_index],
                    "input_value_raw": x_raw[:, time_index, feature_index],
                    "shap_scaled": shap_scaled[:, time_index, feature_index],
                    "shap_raw_unit": shap_raw[:, time_index, feature_index],
                    "abs_shap_scaled": np.abs(shap_scaled[:, time_index, feature_index]),
                    "abs_shap_raw_unit": np.abs(shap_raw[:, time_index, feature_index]),
                }
            )
            row_frames.append(frame)
    local_rows.append(pd.concat(row_frames, ignore_index=True))


def build_summary_tables(local_frame, station_class):
    feature_lag = (
        local_frame.groupby(["station", "target", "input_feature", "time_index", "lag_day", "lag_label"], as_index=False)
        .agg(
            mean_abs_shap_scaled=("abs_shap_scaled", "mean"),
            mean_shap_scaled=("shap_scaled", "mean"),
            mean_abs_shap_raw_unit=("abs_shap_raw_unit", "mean"),
            mean_shap_raw_unit=("shap_raw_unit", "mean"),
            sample_count=("sample_index", "nunique"),
        )
    )
    feature_importance = (
        feature_lag.groupby(["station", "target", "input_feature"], as_index=False)
        .agg(
            mean_abs_shap_scaled=("mean_abs_shap_scaled", "mean"),
            mean_shap_scaled=("mean_shap_scaled", "mean"),
            mean_abs_shap_raw_unit=("mean_abs_shap_raw_unit", "mean"),
            mean_shap_raw_unit=("mean_shap_raw_unit", "mean"),
        )
    )
    feature_importance["rank_within_station_target"] = (
        feature_importance.groupby(["station", "target"])["mean_abs_shap_scaled"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    station_target = (
        feature_importance.groupby(["station", "target"], as_index=False)
        .agg(total_mean_abs_shap_scaled=("mean_abs_shap_scaled", "sum"))
    )

    global_feature = (
        feature_importance.groupby(["target", "input_feature"], as_index=False)
        .agg(
            station_equal_mean_abs_shap_scaled=("mean_abs_shap_scaled", "mean"),
            station_equal_mean_shap_scaled=("mean_shap_scaled", "mean"),
            station_count=("station", "nunique"),
        )
    )
    global_feature["rank_within_target"] = (
        global_feature.groupby("target")["station_equal_mean_abs_shap_scaled"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    global_lag = (
        feature_lag.groupby(["target", "lag_day", "lag_label"], as_index=False)
        .agg(station_equal_mean_abs_shap_scaled=("mean_abs_shap_scaled", "mean"))
    )
    global_feature_lag = (
        feature_lag.groupby(["target", "input_feature", "time_index", "lag_day", "lag_label"], as_index=False)
        .agg(
            station_equal_mean_abs_shap_scaled=("mean_abs_shap_scaled", "mean"),
            station_equal_mean_shap_scaled=("mean_shap_scaled", "mean"),
            station_count=("station", "nunique"),
        )
    )

    group_feature = pd.DataFrame()
    if station_class is not None:
        group_source = feature_importance.merge(station_class, on="station", how="left")
        group_feature = (
            group_source.groupby(["river_reach", "river_type", "target", "input_feature"], dropna=False, as_index=False)
            .agg(
                station_equal_mean_abs_shap_scaled=("mean_abs_shap_scaled", "mean"),
                station_count=("station", "nunique"),
            )
        )

    focus_station_feature = (
        feature_importance[feature_importance["target"].isin(FOCUS_TARGETS)]
        .groupby(["station", "input_feature"], as_index=False)
        .agg(focus_mean_abs_shap_scaled=("mean_abs_shap_scaled", "mean"))
    )

    return {
        "local": local_frame,
        "feature_lag": feature_lag,
        "feature_importance": feature_importance,
        "station_target": station_target,
        "global_feature": global_feature,
        "global_lag": global_lag,
        "global_feature_lag": global_feature_lag,
        "group_feature": group_feature,
        "focus_station_feature": focus_station_feature,
    }


def setup_plot_style():
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti TC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def save_heatmaps(global_feature_lag, output_dir):
    for target, frame in global_feature_lag.groupby("target"):
        pivot = frame.pivot_table(
            index="input_feature",
            columns="lag_day",
            values="station_equal_mean_abs_shap_scaled",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=DEFAULT_TARGETS)
        pivot = pivot.reindex(columns=sorted(pivot.columns, reverse=True))

        fig, ax = plt.subplots(figsize=(9, 3.8))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
        ax.set_title(f"Global SHAP importance heatmap: target {target}")
        ax.set_xlabel("Lag day before prediction")
        ax.set_ylabel("Input feature")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"t-{int(col)}" for col in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(im, ax=ax, label="Mean |SHAP| (scaled output)")
        fig.tight_layout()
        fig.savefig(output_dir / f"fig_feature_lag_heatmap_{target}.png")
        plt.close(fig)


def save_global_feature_plot(global_feature, output_dir):
    targets = [target for target in DEFAULT_TARGETS if target in set(global_feature["target"])]
    features = [feature for feature in DEFAULT_TARGETS if feature in set(global_feature["input_feature"])]
    x = np.arange(len(features))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, target in enumerate(targets):
        values = (
            global_feature[global_feature["target"] == target]
            .set_index("input_feature")
            .reindex(features)["station_equal_mean_abs_shap_scaled"]
            .fillna(0.0)
            .to_numpy()
        )
        ax.bar(x + (idx - (len(targets) - 1) / 2) * width, values, width=width, label=target)
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.set_ylabel("Station-equal mean |SHAP| (scaled output)")
    ax.set_title("Global input-feature importance by predicted target")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_global_feature_importance_by_target.png")
    plt.close(fig)


def save_station_focus_heatmap(focus_station_feature, station_order, output_dir):
    pivot = focus_station_feature.pivot_table(
        index="station",
        columns="input_feature",
        values="focus_mean_abs_shap_scaled",
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=station_order, columns=list(DEFAULT_TARGETS))
    fig_height = max(5, 0.36 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="PuBuGn")
    ax.set_title("Station heterogeneity of Focus-target SHAP importance")
    ax.set_xlabel("Input feature")
    ax.set_ylabel("Station")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean |SHAP| across CODMn/DO/pH")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_station_focus_feature_importance_heatmap.png")
    plt.close(fig)


def save_group_focus_plot(group_feature, output_dir):
    if group_feature.empty:
        return
    focus = group_feature[group_feature["target"].isin(FOCUS_TARGETS)].copy()
    if focus.empty:
        return
    focus["group"] = focus["river_reach"].fillna("NA") + " | " + focus["river_type"].fillna("NA")
    frame = (
        focus.groupby(["group", "input_feature"], as_index=False)
        .agg(value=("station_equal_mean_abs_shap_scaled", "mean"))
    )
    pivot = frame.pivot_table(index="group", columns="input_feature", values="value", aggfunc="mean")
    pivot = pivot.reindex(columns=list(DEFAULT_TARGETS))
    fig_height = max(4, 0.55 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="GnBu")
    ax.set_title("Grouped Focus-target SHAP importance")
    ax.set_xlabel("Input feature")
    ax.set_ylabel("River group")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_group_focus_feature_importance_heatmap.png")
    plt.close(fig)


def save_additivity_plot(additivity_frame, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    frame = additivity_frame.copy()
    frame["label"] = frame["station"] + " / " + frame["target"]
    ax.scatter(frame["mean_abs_output_delta_scaled"], frame["mean_abs_additivity_error_scaled"], s=20)
    ax.set_xlabel("Mean |f(x)-E[f(background)]|")
    ax.set_ylabel("Mean |sum(SHAP)-output delta|")
    ax.set_title("Gradient SHAP additivity approximation check")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_additivity_check.png")
    plt.close(fig)


def write_readme(output_dir, args, device, additivity_frame, tables):
    global_top = (
        tables["global_feature"].sort_values(["target", "rank_within_target"])
        .groupby("target")
        .head(4)[["target", "input_feature", "station_equal_mean_abs_shap_scaled", "rank_within_target"]]
    )
    global_top_text = global_top.to_csv(index=False).strip()
    lines = [
        "# PTL SHAP analysis",
        "",
        f"- Batch dir: `{args.batch_dir}`",
        f"- Device: `{device}`",
        f"- Background windows per station: `{args.background_size}`",
        f"- Explained test windows per station: `{args.explain_size}`",
        f"- Method: `{args.method}`",
        f"- Permutation repeats: `{args.permutation_repeats}`",
        f"- Expected-gradient samples per explained window: `{args.nsamples}`",
        "- Primary method: model-agnostic permutation SHAP for scalar one-step PTL outputs.",
        "- Fallback method: Gradient SHAP / expected gradients, retained for sensitivity checks.",
        "- Attribution unit: primary tables use scaled target output; `*_raw_unit` columns multiply by the target scaler std.",
        "",
        "## Additivity check",
        "",
        f"- Mean absolute additivity error: `{additivity_frame['mean_abs_additivity_error_scaled'].mean():.6f}`",
        f"- Max mean absolute additivity error: `{additivity_frame['mean_abs_additivity_error_scaled'].max():.6f}`",
        "",
        "## Top global feature importance by target",
        "",
        "```csv",
        global_top_text,
        "```",
        "",
    ]
    (output_dir / "SHAP分析说明.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    run_setup = load_run_setup(args.batch_dir)
    config = make_runtime_config(run_setup["custom_config"], device)
    stage_specs = build_stage_specs(config)
    stage3 = next(stage for stage in stage_specs if stage["name"] == "stage3_daily")
    station_order = list(run_setup["selected_station_names"])
    if args.station:
        selected = list(dict.fromkeys(args.station))
        station_order = [station for station in station_order if station in selected]
    if args.max_stations is not None:
        station_order = station_order[: args.max_stations]

    targets = list(dict.fromkeys(args.target or DEFAULT_TARGETS))
    missing_targets = [target for target in targets if target not in config.feature_columns]
    if missing_targets:
        raise ValueError(f"Unknown target(s): {missing_targets}")

    station_class = None
    class_path = args.summary_dir / "站点分类.csv"
    if class_path.exists():
        station_class = pd.read_csv(class_path, encoding="utf-8-sig")
        keep_cols = [col for col in ["station", "river_reach", "river_type", "comparison_group"] if col in station_class.columns]
        station_class = station_class[keep_cols].copy()

    local_rows = []
    additivity_rows = []
    run_rows = []

    for station_position, station in enumerate(station_order, start=1):
        stage_dir = find_stage_dir(args.batch_dir, station)
        model_path = stage_dir / "model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model weights: {model_path}")

        print(f"[{station_position}/{len(station_order)}] {station} | loading data", flush=True)
        stage_data = load_stage_data(config, station, stage3)
        if stage_data is None:
            raise RuntimeError(f"Unable to load stage3 data for station: {station}")

        nh4n_config = build_stage_nh4n_two_stage_config(stage3, stage_data["scaler"], config.feature_columns)
        model = build_model(
            config,
            {**stage3, "input_dim": stage_data["input_dim"]},
            nh4n_two_stage_config=nh4n_config,
        )
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        train_dataset = stage_data["train_loader"].dataset
        test_dataset = stage_data["test_loader"].dataset
        bg_indices = deterministic_indices(len(train_dataset), args.background_size)
        exp_indices = deterministic_indices(len(test_dataset), args.explain_size)
        x_bg, _, _ = stack_dataset(train_dataset, bg_indices)
        x_exp, y_exp, target_times = stack_dataset(test_dataset, exp_indices)
        timestamps = to_datetime_strings(target_times)
        x_raw = inverse_input_values(stage_data["input_scaler"], x_exp)
        x_scaled_np = x_exp.numpy()
        input_features = list(stage_data["input_feature_columns"])

        run_rows.append(
            {
                "station": station,
                "background_windows": len(x_bg),
                "explained_windows": len(x_exp),
                "train_windows": len(train_dataset),
                "test_windows": len(test_dataset),
                "input_dim": stage_data["input_dim"],
                "model_path": str(model_path),
            }
        )

        if args.method == "permutation":
            target_indices = [config.feature_columns.index(target_name) for target_name in targets]
            print(
                f"  permutation targets={','.join(targets)} | background={len(x_bg)} | explain={len(x_exp)} | repeats={args.permutation_repeats}",
                flush=True,
            )
            station_shap, base_values, exp_outputs, additivity_errors, max_evals = permutation_shap_all_targets(
                model,
                x_exp,
                x_bg,
                target_indices=target_indices,
                device=device,
                repeats=args.permutation_repeats,
                model_batch_size=args.model_batch_size,
                seed=args.seed + station_position * 1000,
            )
            for local_target_index, target_name in enumerate(targets):
                target_index = target_indices[local_target_index]
                shap_scaled = station_shap[local_target_index]
                bg_output = float(base_values[:, local_target_index].mean())
                exp_output = exp_outputs[:, local_target_index]
                output_delta = exp_output - base_values[:, local_target_index]
                additivity_error_values = additivity_errors[:, local_target_index]
                method_detail = f"max_evals={max_evals};repeats={args.permutation_repeats}"
                additivity_rows.append(
                    {
                        "station": station,
                        "target": target_name,
                        "method": args.method,
                        "method_detail": method_detail,
                        "background_expected_output_scaled": bg_output,
                        "mean_output_scaled": float(exp_output.mean()),
                        "mean_abs_output_delta_scaled": float(np.abs(output_delta).mean()),
                        "mean_abs_additivity_error_scaled": float(additivity_error_values.mean()),
                        "max_abs_additivity_error_scaled": float(additivity_error_values.max()),
                    }
                )
                add_station_rows(
                    local_rows=local_rows,
                    station=station,
                    target_name=target_name,
                    target_index=target_index,
                    timestamps=timestamps,
                    x_scaled=x_scaled_np,
                    x_raw=x_raw,
                    shap_scaled=shap_scaled,
                    scaler=stage_data["scaler"],
                    input_features=input_features,
                )
            continue

        for target_name in targets:
            target_index = config.feature_columns.index(target_name)
            wrapper = ScalarTargetWrapper(model, target_index).to(device)
            print(
                f"  target={target_name} | method={args.method} | background={len(x_bg)} | explain={len(x_exp)}",
                flush=True,
            )
            if args.method == "expected_gradients":
                shap_scaled = expected_gradients_shap(
                    wrapper,
                    x_exp,
                    x_bg,
                    nsamples=args.nsamples,
                    batch_size=args.shap_batch_size,
                    sample_chunk_size=args.sample_chunk_size,
                    device=device,
                    seed=args.seed + station_position * 100 + target_index,
                )
                bg_output = float(tensor_outputs(wrapper, x_bg, device, batch_size=args.model_batch_size).mean())
                exp_output = tensor_outputs(wrapper, x_exp, device, batch_size=args.model_batch_size)
                output_delta = exp_output - bg_output
                shap_sum = shap_scaled.sum(axis=(1, 2))
                additivity_error_values = np.abs(shap_sum - output_delta)
                method_detail = f"nsamples={args.nsamples}"
            else:
                raise ValueError(f"Unsupported method: {args.method}")
            additivity_rows.append(
                {
                    "station": station,
                    "target": target_name,
                    "method": args.method,
                    "method_detail": method_detail,
                    "background_expected_output_scaled": bg_output,
                    "mean_output_scaled": float(exp_output.mean()),
                    "mean_abs_output_delta_scaled": float(np.abs(output_delta).mean()),
                    "mean_abs_additivity_error_scaled": float(additivity_error_values.mean()),
                    "max_abs_additivity_error_scaled": float(additivity_error_values.max()),
                }
            )
            add_station_rows(
                local_rows=local_rows,
                station=station,
                target_name=target_name,
                target_index=target_index,
                timestamps=timestamps,
                x_scaled=x_scaled_np,
                x_raw=x_raw,
                shap_scaled=shap_scaled,
                scaler=stage_data["scaler"],
                input_features=input_features,
            )

    local_frame = pd.concat(local_rows, ignore_index=True)
    additivity_frame = pd.DataFrame(additivity_rows)
    run_frame = pd.DataFrame(run_rows)
    tables = build_summary_tables(local_frame, station_class)

    output_paths = {
        "local": args.output_dir / "shap_values_long.csv",
        "feature_lag": args.output_dir / "shap_feature_lag_importance.csv",
        "feature_importance": args.output_dir / "shap_feature_importance_by_station_target.csv",
        "station_target": args.output_dir / "shap_station_target_summary.csv",
        "global_feature": args.output_dir / "shap_global_feature_importance.csv",
        "global_lag": args.output_dir / "shap_global_lag_importance.csv",
        "global_feature_lag": args.output_dir / "shap_global_feature_lag_heatmap.csv",
        "group_feature": args.output_dir / "shap_group_feature_importance.csv",
        "focus_station_feature": args.output_dir / "shap_station_focus_feature_importance.csv",
    }
    for key, path in output_paths.items():
        tables[key].to_csv(path, index=False, encoding="utf-8-sig")
    additivity_frame.to_csv(args.output_dir / "shap_additivity_check.csv", index=False, encoding="utf-8-sig")
    run_frame.to_csv(args.output_dir / "shap_run_manifest.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(args.output_dir / "SHAP分析汇总.xlsx") as writer:
        tables["global_feature"].to_excel(writer, sheet_name="global_feature", index=False)
        tables["global_feature_lag"].to_excel(writer, sheet_name="global_feature_lag", index=False)
        tables["feature_importance"].to_excel(writer, sheet_name="station_feature", index=False)
        tables["group_feature"].to_excel(writer, sheet_name="group_feature", index=False)
        additivity_frame.to_excel(writer, sheet_name="additivity", index=False)
        run_frame.to_excel(writer, sheet_name="run_manifest", index=False)

    save_global_feature_plot(tables["global_feature"], args.output_dir)
    save_heatmaps(tables["global_feature_lag"], args.output_dir)
    save_station_focus_heatmap(tables["focus_station_feature"], station_order, args.output_dir)
    save_group_focus_plot(tables["group_feature"], args.output_dir)
    save_additivity_plot(additivity_frame, args.output_dir)
    write_readme(args.output_dir, args, device, additivity_frame, tables)

    print("=" * 80)
    print(f"SHAP analysis complete: {args.output_dir}")
    print(f"Local attribution rows: {len(local_frame)}")
    print(f"Mean additivity error: {additivity_frame['mean_abs_additivity_error_scaled'].mean():.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
