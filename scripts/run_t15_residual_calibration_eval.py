from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptation import source_to_target_moment_match
from src.decoder_eval import (
    add_official_model_training_to_path,
    greedy_ctc_decode,
    load_official_gru_decoder,
    logits_quality_metrics,
    phoneme_error_rate,
    phoneme_ids_to_string,
    resolve_device,
    trim_target_sequence,
)
from run_t15_input_layer_calibration_eval import (
    adjusted_ctc_lengths,
    compute_stats_from_trials,
    forward_custom_layer_logits,
    forward_fixed_layer_logits,
    load_geometry_source_map,
    pad_features,
    pad_labels,
    run_gru_backbone,
    smooth_inputs,
)


METHODS = [
    "native-day",
    "none",
    "moment_match_to_source",
    "feature_diagonal",
    "layer_bias",
    "layer_low_rank",
    "layer_full",
]


@dataclass
class AdapterState:
    method: str
    initial_loss: float
    final_loss: float
    payload: dict[str, np.ndarray]


def resolve_requested_device(device_name: str, gpu_number: int) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return resolve_device(gpu_number if gpu_number >= 0 else 0)
    if torch.cuda.is_available() and gpu_number >= 0:
        return resolve_device(gpu_number)
    return torch.device("cpu")


def _as_plain_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        value = value.tolist()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def _set_use_amp(model_args, enabled: bool) -> None:
    try:
        model_args["use_amp"] = enabled
    except TypeError:
        model_args.use_amp = enabled


def weighted_per(trials: pd.DataFrame) -> float:
    return float(trials["edit_distance"].sum() / trials["num_phonemes"].sum())


def compute_session_stats(file_path: Path) -> tuple[np.ndarray, np.ndarray, int] | None:
    if not file_path.exists():
        return None
    features = []
    with h5py.File(file_path, "r") as handle:
        for trial_id in handle.keys():
            features.append(handle[trial_id]["input_features"][:])
    return compute_stats_from_trials(features)


def forward_feature_diagonal_logits(
    x: torch.Tensor,
    raw_lengths: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    source_input_layer: int,
    model,
    model_args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return forward_fixed_layer_logits(x * scale.view(1, 1, -1) + bias.view(1, 1, -1), raw_lengths, source_input_layer, model, model_args, device)


def forward_low_rank_logits(
    x: torch.Tensor,
    raw_lengths: torch.Tensor,
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    bias_delta: torch.Tensor,
    model,
    model_args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = source_weight + left @ right
    bias = source_bias + bias_delta
    return forward_custom_layer_logits(x, raw_lengths, weight, bias, model, model_args, device)


def ctc_objective(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
    ctc_loss,
) -> torch.Tensor:
    return ctc_loss(
        log_probs=torch.permute(logits.log_softmax(2), [1, 0, 2]),
        targets=labels,
        input_lengths=input_lengths,
        target_lengths=label_lengths,
    ).mean()


def train_residual_adapter(
    method: str,
    data: dict,
    calibration_indices: list[int],
    source_input_layer: int,
    model,
    model_args,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    l2_weight: float,
    grad_clip: float,
    low_rank: int,
    calibration_batch_size: int,
) -> AdapterState:
    features_np = [data["neural_features"][idx].astype(np.float32, copy=False) for idx in calibration_indices]
    labels_np = [data["seq_class_ids"][idx].astype(np.int64, copy=False) for idx in calibration_indices]
    label_lengths = [int(data["seq_len"][idx]) for idx in calibration_indices]
    if calibration_batch_size <= 0:
        calibration_batch_size = len(calibration_indices)
    calibration_batch_size = min(calibration_batch_size, len(calibration_indices))

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    source_weight = model.day_weights[source_input_layer].detach().clone()
    source_bias = model.day_biases[source_input_layer].detach().clone().squeeze(0)
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)

    if method == "feature_diagonal":
        feature_dim = int(features_np[0].shape[1])
        scale = torch.nn.Parameter(torch.ones(feature_dim, device=device))
        bias = torch.nn.Parameter(torch.zeros(feature_dim, device=device))
        params = [scale, bias]

        def forward_for(batch_features: torch.Tensor, batch_lengths: torch.Tensor):
            return forward_feature_diagonal_logits(
                batch_features,
                batch_lengths,
                scale,
                bias,
                source_input_layer,
                model,
                model_args,
                device,
            )

        def regularizer():
            return torch.mean((scale - 1).square()) + torch.mean(bias.square())

    elif method == "layer_bias":
        bias = torch.nn.Parameter(source_bias.clone())
        params = [bias]

        def forward_for(batch_features: torch.Tensor, batch_lengths: torch.Tensor):
            return forward_custom_layer_logits(batch_features, batch_lengths, source_weight, bias, model, model_args, device)

        def regularizer():
            return torch.mean((bias - source_bias).square())

    elif method == "layer_low_rank":
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        left = torch.nn.Parameter(torch.randn(source_weight.shape[0], low_rank, device=device, generator=generator) * 1e-3)
        right = torch.nn.Parameter(torch.zeros(low_rank, source_weight.shape[1], device=device))
        bias_delta = torch.nn.Parameter(torch.zeros_like(source_bias))
        params = [left, right, bias_delta]

        def forward_for(batch_features: torch.Tensor, batch_lengths: torch.Tensor):
            return forward_low_rank_logits(
                batch_features,
                batch_lengths,
                source_weight,
                source_bias,
                left,
                right,
                bias_delta,
                model,
                model_args,
                device,
            )

        def regularizer():
            return torch.mean((left @ right).square()) + torch.mean(bias_delta.square())

    elif method == "layer_full":
        weight = torch.nn.Parameter(source_weight.clone())
        bias = torch.nn.Parameter(source_bias.clone())
        params = [weight, bias]

        def forward_for(batch_features: torch.Tensor, batch_lengths: torch.Tensor):
            return forward_custom_layer_logits(batch_features, batch_lengths, weight, bias, model, model_args, device)

        def regularizer():
            return torch.mean((weight - source_weight).square()) + torch.mean((bias - source_bias).square())

    else:
        raise ValueError(f"Unsupported trainable residual method {method!r}")

    optimizer = torch.optim.Adam(params, lr=learning_rate)

    def iter_batches():
        for start in range(0, len(calibration_indices), calibration_batch_size):
            stop = min(start + calibration_batch_size, len(calibration_indices))
            batch_features, batch_lengths = pad_features(features_np[start:stop], device)
            batch_labels, batch_label_lengths = pad_labels(labels_np[start:stop], label_lengths[start:stop], device)
            yield batch_features, batch_lengths, batch_labels, batch_label_lengths

    def calibration_loss() -> torch.Tensor:
        weighted_losses = []
        total_items = 0
        for batch_features, batch_lengths, batch_labels, batch_label_lengths in iter_batches():
            logits, input_lengths = forward_for(batch_features, batch_lengths)
            batch_loss = ctc_objective(logits, input_lengths, batch_labels, batch_label_lengths, ctc_loss)
            batch_items = int(batch_label_lengths.numel())
            weighted_losses.append(batch_loss * batch_items)
            total_items += batch_items
        return torch.stack(weighted_losses).sum() / max(total_items, 1)

    with torch.no_grad():
        initial_loss = float(calibration_loss().detach().cpu())

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = calibration_loss()
        if l2_weight > 0:
            loss = loss + l2_weight * regularizer()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = float(calibration_loss().detach().cpu())

    payload: dict[str, np.ndarray] = {}
    if method == "feature_diagonal":
        payload["scale"] = scale.detach().cpu().numpy().astype(np.float32)
        payload["bias"] = bias.detach().cpu().numpy().astype(np.float32)
    elif method == "layer_bias":
        payload["weight"] = source_weight.detach().cpu().numpy().astype(np.float32)
        payload["bias"] = bias.detach().cpu().numpy().astype(np.float32)
    elif method == "layer_low_rank":
        weight = source_weight + left @ right
        bias = source_bias + bias_delta
        payload["weight"] = weight.detach().cpu().numpy().astype(np.float32)
        payload["bias"] = bias.detach().cpu().numpy().astype(np.float32)
    elif method == "layer_full":
        payload["weight"] = weight.detach().cpu().numpy().astype(np.float32)
        payload["bias"] = bias.detach().cpu().numpy().astype(np.float32)
    return AdapterState(method=method, initial_loss=initial_loss, final_loss=final_loss, payload=payload)


def apply_moment_match(
    x: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
) -> np.ndarray:
    return source_to_target_moment_match(x.astype(np.float32, copy=False), target_mean, target_std, source_mean, source_std).astype(np.float32, copy=False)


def build_session_summary(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["calibration_trials", "session", "source_session", "source_policy", "method"], as_index=False)
        .agg(
            n_trials=("PER", "size"),
            mean_PER=("PER", "mean"),
            median_PER=("PER", "median"),
            mean_blank_rate=("blank_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_entropy=("entropy", "mean"),
        )
        .reset_index(drop=True)
    )


def build_overall_summary(trials: pd.DataFrame, session_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, k_trials in trials.groupby("calibration_trials"):
        none = k_trials[k_trials["method"] == "none"]
        native = k_trials[k_trials["method"] == "native-day"]
        if none.empty or native.empty:
            continue
        none_weighted = weighted_per(none)
        native_weighted = weighted_per(native)
        gap = none_weighted - native_weighted
        none_by_session = session_summary[
            (session_summary["calibration_trials"] == k) & (session_summary["method"] == "none")
        ][["session", "mean_PER"]].rename(columns={"mean_PER": "none_mean_PER"})
        for method, frame in k_trials.groupby("method", sort=False):
            method_by_session = session_summary[
                (session_summary["calibration_trials"] == k) & (session_summary["method"] == method)
            ][["session", "mean_PER"]]
            joined = method_by_session.merge(none_by_session, on="session", how="inner")
            method_weighted = weighted_per(frame)
            rows.append(
                {
                    "calibration_trials": int(k),
                    "method": method,
                    "weighted_PER": method_weighted,
                    "trial_mean_PER": float(frame["PER"].mean()),
                    "median_trial_PER": float(frame["PER"].median()),
                    "improvement_vs_none": none_weighted - method_weighted,
                    "recovery_fraction": (none_weighted - method_weighted) / gap if gap > 0 else math.nan,
                    "sessions_improved_vs_none": int((joined["mean_PER"] < joined["none_mean_PER"]).sum()) if method != "none" else 0,
                    "sessions_harmed_vs_none": int((joined["mean_PER"] > joined["none_mean_PER"]).sum()) if method != "none" else 0,
                    "num_sessions": int(joined["session"].nunique()),
                    "native_weighted_PER": native_weighted,
                    "none_weighted_PER": none_weighted,
                }
            )
    return pd.DataFrame(rows)


def plot_overall(overall: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = overall.sort_values("weighted_PER")
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(frame["method"], frame["weighted_PER"], color="#4c78a8")
    ax.set_ylabel("Weighted PER")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore conservative residual input calibration variants for T15.")
    parser.add_argument("--model-path", type=Path, default=Path("data/external/btt-25-gru-pure-baseline-0-0898"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val"], default="val")
    parser.add_argument("--source-session", default=None)
    parser.add_argument("--source-policy", choices=["fixed", "previous", "geometry"], default="previous")
    parser.add_argument("--geometry-selection-path", type=Path, default=Path("results/tables/t15_kshot_geometry_source_selection.csv"))
    parser.add_argument("--calibration-trials", nargs="+", type=int, default=[20])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--l2-weight", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--low-rank", type=int, default=8)
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=0,
        help="Number of calibration trials per optimizer batch. Use 0 to fit all K trials at once.",
    )
    parser.add_argument("--min-eval-trials", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-number", type=int, default=-1)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--output-trials", type=Path, default=Path("results/tables/_explore_t15_residual_calibration_trials.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/_explore_t15_residual_calibration_session_summary.csv"))
    parser.add_argument("--output-overall", type=Path, default=Path("results/tables/_explore_t15_residual_calibration_overall.csv"))
    parser.add_argument("--output-training", type=Path, default=Path("results/tables/_explore_t15_residual_calibration_training.csv"))
    parser.add_argument("--output-figure", type=Path, default=Path("results/figures/_explore_t15_residual_calibration_overall.png"))
    args = parser.parse_args()

    device = resolve_requested_device(args.device, args.gpu_number)
    model, model_args = load_official_gru_decoder(ROOT, args.model_path, device)
    _set_use_amp(model_args, enabled=(device.type == "cuda" and bool(model_args["use_amp"]) and not args.disable_amp))
    add_official_model_training_to_path(ROOT)
    from evaluate_model_helpers import load_h5py_file

    b2txt_csv_df = pd.read_csv(args.csv_path)
    sessions = list(model_args["dataset"]["sessions"])
    if args.source_policy == "fixed":
        if args.source_session is None:
            raise ValueError("--source-session is required with --source-policy fixed.")
        if args.source_session not in sessions:
            raise ValueError(f"Unknown source session {args.source_session!r}")
    geometry_source_map = load_geometry_source_map(args.geometry_selection_path) if args.source_policy == "geometry" else {}
    stats_cache: dict[str, tuple[np.ndarray, np.ndarray, int] | None] = {}

    def get_stats(session: str) -> tuple[np.ndarray, np.ndarray, int] | None:
        if session not in stats_cache:
            stats_cache[session] = compute_session_stats(args.data_dir / session / f"data_{args.eval_type}.hdf5")
        return stats_cache[session]

    def choose_source(k: int, target: str) -> str | None:
        if args.source_policy == "fixed":
            return args.source_session
        idx = sessions.index(target)
        if args.source_policy == "previous":
            return sessions[idx - 1] if idx > 0 else None
        if args.source_policy == "geometry":
            return geometry_source_map.get((int(k), target))
        raise ValueError(args.source_policy)

    eval_sessions = []
    for session in sessions:
        eval_file = args.data_dir / session / f"data_{args.eval_type}.hdf5"
        if eval_file.exists():
            eval_sessions.append((session, eval_file))
        if args.max_sessions is not None and len(eval_sessions) >= args.max_sessions:
            break

    rows = []
    training_rows = []
    trainable_methods = [m for m in args.methods if m in {"feature_diagonal", "layer_bias", "layer_low_rank", "layer_full"}]
    total = len(eval_sessions) * len(args.calibration_trials)
    with tqdm(total=total, desc="residual calibration", unit="session-K") as progress:
        for k in sorted(args.calibration_trials):
            for session, eval_file in eval_sessions:
                source_session = choose_source(k, session)
                if source_session is None:
                    progress.update(1)
                    continue
                source_input_layer = sessions.index(source_session)
                source_stats = get_stats(source_session)
                data = load_h5py_file(str(eval_file), b2txt_csv_df)
                n_trials = len(data["neural_features"])
                if n_trials <= k + args.min_eval_trials:
                    progress.update(1)
                    continue

                calibration_indices = list(range(k))
                evaluation_indices = list(range(k, n_trials))
                target_features = [data["neural_features"][idx] for idx in calibration_indices]
                target_mean, target_std, target_n_frames = compute_stats_from_trials(target_features)
                if source_stats is None:
                    source_mean, source_std, source_n_frames = target_mean, target_std, 0
                else:
                    source_mean, source_std, source_n_frames = source_stats
                native_input_layer = sessions.index(session)

                adapters = {}
                for method in trainable_methods:
                    adapter = train_residual_adapter(
                        method=method,
                        data=data,
                        calibration_indices=calibration_indices,
                        source_input_layer=source_input_layer,
                        model=model,
                        model_args=model_args,
                        device=device,
                        epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        l2_weight=args.l2_weight,
                        grad_clip=args.grad_clip,
                        low_rank=args.low_rank,
                        calibration_batch_size=args.calibration_batch_size,
                    )
                    adapters[method] = adapter
                    training_rows.append(
                        {
                            "session": session,
                            "source_session": source_session,
                            "source_policy": args.source_policy,
                            "calibration_trials": k,
                            "method": method,
                            "initial_ctc_loss": adapter.initial_loss,
                            "final_ctc_loss": adapter.final_loss,
                            "epochs": args.epochs,
                            "learning_rate": args.learning_rate,
                            "l2_weight": args.l2_weight,
                            "low_rank": args.low_rank,
                        }
                    )

                for method in args.methods:
                    eval_features = [data["neural_features"][trial_idx].astype(np.float32, copy=False) for trial_idx in evaluation_indices]
                    if method == "moment_match_to_source":
                        eval_features = [apply_moment_match(x, target_mean, target_std, source_mean, source_std) for x in eval_features]
                    eval_input, raw_lengths = pad_features(eval_features, device)
                    with torch.no_grad():
                        if method == "native-day":
                            logits_t, input_lengths_t = forward_fixed_layer_logits(eval_input, raw_lengths, native_input_layer, model, model_args, device)
                            input_layer_session = session
                        elif method in {"none", "moment_match_to_source"}:
                            logits_t, input_lengths_t = forward_fixed_layer_logits(eval_input, raw_lengths, source_input_layer, model, model_args, device)
                            input_layer_session = source_session
                        elif method == "feature_diagonal":
                            scale = torch.tensor(adapters[method].payload["scale"], device=device)
                            bias = torch.tensor(adapters[method].payload["bias"], device=device)
                            logits_t, input_lengths_t = forward_feature_diagonal_logits(
                                eval_input, raw_lengths, scale, bias, source_input_layer, model, model_args, device
                            )
                            input_layer_session = f"{source_session}+feature_diagonal"
                        elif method in {"layer_bias", "layer_low_rank", "layer_full"}:
                            weight = torch.tensor(adapters[method].payload["weight"], device=device)
                            bias = torch.tensor(adapters[method].payload["bias"], device=device)
                            logits_t, input_lengths_t = forward_custom_layer_logits(eval_input, raw_lengths, weight, bias, model, model_args, device)
                            input_layer_session = f"{source_session}+{method}"
                        else:
                            raise ValueError(method)

                    logits_batch = logits_t.float().cpu().numpy()
                    input_lengths = input_lengths_t.detach().cpu().numpy().astype(int)
                    for local_idx, trial_idx in enumerate(evaluation_indices):
                        logits = logits_batch[local_idx, : input_lengths[local_idx], :]
                        pred_ids = greedy_ctc_decode(logits)
                        true_ids = trim_target_sequence(data["seq_class_ids"][trial_idx], data["seq_len"][trial_idx])
                        edit_dist, n_phonemes, per = phoneme_error_rate(true_ids, pred_ids)
                        rows.append(
                            {
                                "session": session,
                                "block": int(data["block_num"][trial_idx]),
                                "trial": int(data["trial_num"][trial_idx]),
                                "source_session": source_session,
                                "source_policy": args.source_policy,
                                "input_layer_session": input_layer_session,
                                "method": method,
                                "calibration_trials": k,
                                "eval_trial_offset": int(trial_idx - k),
                                "source_stats_frames": int(source_n_frames),
                                "target_stats_frames": int(target_n_frames),
                                "sentence_label": _as_plain_string(data["sentence_label"][trial_idx]),
                                "true_phonemes": phoneme_ids_to_string(true_ids),
                                "pred_phonemes": phoneme_ids_to_string(pred_ids),
                                "edit_distance": edit_dist,
                                "num_phonemes": n_phonemes,
                                "PER": per,
                                **logits_quality_metrics(logits),
                            }
                        )
                progress.update(1)

    trials = pd.DataFrame(rows)
    training = pd.DataFrame(training_rows)
    if trials.empty:
        raise ValueError("No rows produced.")
    args.output_trials.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_trials, index=False)
    training.to_csv(args.output_training, index=False)
    session_summary = build_session_summary(trials)
    session_summary.to_csv(args.output_summary, index=False)
    overall = build_overall_summary(trials, session_summary)
    overall.to_csv(args.output_overall, index=False)
    plot_overall(overall, args.output_figure)
    print(overall.sort_values("weighted_PER").to_string(index=False))


if __name__ == "__main__":
    main()
