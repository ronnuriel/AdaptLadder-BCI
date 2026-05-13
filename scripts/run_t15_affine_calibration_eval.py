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
    logits_quality_metrics,
    load_official_gru_decoder,
    phoneme_error_rate,
    phoneme_ids_to_string,
    resolve_device,
    trim_target_sequence,
)
from src.t15_utils import session_date


def resolve_requested_device(device_name: str, gpu_number: int) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        raise ValueError("MPS is not supported for this script because PyTorch CTCLoss does not currently run on MPS.")
    if device_name == "cuda":
        return resolve_device(gpu_number if gpu_number >= 0 else 0)
    if torch.cuda.is_available() and gpu_number >= 0:
        return resolve_device(gpu_number)
    return torch.device("cpu")


METHOD_ORDER = ["native-day", "none", "moment_match_to_source", "diagonal_affine"]
METHOD_LABELS = {
    "native-day": "Native-day",
    "none": "Cross-day none",
    "moment_match_to_source": "Moment match",
    "diagonal_affine": "Diagonal affine",
}
METHOD_COLORS = {
    "native-day": "#2a9d8f",
    "none": "#b23a48",
    "moment_match_to_source": "#7b2cbf",
    "diagonal_affine": "#f4a261",
}


@dataclass
class AdapterState:
    scale: np.ndarray
    bias: np.ndarray
    final_loss: float
    initial_loss: float


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


def _count_trials(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return len(handle.keys())


def _set_use_amp(model_args, enabled: bool) -> None:
    try:
        model_args["use_amp"] = enabled
    except TypeError:
        model_args.use_amp = enabled


def weighted_per(trials: pd.DataFrame) -> float:
    return float(trials["edit_distance"].sum() / trials["num_phonemes"].sum())


def compute_stats_from_trials(features: list[np.ndarray], eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, int]:
    sum_x = None
    sum_x2 = None
    n_frames = 0
    for feature in features:
        x = feature.astype(np.float64, copy=False)
        if sum_x is None:
            sum_x = np.zeros(x.shape[1], dtype=np.float64)
            sum_x2 = np.zeros(x.shape[1], dtype=np.float64)
        sum_x += x.sum(axis=0)
        sum_x2 += np.square(x).sum(axis=0)
        n_frames += x.shape[0]

    if sum_x is None or sum_x2 is None or n_frames == 0:
        raise ValueError("No neural frames available for stats.")

    mean = sum_x / n_frames
    variance = sum_x2 / n_frames - mean * mean
    std = np.sqrt(np.maximum(variance, eps))
    return mean.astype(np.float32), std.astype(np.float32), n_frames


def compute_session_stats(file_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    features = []
    with h5py.File(file_path, "r") as handle:
        for trial_id in handle.keys():
            features.append(handle[trial_id]["input_features"][:])
    return compute_stats_from_trials(features)


def pad_features(features: list[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.shape[0] for x in features)
    feature_dim = features[0].shape[1]
    padded = np.zeros((len(features), max_len, feature_dim), dtype=np.float32)
    lengths = []
    for idx, x in enumerate(features):
        padded[idx, : x.shape[0], :] = x
        lengths.append(x.shape[0])
    return torch.tensor(padded, device=device), torch.tensor(lengths, device=device, dtype=torch.long)


def pad_labels(labels: list[np.ndarray], label_lengths: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(label_lengths)
    padded = np.zeros((len(labels), max_len), dtype=np.int32)
    for idx, (label, length) in enumerate(zip(labels, label_lengths)):
        padded[idx, :length] = label[:length]
    return torch.tensor(padded, device=device, dtype=torch.long), torch.tensor(label_lengths, device=device, dtype=torch.long)


def smoothed_lengths(raw_lengths: torch.Tensor, model_args) -> torch.Tensor:
    """Match evaluate_model_helpers.runSingleDecodingStep, which smooths with valid padding."""
    from scipy.ndimage import gaussian_filter1d

    smooth_args = model_args["dataset"]["data_transforms"]
    kernel_size = int(smooth_args["smooth_kernel_size"])
    kernel_std = float(smooth_args["smooth_kernel_std"])
    impulse = np.zeros(kernel_size, dtype=np.float32)
    impulse[kernel_size // 2] = 1
    kernel = gaussian_filter1d(impulse, kernel_std)
    kernel_len = int(np.argwhere(kernel > 0.01).size)
    return raw_lengths - kernel_len + 1


def adjusted_ctc_lengths(raw_lengths: torch.Tensor, model_args) -> torch.Tensor:
    smooth_lens = smoothed_lengths(raw_lengths, model_args)
    patch_size = int(model_args["model"]["patch_size"])
    patch_stride = int(model_args["model"]["patch_stride"])
    if patch_size <= 0:
        return smooth_lens.to(torch.long)
    return torch.div(smooth_lens - patch_size, patch_stride, rounding_mode="floor").to(torch.long) + 1


def forward_logits(
    x: torch.Tensor,
    raw_lengths: torch.Tensor,
    input_layer: int,
    model,
    model_args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    add_official_model_training_to_path(ROOT)
    from data_augmentations import gauss_smooth

    with torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and bool(model_args["use_amp"])), dtype=torch.bfloat16):
        x = gauss_smooth(
            inputs=x,
            device=device,
            smooth_kernel_std=model_args["dataset"]["data_transforms"]["smooth_kernel_std"],
            smooth_kernel_size=model_args["dataset"]["data_transforms"]["smooth_kernel_size"],
            padding="valid",
        )
        day_idx = torch.full((x.shape[0],), int(input_layer), device=device, dtype=torch.long)
        logits = model(x=x, day_idx=day_idx)
        input_lengths = adjusted_ctc_lengths(raw_lengths, model_args).to(device)
    return logits, input_lengths


def calibration_loss(
    features: torch.Tensor,
    raw_lengths: torch.Tensor,
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
    input_layer: int,
    model,
    model_args,
    device: torch.device,
    log_scale: torch.Tensor,
    bias: torch.Tensor,
    ctc_loss,
    l2_weight: float,
) -> torch.Tensor:
    scale = torch.exp(torch.clamp(log_scale, -2.0, 2.0)).view(1, 1, -1)
    x = features * scale + bias.view(1, 1, -1)
    logits, input_lengths = forward_logits(x, raw_lengths, input_layer, model, model_args, device)
    loss = ctc_loss(
        log_probs=torch.permute(logits.log_softmax(2), [1, 0, 2]),
        targets=labels,
        input_lengths=input_lengths,
        target_lengths=label_lengths,
    ).mean()
    if l2_weight > 0:
        loss = loss + l2_weight * (torch.mean(log_scale.square()) + torch.mean(bias.square()))
    return loss


def train_diagonal_affine_adapter(
    data: dict,
    calibration_indices: list[int],
    input_layer: int,
    model,
    model_args,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    l2_weight: float,
    grad_clip: float,
) -> AdapterState:
    features_np = [data["neural_features"][idx].astype(np.float32, copy=False) for idx in calibration_indices]
    labels_np = [data["seq_class_ids"][idx].astype(np.int32, copy=False) for idx in calibration_indices]
    label_lengths = [int(data["seq_len"][idx]) for idx in calibration_indices]
    features, raw_lengths = pad_features(features_np, device)
    labels, phone_seq_lens = pad_labels(labels_np, label_lengths, device)

    feature_dim = features.shape[-1]
    log_scale = torch.nn.Parameter(torch.zeros(feature_dim, device=device))
    bias = torch.nn.Parameter(torch.zeros(feature_dim, device=device))
    optimizer = torch.optim.Adam([log_scale, bias], lr=learning_rate)
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    with torch.no_grad():
        initial_loss = float(
            calibration_loss(
                features,
                raw_lengths,
                labels,
                phone_seq_lens,
                input_layer,
                model,
                model_args,
                device,
                log_scale,
                bias,
                ctc_loss,
                l2_weight=0.0,
            )
            .detach()
            .cpu()
        )

    for _ in range(epochs):
        # cuDNN RNN backward on CUDA requires train mode even though the
        # pretrained decoder weights remain frozen.
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = calibration_loss(
            features,
            raw_lengths,
            labels,
            phone_seq_lens,
            input_layer,
            model,
            model_args,
            device,
            log_scale,
            bias,
            ctc_loss,
            l2_weight=l2_weight,
        )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([log_scale, bias], max_norm=grad_clip)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = float(
            calibration_loss(
                features,
                raw_lengths,
                labels,
                phone_seq_lens,
                input_layer,
                model,
                model_args,
                device,
                log_scale,
                bias,
                ctc_loss,
                l2_weight=0.0,
            )
            .detach()
            .cpu()
        )
    scale = torch.exp(torch.clamp(log_scale, -2.0, 2.0)).detach().cpu().numpy().astype(np.float32)
    bias_np = bias.detach().cpu().numpy().astype(np.float32)
    return AdapterState(scale=scale, bias=bias_np, initial_loss=initial_loss, final_loss=final_loss)


def apply_method(
    x: np.ndarray,
    method: str,
    adapter: AdapterState | None,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if method in {"native-day", "none"}:
        return x
    if method == "moment_match_to_source":
        return source_to_target_moment_match(x, target_mean, target_std, source_mean, source_std).astype(np.float32, copy=False)
    if method == "diagonal_affine":
        if adapter is None:
            raise ValueError("diagonal_affine requires a trained adapter.")
        return (x * adapter.scale + adapter.bias).astype(np.float32, copy=False)
    raise ValueError(f"Unknown method {method!r}")


def build_session_summary(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["calibration_trials", "session", "source_session", "method"], as_index=False)
        .agg(
            n_trials=("PER", "size"),
            mean_PER=("PER", "mean"),
            median_PER=("PER", "median"),
            mean_blank_rate=("blank_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_entropy=("entropy", "mean"),
        )
        .sort_values(["calibration_trials", "method", "session"])
        .reset_index(drop=True)
    )


def build_overall_summary(trials: pd.DataFrame, session_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, k_trials in trials.groupby("calibration_trials", sort=True):
        none = k_trials[k_trials["method"] == "none"]
        native = k_trials[k_trials["method"] == "native-day"]
        if none.empty or native.empty:
            continue
        none_weighted = weighted_per(none)
        native_weighted = weighted_per(native)
        recoverable_gap = none_weighted - native_weighted
        none_by_session = session_summary[
            (session_summary["calibration_trials"] == k) & (session_summary["method"] == "none")
        ][["session", "mean_PER"]].rename(columns={"mean_PER": "none_mean_PER"})

        for method, frame in k_trials.groupby("method", sort=False):
            method_weighted = weighted_per(frame)
            method_by_session = session_summary[
                (session_summary["calibration_trials"] == k) & (session_summary["method"] == method)
            ][["session", "mean_PER"]]
            joined = method_by_session.merge(none_by_session, on="session", how="inner")
            rows.append(
                {
                    "calibration_trials": int(k),
                    "method": method,
                    "weighted_PER": method_weighted,
                    "trial_mean_PER": float(frame["PER"].mean()),
                    "median_trial_PER": float(frame["PER"].median()),
                    "delta_vs_none": method_weighted - none_weighted,
                    "improvement_vs_none": none_weighted - method_weighted,
                    "recovery_fraction": (none_weighted - method_weighted) / recoverable_gap if recoverable_gap > 0 else math.nan,
                    "sessions_improved_vs_none": int((joined["mean_PER"] < joined["none_mean_PER"]).sum()) if method != "none" else 0,
                    "sessions_harmed_vs_none": int((joined["mean_PER"] > joined["none_mean_PER"]).sum()) if method != "none" else 0,
                    "num_sessions": int(joined["session"].nunique()),
                    "native_weighted_PER": native_weighted,
                    "none_weighted_PER": none_weighted,
                }
            )

    summary = pd.DataFrame(rows)
    order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    summary["sort_order"] = summary["method"].map(order).fillna(99)
    return summary.sort_values(["calibration_trials", "sort_order"]).drop(columns=["sort_order"]).reset_index(drop=True)


def plot_weighted_per(overall: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ks = sorted(overall["calibration_trials"].unique())
    methods = [method for method in METHOD_ORDER if method in set(overall["method"])]
    x = np.arange(len(ks))
    width = 0.18
    for idx, method in enumerate(methods):
        frame = overall[overall["method"] == method].set_index("calibration_trials").loc[ks].reset_index()
        offset = (idx - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, frame["weighted_PER"], width=width, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS[method])
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Phoneme-weighted PER")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_recovery(overall: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_methods = [method for method in ("moment_match_to_source", "diagonal_affine") if method in set(overall["method"])]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for method in plot_methods:
        frame = overall[overall["method"] == method].sort_values("calibration_trials")
        ax.plot(
            frame["calibration_trials"],
            frame["recovery_fraction"] * 100,
            marker="o",
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS[method],
        )
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_xlabel("Labeled calibration trials")
    ax.set_ylabel("Recovered cross-day gap (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate small supervised diagonal affine calibration for T15 cross-day decoding.")
    parser.add_argument("--model-path", type=Path, default=Path("data/external/btt-25-gru-pure-baseline-0-0898"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val"], default="val")
    parser.add_argument("--source-session", required=True)
    parser.add_argument("--calibration-trials", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--min-eval-trials", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--gpu-number", type=int, default=-1)
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable CUDA autocast/AMP. Useful because CUDA CTC loss does not support bfloat16 on some PyTorch builds.",
    )
    parser.add_argument("--output-trials", type=Path, default=Path("results/tables/t15_affine_calibration_trial_results_source_middle.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/t15_affine_calibration_session_summary_source_middle.csv"))
    parser.add_argument("--output-overall", type=Path, default=Path("results/tables/t15_affine_calibration_overall_summary_source_middle.csv"))
    parser.add_argument("--output-training", type=Path, default=Path("results/tables/t15_affine_calibration_training_summary_source_middle.csv"))
    parser.add_argument(
        "--output-weighted-figure",
        type=Path,
        default=Path("results/figures/t15_affine_calibration_weighted_per_source_middle.png"),
    )
    parser.add_argument(
        "--output-recovery-figure",
        type=Path,
        default=Path("results/figures/t15_affine_calibration_recovery_source_middle.png"),
    )
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    if "none" not in args.methods or "native-day" not in args.methods:
        raise ValueError("--methods must include native-day and none so recovery can be measured.")
    if "diagonal_affine" not in args.methods:
        raise ValueError("--methods must include diagonal_affine for this calibration experiment.")

    device = resolve_requested_device(args.device, args.gpu_number)
    model, model_args = load_official_gru_decoder(ROOT, args.model_path, device)
    _set_use_amp(model_args, enabled=(device.type == "cuda" and bool(model_args["use_amp"]) and not args.disable_amp))
    add_official_model_training_to_path(ROOT)
    from evaluate_model_helpers import load_h5py_file

    b2txt_csv_df = pd.read_csv(args.csv_path)
    sessions = list(model_args["dataset"]["sessions"])
    if args.source_session not in sessions:
        raise ValueError(f"Unknown source session {args.source_session!r}")
    source_input_layer = sessions.index(args.source_session)
    source_stats_file = args.data_dir / args.source_session / f"data_{args.eval_type}.hdf5"
    source_mean, source_std, source_n_frames = compute_session_stats(source_stats_file)

    eval_sessions = []
    for session in sessions:
        eval_file = args.data_dir / session / f"data_{args.eval_type}.hdf5"
        if eval_file.exists():
            eval_sessions.append((session, eval_file))
        if args.max_sessions is not None and len(eval_sessions) >= args.max_sessions:
            break
    if not eval_sessions:
        raise FileNotFoundError(f"No data_{args.eval_type}.hdf5 files found in {args.data_dir}")

    rows = []
    training_rows = []
    total_sessions = len(eval_sessions) * len(args.calibration_trials)

    with tqdm(total=total_sessions, desc="affine calibration", unit="session-K") as progress:
        for k in sorted(args.calibration_trials):
            for session, eval_file in eval_sessions:
                data = load_h5py_file(str(eval_file), b2txt_csv_df)
                n_trials = len(data["neural_features"])
                if n_trials <= k + args.min_eval_trials:
                    progress.update(1)
                    continue

                calibration_indices = list(range(k))
                evaluation_indices = list(range(k, n_trials))
                target_features = [data["neural_features"][idx] for idx in calibration_indices]
                target_mean, target_std, target_n_frames = compute_stats_from_trials(target_features)
                native_input_layer = sessions.index(session)

                adapter = train_diagonal_affine_adapter(
                    data=data,
                    calibration_indices=calibration_indices,
                    input_layer=source_input_layer,
                    model=model,
                    model_args=model_args,
                    device=device,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    l2_weight=args.l2_weight,
                    grad_clip=args.grad_clip,
                )
                training_rows.append(
                    {
                        "session": session,
                        "source_session": args.source_session,
                        "calibration_trials": k,
                        "initial_ctc_loss": adapter.initial_loss,
                        "final_ctc_loss": adapter.final_loss,
                        "epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "l2_weight": args.l2_weight,
                    }
                )

                for method in args.methods:
                    input_layer = native_input_layer if method == "native-day" else source_input_layer
                    eval_features = [
                        apply_method(
                            data["neural_features"][trial_idx],
                            method,
                            adapter,
                            target_mean,
                            target_std,
                            source_mean,
                            source_std,
                        )
                        for trial_idx in evaluation_indices
                    ]
                    eval_input, raw_lengths = pad_features(eval_features, device)
                    with torch.no_grad():
                        logits_t, input_lengths_t = forward_logits(eval_input, raw_lengths, input_layer, model, model_args, device)
                    logits_batch = logits_t.float().cpu().numpy()
                    input_lengths = input_lengths_t.detach().cpu().numpy().astype(int)

                    for local_idx, trial_idx in enumerate(evaluation_indices):
                        logits = logits_batch[local_idx, : input_lengths[local_idx], :]
                        pred_ids = greedy_ctc_decode(logits)
                        true_ids = trim_target_sequence(data["seq_class_ids"][trial_idx], data["seq_len"][trial_idx])
                        edit_dist, n_phonemes, per = phoneme_error_rate(true_ids, pred_ids)
                        quality = logits_quality_metrics(logits)
                        rows.append(
                            {
                                "session": session,
                                "block": int(data["block_num"][trial_idx]),
                                "trial": int(data["trial_num"][trial_idx]),
                                "source_session": args.source_session,
                                "input_layer_session": sessions[input_layer],
                                "method": method,
                                "calibration_trials": k,
                                "eval_trial_offset": int(trial_idx - k),
                                "stats_mode": "first_k_calibration",
                                "source_stats_frames": int(source_n_frames),
                                "target_stats_frames": int(target_n_frames),
                                "sentence_label": _as_plain_string(data["sentence_label"][trial_idx]),
                                "true_phonemes": phoneme_ids_to_string(true_ids),
                                "pred_phonemes": phoneme_ids_to_string(pred_ids),
                                "edit_distance": edit_dist,
                                "num_phonemes": n_phonemes,
                                "PER": per,
                                **quality,
                            }
                        )
                progress.update(1)

    trials = pd.DataFrame(rows)
    training = pd.DataFrame(training_rows)
    if trials.empty:
        raise ValueError("No evaluation rows were produced. Try smaller K or lower --min-eval-trials.")

    args.output_trials.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_overall.parent.mkdir(parents=True, exist_ok=True)
    args.output_training.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_trials, index=False)
    training.to_csv(args.output_training, index=False)

    session_summary = build_session_summary(trials)
    session_summary.to_csv(args.output_summary, index=False)
    overall = build_overall_summary(trials, session_summary)
    overall.to_csv(args.output_overall, index=False)

    source_label = args.source_session.replace("t15.", "")
    plot_weighted_per(overall, args.output_weighted_figure, f"T15 small affine calibration via source {source_label}")
    plot_recovery(overall, args.output_recovery_figure, f"T15 recovered gap via source {source_label}")

    print(f"Loaded official GRU checkpoint on {device}.")
    print(f"Source session: {args.source_session} (input layer {source_input_layer})")
    print(f"Calibration trials: {', '.join(str(k) for k in sorted(args.calibration_trials))}")
    print(f"Wrote trial-level results to {args.output_trials}")
    print(f"Wrote session summary to {args.output_summary}")
    print(f"Wrote overall summary to {args.output_overall}")
    print(f"Wrote training summary to {args.output_training}")
    print(f"Wrote {args.output_weighted_figure}")
    print(f"Wrote {args.output_recovery_figure}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
