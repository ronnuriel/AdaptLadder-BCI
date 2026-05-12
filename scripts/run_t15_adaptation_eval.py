from __future__ import annotations

import argparse
import math
import sys
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

from src.adaptation import source_to_target_moment_match, target_zscore
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


ADAPTATION_METHODS = ("none", "target_zscore", "moment_match_to_source")
METHOD_LABELS = {
    "native-day": "Native-day",
    "none": "Cross-day none",
    "target_zscore": "Target z-score",
    "moment_match_to_source": "Moment match",
}
METHOD_COLORS = {
    "native-day": "#2a9d8f",
    "none": "#b23a48",
    "target_zscore": "#457b9d",
    "moment_match_to_source": "#7b2cbf",
}


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


def compute_session_stats(file_path: Path, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, int]:
    sum_x = None
    sum_x2 = None
    n_frames = 0

    with h5py.File(file_path, "r") as handle:
        for trial_id in handle.keys():
            x = handle[trial_id]["input_features"][:].astype(np.float64, copy=False)
            if sum_x is None:
                sum_x = np.zeros(x.shape[1], dtype=np.float64)
                sum_x2 = np.zeros(x.shape[1], dtype=np.float64)
            sum_x += x.sum(axis=0)
            sum_x2 += np.square(x).sum(axis=0)
            n_frames += x.shape[0]

    if sum_x is None or sum_x2 is None or n_frames == 0:
        raise ValueError(f"No neural frames found in {file_path}")

    mean = sum_x / n_frames
    variance = sum_x2 / n_frames - mean * mean
    std = np.sqrt(np.maximum(variance, eps))
    return mean.astype(np.float32), std.astype(np.float32), n_frames


def transform_features(
    x: np.ndarray,
    method: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if method == "none":
        return x
    if method == "target_zscore":
        return target_zscore(x, target_mean, target_std).astype(np.float32, copy=False)
    if method == "moment_match_to_source":
        return source_to_target_moment_match(x, target_mean, target_std, source_mean, source_std).astype(np.float32, copy=False)
    raise ValueError(f"Unknown adaptation method {method!r}")


def build_session_summary(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["session", "source_session", "adaptation_method"], as_index=False)
        .agg(
            n_trials=("PER", "size"),
            mean_PER=("PER", "mean"),
            median_PER=("PER", "median"),
            mean_blank_rate=("blank_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_entropy=("entropy", "mean"),
        )
        .sort_values(["adaptation_method", "session"])
        .reset_index(drop=True)
    )


def build_overall_summary(trials: pd.DataFrame, session_summary: pd.DataFrame, native_trials_path: Path) -> pd.DataFrame:
    if not native_trials_path.exists():
        raise FileNotFoundError(f"Native trial results not found: {native_trials_path}")

    native_trials = pd.read_csv(native_trials_path)
    native_weighted = weighted_per(native_trials)
    native_trial_mean = float(native_trials["PER"].mean())
    none_trials = trials[trials["adaptation_method"] == "none"]
    if none_trials.empty:
        raise ValueError("Overall summary requires the 'none' adaptation baseline.")

    none_weighted = weighted_per(none_trials)
    recoverable_gap = none_weighted - native_weighted
    none_by_session = session_summary[session_summary["adaptation_method"] == "none"][["session", "mean_PER"]].rename(
        columns={"mean_PER": "none_mean_PER"}
    )

    rows = [
        {
            "adaptation_method": "native-day",
            "weighted_PER": native_weighted,
            "trial_mean_PER": native_trial_mean,
            "median_trial_PER": float(native_trials["PER"].median()),
            "delta_vs_none": native_weighted - none_weighted,
            "improvement_vs_none": none_weighted - native_weighted,
            "recovery_fraction": 1.0 if recoverable_gap > 0 else math.nan,
            "sessions_improved_vs_none": math.nan,
            "sessions_harmed_vs_none": math.nan,
            "num_sessions": int(native_trials["session"].nunique()),
            "native_weighted_PER": native_weighted,
            "none_weighted_PER": none_weighted,
        }
    ]

    for method, frame in trials.groupby("adaptation_method", sort=False):
        method_weighted = weighted_per(frame)
        joined = session_summary[session_summary["adaptation_method"] == method][["session", "mean_PER"]].merge(
            none_by_session, on="session", how="inner"
        )
        rows.append(
            {
                "adaptation_method": method,
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
    order = {method: idx for idx, method in enumerate(["native-day", "none", "target_zscore", "moment_match_to_source"])}
    summary["sort_order"] = summary["adaptation_method"].map(order).fillna(99)
    return summary.sort_values("sort_order").drop(columns=["sort_order"]).reset_index(drop=True)


def plot_weighted_per(overall: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = overall.copy()
    labels = [METHOD_LABELS.get(method, method) for method in plot_frame["adaptation_method"]]
    colors = [METHOD_COLORS.get(method, "#555555") for method in plot_frame["adaptation_method"]]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bars = ax.bar(labels, plot_frame["weighted_PER"], color=colors)
    for bar, value in zip(bars, plot_frame["weighted_PER"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Phoneme-weighted PER")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(plot_frame["weighted_PER"].max() * 1.18, 0.12))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_delta_by_session(session_summary: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    none = session_summary[session_summary["adaptation_method"] == "none"][["session", "mean_PER"]].rename(
        columns={"mean_PER": "none_mean_PER"}
    )
    methods = [method for method in ("target_zscore", "moment_match_to_source") if method in set(session_summary["adaptation_method"])]
    if not methods:
        return

    target_order = sorted(none["session"].unique(), key=session_date)
    x = np.arange(len(target_order))
    width = 0.38 if len(methods) == 2 else 0.6

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    for idx, method in enumerate(methods):
        method_frame = session_summary[session_summary["adaptation_method"] == method][["session", "mean_PER"]].merge(
            none, on="session", how="inner"
        )
        method_frame["delta_vs_none"] = method_frame["mean_PER"] - method_frame["none_mean_PER"]
        method_frame = method_frame.set_index("session").loc[target_order].reset_index()
        offset = (idx - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            method_frame["delta_vs_none"],
            width=width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#555555"),
        )

    tick_step = max(1, len(target_order) // 12)
    ticks = np.arange(0, len(target_order), tick_step)
    ax.axhline(0, color="0.25", linewidth=1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels([target_order[i].replace("t15.", "") for i in ticks], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Delta mean trial PER vs none")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lightweight input adaptation on the T15 cross-day decoder stress test.")
    parser.add_argument("--model-path", type=Path, default=Path("data/external/btt-25-gru-pure-baseline-0-0898"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val", "test"], default="val")
    parser.add_argument("--source-session", required=True)
    parser.add_argument("--adaptations", nargs="+", choices=ADAPTATION_METHODS, default=list(ADAPTATION_METHODS))
    parser.add_argument("--gpu-number", type=int, default=-1)
    parser.add_argument("--native-trials", type=Path, default=Path("results/tables/t15_decoder_probe_val.csv"))
    parser.add_argument("--output-trials", type=Path, default=Path("results/tables/t15_adaptation_trial_results.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/t15_adaptation_session_summary.csv"))
    parser.add_argument("--output-overall", type=Path, default=Path("results/tables/t15_adaptation_overall_summary.csv"))
    parser.add_argument(
        "--output-weighted-figure",
        type=Path,
        default=Path("results/figures/t15_adaptation_ladder_weighted_per.png"),
    )
    parser.add_argument(
        "--output-delta-figure",
        type=Path,
        default=Path("results/figures/t15_adaptation_ladder_delta_per_by_session.png"),
    )
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--max-trials-per-session", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    if args.eval_type != "val":
        raise ValueError("Adaptation evaluation currently requires validation labels for PER.")
    if "none" not in args.adaptations:
        raise ValueError("--adaptations must include 'none' so recovery can be measured.")

    device = resolve_device(args.gpu_number)
    model, model_args = load_official_gru_decoder(ROOT, args.model_path, device)
    _set_use_amp(model_args, enabled=(device.type == "cuda" and bool(model_args["use_amp"])))
    add_official_model_training_to_path(ROOT)
    from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep

    b2txt_csv_df = pd.read_csv(args.csv_path)
    sessions = list(model_args["dataset"]["sessions"])
    if args.source_session not in sessions:
        raise ValueError(f"Unknown source session {args.source_session!r}")
    source_input_layer = sessions.index(args.source_session)

    eval_sessions = []
    total_trials = 0
    for session in sessions:
        eval_file = args.data_dir / session / f"data_{args.eval_type}.hdf5"
        if not eval_file.exists():
            continue
        eval_sessions.append((session, eval_file))
        total_trials += _count_trials(eval_file)
        if args.max_sessions is not None and len(eval_sessions) >= args.max_sessions:
            break

    if not eval_sessions:
        raise FileNotFoundError(f"No data_{args.eval_type}.hdf5 files found in {args.data_dir}")

    if args.max_trials_per_session is not None:
        total_trials = sum(min(_count_trials(path), args.max_trials_per_session) for _, path in eval_sessions)

    source_file = args.data_dir / args.source_session / f"data_{args.eval_type}.hdf5"
    source_mean, source_std, source_n_frames = compute_session_stats(source_file)
    session_stats = {}
    for session, eval_file in tqdm(eval_sessions, desc="session stats", unit="session"):
        session_stats[session] = compute_session_stats(eval_file)

    rows = []
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    total_evals = total_trials * len(args.adaptations)
    desc = f"adaptation {args.eval_type} PER via {args.source_session}"

    with tqdm(total=total_evals, desc=desc, unit="trial") as progress:
        for session, eval_file in eval_sessions:
            data = load_h5py_file(str(eval_file), b2txt_csv_df)
            target_mean, target_std, target_n_frames = session_stats[session]
            n_trials = len(data["neural_features"])
            if args.max_trials_per_session is not None:
                n_trials = min(n_trials, args.max_trials_per_session)

            for method in args.adaptations:
                for trial_idx in range(n_trials):
                    x_corr = transform_features(
                        data["neural_features"][trial_idx],
                        method,
                        target_mean,
                        target_std,
                        source_mean,
                        source_std,
                    )
                    neural_input = torch.tensor(np.expand_dims(x_corr, axis=0), device=device, dtype=dtype)

                    logits = runSingleDecodingStep(
                        neural_input,
                        source_input_layer,
                        model,
                        model_args,
                        device,
                    )[0]

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
                            "input_layer_session": sessions[source_input_layer],
                            "adaptation_method": method,
                            "stats_mode": "full_session_unsupervised",
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
    args.output_trials.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_overall.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_trials, index=False)

    session_summary = build_session_summary(trials)
    session_summary.to_csv(args.output_summary, index=False)

    overall = build_overall_summary(trials, session_summary, args.native_trials)
    overall.to_csv(args.output_overall, index=False)

    source_label = args.source_session.replace("t15.", "")
    plot_weighted_per(overall, args.output_weighted_figure, f"T15 adaptation ladder via source {source_label}")
    plot_delta_by_session(session_summary, args.output_delta_figure, f"T15 adaptation deltas via source {source_label}")

    print(f"Loaded official GRU checkpoint on {device}.")
    print(f"Source session: {args.source_session} (input layer {source_input_layer})")
    print(f"Adaptations: {', '.join(args.adaptations)}")
    print(f"Wrote trial-level results to {args.output_trials}")
    print(f"Wrote session summary to {args.output_summary}")
    print(f"Wrote overall summary to {args.output_overall}")
    print(f"Wrote {args.output_weighted_figure}")
    print(f"Wrote {args.output_delta_figure}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
