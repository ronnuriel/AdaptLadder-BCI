#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t15_geometry_source_selection_eval import (  # noqa: E402
    _as_plain_string,
    _set_use_amp,
    compute_session_stats,
    covariance_stats,
    discover_sessions,
    weighted_per,
)
from scripts.run_t15_kshot_geometry_source_selection import calibration_window_stats  # noqa: E402
from src.decoder_eval import (  # noqa: E402
    add_official_model_training_to_path,
    greedy_ctc_decode,
    logits_quality_metrics,
    load_official_gru_decoder,
    phoneme_error_rate,
    phoneme_ids_to_string,
    resolve_device,
    trim_target_sequence,
)


def psd_power(matrix: np.ndarray, power: float, eps: float = 1e-5) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix.astype(np.float64, copy=False))
    eigvals = np.maximum(eigvals, eps)
    return (eigvecs * (eigvals**power)) @ eigvecs.T


def coral_align_features(
    x: np.ndarray,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    source_mean: np.ndarray,
    source_cov: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    # Align target features toward source second-order geometry:
    # whiten by target covariance estimated from the beginning-of-day window,
    # then recolor by source covariance.
    target_inv_sqrt = psd_power(target_cov, -0.5, eps=eps)
    source_sqrt = psd_power(source_cov, 0.5, eps=eps)
    aligned = (x.astype(np.float64, copy=False) - target_mean) @ target_inv_sqrt @ source_sqrt + source_mean
    return aligned.astype(np.float32, copy=False)


def session_weighted_summary(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["calibration_trials", "session", "policy", "source_session"], as_index=False)
        .agg(
            edit_distance=("edit_distance", "sum"),
            num_phonemes=("num_phonemes", "sum"),
            n_trials=("PER", "size"),
            mean_trial_PER=("PER", "mean"),
            median_trial_PER=("PER", "median"),
            mean_blank_rate=("blank_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_entropy=("entropy", "mean"),
        )
        .assign(weighted_PER=lambda x: x["edit_distance"] / x["num_phonemes"])
    )


def overall_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (k, policy), group in trials.groupby(["calibration_trials", "policy"]):
        rows.append(
            {
                "calibration_trials": int(k),
                "policy": policy,
                "weighted_PER": weighted_per(group),
                "trial_mean_PER": float(group["PER"].mean()),
                "num_sessions": int(group["session"].nunique()),
                "num_trials": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["calibration_trials", "policy"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe whether CORAL feature alignment improves T15 previous/geometry source retrieval."
    )
    parser.add_argument("--model-path", type=Path, default=Path("data/external/btt-25-gru-pure-baseline-0-0898"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val", "test"], default="val")
    parser.add_argument("--source-stats-split", default="train")
    parser.add_argument("--calibration-trials", type=int, nargs="+", default=[20])
    parser.add_argument("--policies", nargs="+", default=["previous", "previous_coral", "geometry", "geometry_coral"])
    parser.add_argument("--selection-path", type=Path, default=Path("results/tables/t15_kshot_selected_sources_annotated.csv"))
    parser.add_argument("--max-source-frames", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--cov-shrinkage", type=float, default=0.05)
    parser.add_argument("--coral-eps", type=float, default=1e-5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-number", type=int, default=-1)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-trials-per-session", type=int, default=None)
    parser.add_argument("--output-trials", type=Path, default=Path("results/tables/_explore_t15_geometry_alignment_trials.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/_explore_t15_geometry_alignment_session_summary.csv"))
    parser.add_argument("--output-overall", type=Path, default=Path("results/tables/_explore_t15_geometry_alignment_overall.csv"))
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device(f"cuda:{args.gpu_number if args.gpu_number >= 0 else 0}")
    else:
        device = resolve_device(args.gpu_number)

    model, model_args = load_official_gru_decoder(ROOT, args.model_path, device)
    _set_use_amp(model_args, enabled=(device.type == "cuda" and not args.disable_amp and bool(model_args["use_amp"])))
    add_official_model_training_to_path(ROOT)
    from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep

    selection = pd.read_csv(args.selection_path)
    b2txt_csv_df = pd.read_csv(args.csv_path)
    model_sessions = list(model_args["dataset"]["sessions"])
    eval_sessions = discover_sessions(args.data_dir, args.eval_type, model_sessions)
    source_sessions = discover_sessions(args.data_dir, args.source_stats_split, model_sessions)
    sessions = [session for session in eval_sessions if session in set(selection["target_session"])]
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]
    if not sessions:
        raise ValueError("No sessions to evaluate.")

    source_stats = compute_session_stats(
        data_dir=args.data_dir,
        sessions=source_sessions,
        split=args.source_stats_split,
        max_frames=args.max_source_frames,
        seed=args.seed,
        n_components=args.n_components,
        shrinkage=args.cov_shrinkage,
    )

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    loaded = {}
    total = 0
    jobs: list[tuple[str, int, str, str, bool]] = []
    for session in sessions:
        data = load_h5py_file(str(args.data_dir / session / f"data_{args.eval_type}.hdf5"), b2txt_csv_df)
        if args.max_trials_per_session is not None:
            for key in ["neural_features", "seq_class_ids", "seq_len", "block_num", "trial_num", "sentence_label"]:
                if key in data:
                    data[key] = data[key][: args.max_trials_per_session]
        loaded[session] = data
        n_trials = len(data["neural_features"])
        for k in args.calibration_trials:
            if n_trials <= k:
                continue
            row = selection[(selection["target_session"] == session) & (selection["calibration_trials"] == k)]
            if row.empty:
                continue
            row = row.iloc[0]
            source_by_policy = {
                "previous": str(row["previous_session"]),
                "previous_coral": str(row["previous_session"]),
                "geometry": str(row["source_session"]),
                "geometry_coral": str(row["source_session"]),
            }
            for policy in args.policies:
                if policy not in source_by_policy:
                    raise ValueError(f"Unknown policy {policy!r}")
                source = source_by_policy[policy]
                if source not in model_sessions or source not in source_stats:
                    continue
                jobs.append((session, k, policy, source, policy.endswith("_coral")))
                total += n_trials - k

    rows = []
    with tqdm(total=total, desc="geometry alignment probe", unit="trial") as progress:
        for session, k, policy, source_session, do_coral in jobs:
            data = loaded[session]
            target_stats = calibration_window_stats(
                data["neural_features"],
                calibration_trials=k,
                n_components=args.n_components,
                shrinkage=args.cov_shrinkage,
            )
            source = source_stats[source_session]
            input_layer = model_sessions.index(source_session)
            for trial_idx in range(k, len(data["neural_features"])):
                x = data["neural_features"][trial_idx].astype(np.float32, copy=False)
                if do_coral:
                    x = coral_align_features(
                        x,
                        target_mean=target_stats["mean"],
                        target_cov=target_stats["cov"],
                        source_mean=source["mean"],
                        source_cov=source["cov"],
                        eps=args.coral_eps,
                    )
                neural_input = torch.tensor(np.expand_dims(x, axis=0), device=device, dtype=dtype)
                logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)[0]
                pred_ids = greedy_ctc_decode(logits)
                if args.eval_type == "val":
                    true_ids = trim_target_sequence(data["seq_class_ids"][trial_idx], data["seq_len"][trial_idx])
                    edit_dist, n_phonemes, per = phoneme_error_rate(true_ids, pred_ids)
                    true_phonemes = phoneme_ids_to_string(true_ids)
                else:
                    edit_dist, n_phonemes, per = np.nan, np.nan, np.nan
                    true_phonemes = ""
                quality = logits_quality_metrics(logits)
                rows.append(
                    {
                        "calibration_trials": int(k),
                        "session": session,
                        "trial_index_within_session": int(trial_idx),
                        "block": int(data["block_num"][trial_idx]),
                        "trial": int(data["trial_num"][trial_idx]),
                        "policy": policy,
                        "source_session": source_session,
                        "coral_aligned": bool(do_coral),
                        "sentence_label": _as_plain_string(data["sentence_label"][trial_idx]),
                        "true_phonemes": true_phonemes,
                        "pred_phonemes": phoneme_ids_to_string(pred_ids),
                        "edit_distance": edit_dist,
                        "num_phonemes": n_phonemes,
                        "PER": per,
                        **quality,
                    }
                )
                progress.update(1)

    trials = pd.DataFrame(rows)
    summary = session_weighted_summary(trials) if not trials.empty else pd.DataFrame()
    overall = overall_summary(trials) if not trials.empty else pd.DataFrame()
    for path in [args.output_trials, args.output_summary, args.output_overall]:
        path.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_trials, index=False)
    summary.to_csv(args.output_summary, index=False)
    overall.to_csv(args.output_overall, index=False)

    print(f"Loaded official GRU checkpoint on {device}.")
    print(f"Wrote {args.output_trials}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_overall}")
    if not overall.empty:
        printable = overall.copy()
        printable["weighted_PER"] = (100 * printable["weighted_PER"]).map(lambda v: f"{v:.2f}%")
        printable["trial_mean_PER"] = (100 * printable["trial_mean_PER"]).map(lambda v: f"{v:.2f}%")
        print(printable.to_string(index=False))


if __name__ == "__main__":
    main()
