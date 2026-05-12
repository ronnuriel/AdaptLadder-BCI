from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t15_geometry_source_selection_eval import (
    SELECTION_METRICS,
    _as_plain_string,
    _set_use_amp,
    compute_session_stats,
    covariance_stats,
    discover_sessions,
    pairwise_geometry_table,
    select_sources,
    top_basis,
    weighted_per,
)
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


def calibration_window_stats(
    neural_features: list[np.ndarray],
    calibration_trials: int,
    n_components: int,
    shrinkage: float,
) -> dict[str, np.ndarray | int]:
    if len(neural_features) < calibration_trials:
        raise ValueError(f"Need {calibration_trials} calibration trials, found {len(neural_features)}")
    frames = np.concatenate(neural_features[:calibration_trials], axis=0).astype(np.float64, copy=False)
    stats = covariance_stats(frames, shrinkage=shrinkage)
    basis, eigvals = top_basis(stats["cov"], n_components=n_components)
    stats["basis"] = basis
    stats["eigvals"] = eigvals
    return stats


def build_kshot_selection(
    source_stats: dict[str, dict[str, np.ndarray | int]],
    target_session: str,
    target_stats: dict[str, np.ndarray | int],
    source_sessions: list[str],
    source_candidate_mode: str,
    selection_metric: str,
) -> pd.Series | None:
    stats = dict(source_stats)
    stats[target_session] = target_stats
    pairwise = pairwise_geometry_table(
        stats,
        sessions=[target_session, *source_sessions],
        allow_native=False,
        source_candidate_mode=source_candidate_mode,
    )
    pairwise = pairwise[pairwise["target_session"] == target_session].copy()
    if pairwise.empty:
        return None
    return select_sources(pairwise, selection_metric).iloc[0]


def add_trial_indices(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["trial_index_within_session"] = frame.groupby("session").cumcount()
    return frame


def subset_existing_trials(path: Path, selected: pd.DataFrame) -> pd.DataFrame | None:
    if not path.exists():
        return None
    trials = add_trial_indices(pd.read_csv(path))
    keep = selected[["session", "calibration_trials"]]
    merged = trials.merge(keep, on="session", how="inner")
    return merged[merged["trial_index_within_session"] >= merged["calibration_trials"]].copy()


def overall_rows_from_trials(method: str, trials: pd.DataFrame, calibration_trials: int) -> dict[str, float | int | str]:
    return {
        "calibration_trials": int(calibration_trials),
        "method": method,
        "weighted_PER": weighted_per(trials),
        "trial_mean_PER": float(trials["PER"].mean()),
        "num_sessions": int(trials["session"].nunique()),
        "num_trials": int(len(trials)),
    }


def summarize_overall(
    kshot_trials: pd.DataFrame,
    selected: pd.DataFrame,
    native_trials_path: Path,
    fixed_middle_trials_path: Path,
    full_geometry_trials_path: Path | None,
) -> pd.DataFrame:
    rows = []
    for k, frame in kshot_trials.groupby("calibration_trials"):
        selected_k = selected[selected["calibration_trials"] == k]
        native = subset_existing_trials(native_trials_path, selected_k)
        fixed = subset_existing_trials(fixed_middle_trials_path, selected_k)
        full_geometry = subset_existing_trials(full_geometry_trials_path, selected_k) if full_geometry_trials_path else None

        if native is not None and not native.empty:
            rows.append(overall_rows_from_trials("native-day", native, int(k)))
        if fixed is not None and not fixed.empty:
            rows.append(overall_rows_from_trials("fixed_middle_source", fixed, int(k)))
        if full_geometry is not None and not full_geometry.empty:
            rows.append(overall_rows_from_trials("full_session_geometry_nearest", full_geometry, int(k)))
        rows.append(overall_rows_from_trials("kshot_geometry_nearest", frame, int(k)))

    overall = pd.DataFrame(rows)
    native_by_k = overall[overall["method"] == "native-day"][["calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "native_weighted_PER"}
    )
    fixed_by_k = overall[overall["method"] == "fixed_middle_source"][["calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "fixed_middle_weighted_PER"}
    )
    overall = overall.merge(native_by_k, on="calibration_trials", how="left")
    overall = overall.merge(fixed_by_k, on="calibration_trials", how="left")
    overall["delta_vs_native_weighted_PER"] = overall["weighted_PER"] - overall["native_weighted_PER"]
    overall["gain_vs_fixed_middle_weighted_PER"] = overall["fixed_middle_weighted_PER"] - overall["weighted_PER"]
    gap = overall["fixed_middle_weighted_PER"] - overall["native_weighted_PER"]
    overall["recovery_fraction_vs_fixed_middle"] = overall["gain_vs_fixed_middle_weighted_PER"] / gap.replace(0, np.nan)
    return overall


def session_summary(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["calibration_trials", "session", "mode", "input_layer_session", "selection_metric"], as_index=False)
        .agg(
            n_trials=("PER", "size"),
            mean_PER=("PER", "mean"),
            median_PER=("PER", "median"),
            mean_blank_rate=("blank_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_entropy=("entropy", "mean"),
        )
    )


def plot_weighted_per(overall: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_order = [
        "native-day",
        "fixed_middle_source",
        "full_session_geometry_nearest",
        "kshot_geometry_nearest",
    ]
    colors = {
        "native-day": "#2a9d8f",
        "fixed_middle_source": "#457b9d",
        "full_session_geometry_nearest": "#7b2cbf",
        "kshot_geometry_nearest": "#b23a48",
    }
    ks = sorted(overall["calibration_trials"].unique())
    x = np.arange(len(ks), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(method_order))

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for offset, method in zip(offsets, method_order, strict=True):
        values = []
        for k in ks:
            row = overall[(overall["calibration_trials"] == k) & (overall["method"] == method)]
            values.append(float(row["weighted_PER"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=method.replace("_", " "), color=colors[method])

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={int(k)}" for k in ks])
    ax.set_ylabel("Phoneme-weighted PER on remaining trials")
    ax.set_title("Beginning-of-day geometry source selection")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate K-shot beginning-of-day geometry source selection for T15.")
    parser.add_argument("--model-path", type=Path, default=Path("data/external/btt-25-gru-pure-baseline-0-0898"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val", "test"], default="val")
    parser.add_argument("--source-stats-split", default="train")
    parser.add_argument("--calibration-trials", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--min-eval-trials", type=int, default=1)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="cov_relative_fro_shift_from_source")
    parser.add_argument(
        "--source-candidate-mode",
        choices=["all", "past-only"],
        default="past-only",
        help="Use all non-native source layers, or only source layers earlier than the target session.",
    )
    parser.add_argument("--max-source-frames", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--cov-shrinkage", type=float, default=0.05)
    parser.add_argument("--gpu-number", type=int, default=-1)
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--max-trials-per-session", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--native-trials", type=Path, default=Path("results/tables/t15_decoder_probe_val.csv"))
    parser.add_argument(
        "--fixed-middle-trials",
        type=Path,
        default=Path("results/tables/t15_decoder_probe_cross_day_source_2023_11_26_val.csv"),
    )
    parser.add_argument(
        "--full-geometry-trials",
        type=Path,
        default=Path("results/tables/t15_geometry_nearest_past_source_trial_results.csv"),
    )
    parser.add_argument(
        "--output-selection",
        type=Path,
        default=Path("results/tables/t15_kshot_geometry_source_selection.csv"),
    )
    parser.add_argument(
        "--output-trials",
        type=Path,
        default=Path("results/tables/t15_kshot_geometry_source_trial_results.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/t15_kshot_geometry_source_session_summary.csv"),
    )
    parser.add_argument(
        "--output-overall",
        type=Path,
        default=Path("results/tables/t15_kshot_geometry_source_overall_summary.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/figures/t15_kshot_geometry_source_weighted_per.png"),
    )
    args = parser.parse_args()

    device = resolve_device(args.gpu_number)
    model, model_args = load_official_gru_decoder(ROOT, args.model_path, device)
    _set_use_amp(model_args, enabled=(device.type == "cuda" and bool(model_args["use_amp"])))
    add_official_model_training_to_path(ROOT)
    from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep

    b2txt_csv_df = pd.read_csv(args.csv_path)
    model_sessions = list(model_args["dataset"]["sessions"])
    eval_sessions = discover_sessions(args.data_dir, args.eval_type, model_sessions)
    source_sessions = discover_sessions(args.data_dir, args.source_stats_split, model_sessions)
    sessions = [session for session in eval_sessions if session in source_sessions]
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]
        source_sessions = [session for session in source_sessions if session in sessions]
    if len(sessions) < 2:
        raise ValueError("K-shot geometry source selection requires at least two sessions.")

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
    rows = []
    selection_rows = []
    total_eval_trials = 0
    loaded_data = {}
    for session in sessions:
        data = load_h5py_file(str(args.data_dir / session / f"data_{args.eval_type}.hdf5"), b2txt_csv_df)
        loaded_data[session] = data
        n_trials = len(data["neural_features"])
        if args.max_trials_per_session is not None:
            n_trials = min(n_trials, args.max_trials_per_session)
            data["neural_features"] = data["neural_features"][:n_trials]
        for k in args.calibration_trials:
            if n_trials >= k + args.min_eval_trials:
                target_date = session_date(session)
                if args.source_candidate_mode == "past-only" and not any(
                    session_date(source_session) < target_date for source_session in source_sessions
                ):
                    continue
                total_eval_trials += n_trials - k

    with tqdm(total=total_eval_trials, desc="k-shot geometry PER", unit="trial") as progress:
        for session in sessions:
            data = loaded_data[session]
            n_trials = len(data["neural_features"])
            for k in args.calibration_trials:
                if n_trials < k + args.min_eval_trials:
                    continue
                target_date = session_date(session)
                candidate_sources = [
                    source_session
                    for source_session in source_sessions
                    if source_session != session
                    and (
                        args.source_candidate_mode == "all"
                        or session_date(source_session) < target_date
                    )
                ]
                if not candidate_sources:
                    continue

                target_stats = calibration_window_stats(
                    data["neural_features"],
                    calibration_trials=k,
                    n_components=args.n_components,
                    shrinkage=args.cov_shrinkage,
                )
                selected = build_kshot_selection(
                    source_stats=source_stats,
                    target_session=session,
                    target_stats=target_stats,
                    source_sessions=candidate_sources,
                    source_candidate_mode=args.source_candidate_mode,
                    selection_metric=args.selection_metric,
                )
                if selected is None:
                    continue
                source_session = str(selected["source_session"])
                input_layer = model_sessions.index(source_session)
                selection_rows.append(
                    {
                        "calibration_trials": int(k),
                        "target_session": session,
                        "source_session": source_session,
                        "target_date": session_date(session).isoformat(),
                        "source_date": session_date(source_session).isoformat(),
                        "eval_trials": int(n_trials - k),
                        "target_calibration_frames": int(target_stats["n_frames_sampled"]),
                        "selection_metric": args.selection_metric,
                        "selection_metric_value": float(selected["selection_metric_value"]),
                        "abs_days_from_source": int(selected["abs_days_from_source"]),
                    }
                )

                for trial_idx in range(k, n_trials):
                    neural_input = np.expand_dims(data["neural_features"][trial_idx], axis=0)
                    neural_input = torch.tensor(neural_input, device=device, dtype=dtype)
                    logits = runSingleDecodingStep(
                        neural_input,
                        input_layer,
                        model,
                        model_args,
                        device,
                    )[0]
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
                            "mode": "kshot-geometry-nearest-source",
                            "input_layer_session": source_session,
                            "selection_metric": args.selection_metric,
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
    selected = pd.DataFrame(selection_rows)
    summary = session_summary(trials) if not trials.empty else pd.DataFrame()
    if not summary.empty:
        summary = summary.merge(
            selected[
                [
                    "calibration_trials",
                    "target_session",
                    "source_session",
                    "target_calibration_frames",
                    "selection_metric_value",
                    "abs_days_from_source",
                ]
            ],
            left_on=["calibration_trials", "session", "input_layer_session"],
            right_on=["calibration_trials", "target_session", "source_session"],
            how="left",
        ).drop(columns=["target_session", "source_session"])
    overall = summarize_overall(
        kshot_trials=trials,
        selected=selected.rename(columns={"target_session": "session"}),
        native_trials_path=args.native_trials,
        fixed_middle_trials_path=args.fixed_middle_trials,
        full_geometry_trials_path=args.full_geometry_trials,
    )

    args.output_selection.parent.mkdir(parents=True, exist_ok=True)
    args.output_trials.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_overall.parent.mkdir(parents=True, exist_ok=True)
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_selection, index=False)
    trials.to_csv(args.output_trials, index=False)
    summary.to_csv(args.output_summary, index=False)
    overall.to_csv(args.output_overall, index=False)
    plot_weighted_per(overall, args.output_figure)

    print(f"Loaded official GRU checkpoint on {device}.")
    print(f"Source candidate mode: {args.source_candidate_mode}.")
    print(f"Wrote {args.output_selection}")
    print(f"Wrote {args.output_trials}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_overall}")
    print(f"Wrote {args.output_figure}")
    print(overall.to_string(index=False))
    print(selected.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
