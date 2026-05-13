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
    discover_sessions,
)
from scripts.run_t15_kshot_geometry_source_selection import (
    build_kshot_selection,
    calibration_window_stats,
    overall_rows_from_trials,
    subset_existing_trials,
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


def policy_name(library_size: int | None) -> str:
    if library_size == 1:
        return "previous_source"
    if library_size is None:
        return "all_past"
    return f"last_{library_size}"


def candidate_sources_for_policy(source_sessions: list[str], target_session: str, library_size: int | None) -> list[str]:
    target_date = session_date(target_session)
    past = sorted(
        [session for session in source_sessions if session != target_session and session_date(session) < target_date],
        key=session_date,
    )
    if library_size is None:
        return past
    return past[-library_size:]


def choose_source_for_policy(
    source_stats: dict[str, dict[str, np.ndarray | int]],
    target_session: str,
    target_stats: dict[str, np.ndarray | int],
    candidate_sources: list[str],
    selection_metric: str,
) -> pd.Series | None:
    if not candidate_sources:
        return None
    selected = build_kshot_selection(
        source_stats=source_stats,
        target_session=target_session,
        target_stats=target_stats,
        source_sessions=candidate_sources,
        source_candidate_mode="all",
        selection_metric=selection_metric,
    )
    return selected


def build_overall(
    trials: pd.DataFrame,
    selection: pd.DataFrame,
    native_trials_path: Path,
    fixed_middle_trials_path: Path,
) -> pd.DataFrame:
    rows = []
    for k, selected_k in selection.groupby("calibration_trials"):
        selected_for_subset = selected_k[["target_session", "calibration_trials"]].drop_duplicates().rename(
            columns={"target_session": "session"}
        )
        native = subset_existing_trials(native_trials_path, selected_for_subset)
        fixed = subset_existing_trials(fixed_middle_trials_path, selected_for_subset)
        if native is not None and not native.empty:
            rows.append(overall_rows_from_trials("native-day", native, int(k)))
        if fixed is not None and not fixed.empty:
            rows.append(overall_rows_from_trials("fixed_middle_source", fixed, int(k)))
        for method, frame in trials[trials["calibration_trials"] == k].groupby("library_policy"):
            rows.append(overall_rows_from_trials(str(method), frame, int(k)))

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
    order = {
        "native-day": 0,
        "fixed_middle_source": 1,
        "previous_source": 2,
        "last_3": 3,
        "last_5": 4,
        "all_past": 5,
    }
    overall["method_order"] = overall["method"].map(order).fillna(99)
    return overall.sort_values(["calibration_trials", "method_order"]).drop(columns=["method_order"]).reset_index(drop=True)


def library_session_summary(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    return (
        trials.groupby(["calibration_trials", "library_policy", "session", "mode", "input_layer_session"], as_index=False)
        .agg(
            n_trials=("PER", "size"),
            mean_PER=("PER", "mean"),
            median_PER=("PER", "median"),
            mean_blank_rate=("blank_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_entropy=("entropy", "mean"),
        )
    )


def plot_library_size(overall: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_order = ["native-day", "fixed_middle_source", "previous_source", "last_3", "last_5", "all_past"]
    colors = {
        "native-day": "#2a9d8f",
        "fixed_middle_source": "#457b9d",
        "previous_source": "#e76f51",
        "last_3": "#f4a261",
        "last_5": "#7b2cbf",
        "all_past": "#b23a48",
    }
    ks = sorted(overall["calibration_trials"].unique())
    x = np.arange(len(ks), dtype=float)
    width = 0.12
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(method_order))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset, method in zip(offsets, method_order, strict=True):
        values = []
        for k in ks:
            row = overall[(overall["calibration_trials"] == k) & (overall["method"] == method)]
            values.append(float(row["weighted_PER"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=method.replace("_", " "), color=colors[method])

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={int(k)}" for k in ks])
    ax.set_ylabel("Phoneme-weighted PER on remaining trials")
    ax.set_title("K-shot input-layer library-size ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate how many past T15 input layers are needed for K-shot source selection.")
    parser.add_argument("--model-path", type=Path, default=Path("data/external/btt-25-gru-pure-baseline-0-0898"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val", "test"], default="val")
    parser.add_argument("--source-stats-split", default="train")
    parser.add_argument("--calibration-trials", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--library-sizes", nargs="+", default=["1", "3", "5", "all"])
    parser.add_argument("--min-eval-trials", type=int, default=1)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="cov_relative_fro_shift_from_source")
    parser.add_argument("--max-source-frames", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--cov-shrinkage", type=float, default=0.05)
    parser.add_argument("--gpu-number", type=int, default=-1)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--native-trials", type=Path, default=Path("results/tables/t15_decoder_probe_val.csv"))
    parser.add_argument(
        "--fixed-middle-trials",
        type=Path,
        default=Path("results/tables/t15_decoder_probe_cross_day_source_2023_11_26_val.csv"),
    )
    parser.add_argument(
        "--output-selection",
        type=Path,
        default=Path("results/tables/t15_kshot_library_size_selection.csv"),
    )
    parser.add_argument(
        "--output-trials",
        type=Path,
        default=Path("results/tables/t15_kshot_library_size_trial_results.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/t15_kshot_library_size_session_summary.csv"),
    )
    parser.add_argument(
        "--output-overall",
        type=Path,
        default=Path("results/tables/t15_kshot_library_size_summary.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/figures/t15_kshot_library_size_per.png"),
    )
    args = parser.parse_args()

    library_sizes: list[int | None] = []
    for raw in args.library_sizes:
        library_sizes.append(None if str(raw).lower() in {"all", "all_past"} else int(raw))

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
        raise ValueError("Library-size ablation requires at least two sessions.")

    source_stats = compute_session_stats(
        data_dir=args.data_dir,
        sessions=source_sessions,
        split=args.source_stats_split,
        max_frames=args.max_source_frames,
        seed=args.seed,
        n_components=args.n_components,
        shrinkage=args.cov_shrinkage,
    )

    loaded_data = {}
    total_decode_jobs = 0
    planned: list[dict[str, object]] = []
    for session in sessions:
        data = load_h5py_file(str(args.data_dir / session / f"data_{args.eval_type}.hdf5"), b2txt_csv_df)
        loaded_data[session] = data
        n_trials = len(data["neural_features"])
        for k in args.calibration_trials:
            if n_trials < k + args.min_eval_trials:
                continue
            target_stats = calibration_window_stats(
                data["neural_features"],
                calibration_trials=k,
                n_components=args.n_components,
                shrinkage=args.cov_shrinkage,
            )
            selections_for_k = []
            for library_size in library_sizes:
                candidates = candidate_sources_for_policy(source_sessions, session, library_size)
                selected = choose_source_for_policy(source_stats, session, target_stats, candidates, args.selection_metric)
                if selected is None:
                    continue
                selections_for_k.append(
                    {
                        "library_size": library_size,
                        "library_policy": policy_name(library_size),
                        "candidate_count": len(candidates),
                        "selected": selected,
                    }
                )
            unique_sources = sorted({str(item["selected"]["source_session"]) for item in selections_for_k})
            total_decode_jobs += (n_trials - k) * len(unique_sources)
            planned.append(
                {
                    "session": session,
                    "k": k,
                    "n_trials": n_trials,
                    "target_stats": target_stats,
                    "selections": selections_for_k,
                    "unique_sources": unique_sources,
                }
            )

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    trial_cache: dict[tuple[str, int, str], dict[str, object]] = {}
    rows = []
    selection_rows = []
    with tqdm(total=total_decode_jobs, desc="library-size PER", unit="trial") as progress:
        for plan in planned:
            session = str(plan["session"])
            k = int(plan["k"])
            n_trials = int(plan["n_trials"])
            data = loaded_data[session]
            for source_session in plan["unique_sources"]:
                input_layer = model_sessions.index(source_session)
                for trial_idx in range(k, n_trials):
                    neural_input = np.expand_dims(data["neural_features"][trial_idx], axis=0)
                    neural_input = torch.tensor(neural_input, device=device, dtype=dtype)
                    logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)[0]
                    pred_ids = greedy_ctc_decode(logits)
                    if args.eval_type == "val":
                        true_ids = trim_target_sequence(data["seq_class_ids"][trial_idx], data["seq_len"][trial_idx])
                        edit_dist, n_phonemes, per = phoneme_error_rate(true_ids, pred_ids)
                        true_phonemes = phoneme_ids_to_string(true_ids)
                    else:
                        edit_dist, n_phonemes, per = np.nan, np.nan, np.nan
                        true_phonemes = ""
                    trial_cache[(session, trial_idx, source_session)] = {
                        "calibration_trials": k,
                        "session": session,
                        "trial_index_within_session": int(trial_idx),
                        "block": int(data["block_num"][trial_idx]),
                        "trial": int(data["trial_num"][trial_idx]),
                        "input_layer_session": source_session,
                        "sentence_label": _as_plain_string(data["sentence_label"][trial_idx]),
                        "true_phonemes": true_phonemes,
                        "pred_phonemes": phoneme_ids_to_string(pred_ids),
                        "edit_distance": edit_dist,
                        "num_phonemes": n_phonemes,
                        "PER": per,
                        **logits_quality_metrics(logits),
                    }
                    progress.update(1)

            for item in plan["selections"]:
                selected = item["selected"]
                source_session = str(selected["source_session"])
                policy = str(item["library_policy"])
                selection_rows.append(
                    {
                        "calibration_trials": k,
                        "target_session": session,
                        "library_policy": policy,
                        "library_size": "all" if item["library_size"] is None else int(item["library_size"]),
                        "candidate_count": int(item["candidate_count"]),
                        "source_session": source_session,
                        "target_date": session_date(session).isoformat(),
                        "source_date": session_date(source_session).isoformat(),
                        "eval_trials": int(n_trials - k),
                        "target_calibration_frames": int(plan["target_stats"]["n_frames_sampled"]),
                        "selection_metric": args.selection_metric,
                        "selection_metric_value": float(selected["selection_metric_value"]),
                        "abs_days_from_source": int(selected["abs_days_from_source"]),
                    }
                )
                for trial_idx in range(k, n_trials):
                    cached = dict(trial_cache[(session, trial_idx, source_session)])
                    cached["mode"] = "kshot-library-size"
                    cached["library_policy"] = policy
                    cached["selection_metric"] = args.selection_metric
                    rows.append(cached)

    trials = pd.DataFrame(rows)
    selection = pd.DataFrame(selection_rows)
    summary = library_session_summary(trials)
    overall = build_overall(
        trials=trials,
        selection=selection,
        native_trials_path=args.native_trials,
        fixed_middle_trials_path=args.fixed_middle_trials,
    )

    for path in [args.output_selection, args.output_trials, args.output_summary, args.output_overall, args.output_figure]:
        path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output_selection, index=False)
    trials.to_csv(args.output_trials, index=False)
    summary.to_csv(args.output_summary, index=False)
    overall.to_csv(args.output_overall, index=False)
    plot_library_size(overall, args.output_figure)

    print(f"Loaded official GRU checkpoint on {device}.")
    print(f"Wrote {args.output_selection}")
    print(f"Wrote {args.output_trials}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_overall}")
    print(f"Wrote {args.output_figure}")
    print(overall.to_string(index=False))
    print(selection.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
