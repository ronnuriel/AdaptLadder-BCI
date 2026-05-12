from __future__ import annotations

import argparse
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


SELECTION_METRICS = [
    "abs_days_from_source",
    "mean_shift_from_source",
    "scale_shift_from_source",
    "diag_cov_shift_from_source",
    "cov_relative_fro_shift_from_source",
    "coral_distance_from_source",
    "mean_principal_angle_deg",
    "subspace_chordal_distance",
    "basis_procrustes_error",
]


def _as_plain_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _set_use_amp(model_args, enabled: bool) -> None:
    try:
        model_args["use_amp"] = enabled
    except TypeError:
        model_args.use_amp = enabled


def _count_trials(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return len(handle.keys())


def trial_lengths(file_path: Path) -> list[tuple[str, int]]:
    lengths = []
    with h5py.File(file_path, "r") as handle:
        for key in handle.keys():
            lengths.append((key, int(handle[key]["input_features"].shape[0])))
    return lengths


def sample_session_frames(file_path: Path, max_frames: int, seed: int) -> np.ndarray:
    lengths = trial_lengths(file_path)
    total_frames = sum(length for _key, length in lengths)
    if total_frames == 0:
        raise ValueError(f"No frames found in {file_path}")

    rng = np.random.default_rng(seed)
    if total_frames <= max_frames:
        selected = np.arange(total_frames)
    else:
        selected = np.sort(rng.choice(total_frames, size=max_frames, replace=False))

    frames = []
    selected_pos = 0
    offset = 0
    with h5py.File(file_path, "r") as handle:
        for key, length in lengths:
            stop = offset + length
            local = []
            while selected_pos < len(selected) and offset <= selected[selected_pos] < stop:
                local.append(int(selected[selected_pos] - offset))
                selected_pos += 1
            if local:
                frames.append(handle[key]["input_features"][local])
            offset = stop

    if not frames:
        raise ValueError(f"No sampled frames gathered from {file_path}")
    return np.concatenate(frames, axis=0).astype(np.float64, copy=False)


def covariance_stats(frames: np.ndarray, shrinkage: float) -> dict[str, np.ndarray | int]:
    mean = frames.mean(axis=0)
    centered = frames - mean
    cov = centered.T @ centered / max(len(frames) - 1, 1)
    if shrinkage > 0:
        diagonal = np.diag(np.diag(cov))
        cov = (1 - shrinkage) * cov + shrinkage * diagonal
    std = np.sqrt(np.maximum(np.diag(cov), 1e-8))
    return {
        "n_frames_sampled": int(frames.shape[0]),
        "mean": mean,
        "std": std,
        "cov": cov,
    }


def top_basis(cov: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvecs[:, :n_components], eigvals[:n_components]


def subspace_metrics(source_basis: np.ndarray, target_basis: np.ndarray) -> dict[str, float]:
    singular_values = np.linalg.svd(source_basis.T @ target_basis, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    angles = np.arccos(singular_values)

    u, _s, vt = np.linalg.svd(target_basis.T @ source_basis)
    rotation = u @ vt
    procrustes_error = np.linalg.norm(target_basis @ rotation - source_basis, ord="fro") / np.sqrt(source_basis.shape[1])

    return {
        "mean_principal_angle_deg": float(np.degrees(np.mean(angles))),
        "subspace_chordal_distance": float(np.sqrt(np.sum(np.sin(angles) ** 2))),
        "basis_procrustes_error": float(procrustes_error),
    }


def discover_sessions(data_dir: Path, split: str, model_sessions: list[str]) -> list[str]:
    sessions = []
    for session in model_sessions:
        if (data_dir / session / f"data_{split}.hdf5").exists():
            sessions.append(session)
    return sessions


def compute_session_stats(
    data_dir: Path,
    sessions: list[str],
    split: str,
    max_frames: int,
    seed: int,
    n_components: int,
    shrinkage: float,
) -> dict[str, dict[str, np.ndarray | int]]:
    stats = {}
    for idx, session in enumerate(tqdm(sessions, desc=f"Sampling {split} geometry", unit="session")):
        file_path = data_dir / session / f"data_{split}.hdf5"
        frames = sample_session_frames(file_path, max_frames=max_frames, seed=seed + idx)
        stats[session] = covariance_stats(frames, shrinkage=shrinkage)
        basis, eigvals = top_basis(stats[session]["cov"], n_components=n_components)
        stats[session]["basis"] = basis
        stats[session]["eigvals"] = eigvals
    return stats


def pairwise_geometry_table(
    stats: dict[str, dict[str, np.ndarray | int]],
    sessions: list[str],
    allow_native: bool,
    source_candidate_mode: str,
) -> pd.DataFrame:
    rows = []
    for target_session in sessions:
        target = stats[target_session]
        target_date = session_date(target_session)
        for source_session in sessions:
            source_date = session_date(source_session)
            if not allow_native and source_session == target_session:
                continue
            if source_candidate_mode == "past-only" and source_date >= target_date:
                continue
            source = stats[source_session]

            cov_delta = target["cov"] - source["cov"]
            source_cov_norm = max(float(np.linalg.norm(source["cov"], ord="fro")), 1e-8)
            row = {
                "target_session": target_session,
                "source_session": source_session,
                "target_date": target_date.isoformat(),
                "source_date": source_date.isoformat(),
                "days_from_source": int((target_date - source_date).days),
                "abs_days_from_source": int(abs((target_date - source_date).days)),
                "target_n_frames_sampled": int(target["n_frames_sampled"]),
                "source_n_frames_sampled": int(source["n_frames_sampled"]),
                "mean_shift_from_source": float(np.linalg.norm(target["mean"] - source["mean"])),
                "scale_shift_from_source": float(np.linalg.norm(target["std"] - source["std"])),
                "diag_cov_shift_from_source": float(
                    np.linalg.norm(np.square(target["std"]) - np.square(source["std"]))
                ),
                "cov_relative_fro_shift_from_source": float(np.linalg.norm(cov_delta, ord="fro") / source_cov_norm),
                "coral_distance_from_source": float(
                    (np.linalg.norm(cov_delta, ord="fro") ** 2) / (4 * target["cov"].shape[0] * target["cov"].shape[0])
                ),
            }
            row.update(subspace_metrics(source["basis"], target["basis"]))
            rows.append(row)
    return pd.DataFrame(rows)


def select_sources(pairwise: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    if selection_metric not in pairwise:
        raise ValueError(f"Unknown selection metric {selection_metric!r}")
    selected = (
        pairwise.sort_values(["target_session", selection_metric, "abs_days_from_source", "source_session"])
        .groupby("target_session", as_index=False)
        .first()
    )
    selected["selection_metric"] = selection_metric
    selected["selection_metric_value"] = selected[selection_metric]
    return selected.sort_values("target_date").reset_index(drop=True)


def weighted_per(trials: pd.DataFrame) -> float:
    return float(trials["edit_distance"].sum() / trials["num_phonemes"].sum())


def summarize_overall(
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    selected_sessions: list[str],
    native_trials_path: Path,
    fixed_middle_trials_path: Path,
) -> pd.DataFrame:
    selected_set = set(selected_sessions)
    rows = [
        {
            "method": "geometry_nearest",
            "weighted_PER": weighted_per(trials),
            "trial_mean_PER": float(trials["PER"].mean()),
            "harmed_sessions_vs_native": np.nan,
            "improved_sessions_vs_native": np.nan,
            "num_sessions": int(summary["session"].nunique()),
        }
    ]
    if native_trials_path.exists():
        native_trials = pd.read_csv(native_trials_path)
        native_trials = native_trials[native_trials["session"].isin(selected_set)]
        native_weighted = weighted_per(native_trials)
        rows.insert(
            0,
            {
                "method": "native-day",
                "weighted_PER": native_weighted,
                "trial_mean_PER": float(native_trials["PER"].mean()),
                "harmed_sessions_vs_native": np.nan,
                "improved_sessions_vs_native": np.nan,
                "num_sessions": int(native_trials["session"].nunique()),
            },
        )
        rows[-1]["delta_vs_native_weighted_PER"] = rows[-1]["weighted_PER"] - native_weighted

    if fixed_middle_trials_path.exists():
        fixed_trials = pd.read_csv(fixed_middle_trials_path)
        fixed_trials = fixed_trials[fixed_trials["session"].isin(selected_set)]
        rows.append(
            {
                "method": "fixed_middle_source",
                "weighted_PER": weighted_per(fixed_trials),
                "trial_mean_PER": float(fixed_trials["PER"].mean()),
                "harmed_sessions_vs_native": np.nan,
                "improved_sessions_vs_native": np.nan,
                "num_sessions": int(fixed_trials["session"].nunique()),
            }
        )
        if native_trials_path.exists():
            rows[-1]["delta_vs_native_weighted_PER"] = rows[-1]["weighted_PER"] - native_weighted

    overall = pd.DataFrame(rows)
    if "delta_vs_native_weighted_PER" not in overall:
        overall["delta_vs_native_weighted_PER"] = np.nan
    if native_trials_path.exists():
        native_weighted = float(overall.loc[overall["method"] == "native-day", "weighted_PER"].iloc[0])
        overall["delta_vs_native_weighted_PER"] = overall["weighted_PER"] - native_weighted
    return overall


def plot_weighted_per(overall: pd.DataFrame, output_path: Path) -> None:
    if overall.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = ["native-day", "fixed_middle_source", "geometry_nearest"]
    plot_frame = overall.set_index("method").loc[[method for method in order if method in set(overall["method"])]].reset_index()
    colors = {
        "native-day": "#2a9d8f",
        "fixed_middle_source": "#457b9d",
        "geometry_nearest": "#7b2cbf",
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    x = np.arange(len(plot_frame))
    ax.bar(x, plot_frame["weighted_PER"], color=[colors.get(method, "#666666") for method in plot_frame["method"]])
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", "\n") for label in plot_frame["method"]])
    ax.set_ylabel("Phoneme-weighted PER")
    ax.set_title("Geometry-based non-native source selection")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate geometry-nearest non-native T15 input-layer source selection.")
    parser.add_argument("--model-path", type=Path, default=Path("data/external/btt-25-gru-pure-baseline-0-0898"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val", "test"], default="val")
    parser.add_argument("--stats-split", default="train")
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="cov_relative_fro_shift_from_source")
    parser.add_argument(
        "--source-candidate-mode",
        choices=["all", "past-only"],
        default="all",
        help="Use all non-native source layers, or only source layers earlier than the target session.",
    )
    parser.add_argument("--allow-native-source", action="store_true")
    parser.add_argument("--max-frames", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--cov-shrinkage", type=float, default=0.05)
    parser.add_argument("--gpu-number", type=int, default=-1)
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--max-trials-per-session", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument(
        "--native-trials",
        type=Path,
        default=Path("results/tables/t15_decoder_probe_val.csv"),
    )
    parser.add_argument(
        "--fixed-middle-trials",
        type=Path,
        default=Path("results/tables/t15_decoder_probe_cross_day_source_2023_11_26_val.csv"),
    )
    parser.add_argument(
        "--output-pairwise",
        type=Path,
        default=Path("results/tables/t15_geometry_source_pairwise_distances.csv"),
    )
    parser.add_argument(
        "--output-selection",
        type=Path,
        default=Path("results/tables/t15_geometry_nearest_source_selection.csv"),
    )
    parser.add_argument(
        "--output-trials",
        type=Path,
        default=Path("results/tables/t15_geometry_nearest_source_trial_results.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/t15_geometry_nearest_source_session_summary.csv"),
    )
    parser.add_argument(
        "--output-overall",
        type=Path,
        default=Path("results/tables/t15_geometry_nearest_source_overall_summary.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/figures/t15_geometry_nearest_source_weighted_per.png"),
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
    stats_sessions = discover_sessions(args.data_dir, args.stats_split, model_sessions)
    sessions = [session for session in eval_sessions if session in stats_sessions]
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]
    if len(sessions) < 2:
        raise ValueError("Geometry source selection requires at least two sessions with eval and stats splits.")

    stats = compute_session_stats(
        data_dir=args.data_dir,
        sessions=sessions,
        split=args.stats_split,
        max_frames=args.max_frames,
        seed=args.seed,
        n_components=args.n_components,
        shrinkage=args.cov_shrinkage,
    )
    pairwise = pairwise_geometry_table(
        stats,
        sessions=sessions,
        allow_native=args.allow_native_source,
        source_candidate_mode=args.source_candidate_mode,
    )
    if pairwise.empty:
        raise ValueError(f"No valid source-target pairs for source candidate mode {args.source_candidate_mode!r}.")
    selection = select_sources(pairwise, args.selection_metric)
    selected_sessions = selection["target_session"].tolist()
    selection_map = dict(zip(selection["target_session"], selection["source_session"], strict=True))

    args.output_pairwise.parent.mkdir(parents=True, exist_ok=True)
    args.output_selection.parent.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(args.output_pairwise, index=False)
    selection.to_csv(args.output_selection, index=False)

    rows = []
    total_trials = sum(_count_trials(args.data_dir / session / f"data_{args.eval_type}.hdf5") for session in selected_sessions)
    if args.max_trials_per_session is not None:
        total_trials = sum(
            min(_count_trials(args.data_dir / session / f"data_{args.eval_type}.hdf5"), args.max_trials_per_session)
            for session in selected_sessions
        )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with tqdm(total=total_trials, desc="geometry-nearest PER", unit="trial") as progress:
        for target_session in selected_sessions:
            source_session = selection_map[target_session]
            input_layer = model_sessions.index(source_session)
            eval_file = args.data_dir / target_session / f"data_{args.eval_type}.hdf5"
            data = load_h5py_file(str(eval_file), b2txt_csv_df)
            n_trials = len(data["neural_features"])
            if args.max_trials_per_session is not None:
                n_trials = min(n_trials, args.max_trials_per_session)

            for trial_idx in range(n_trials):
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
                    true_ids = []
                    edit_dist, n_phonemes, per = np.nan, np.nan, np.nan
                    true_phonemes = ""

                quality = logits_quality_metrics(logits)
                rows.append(
                    {
                        "session": target_session,
                        "block": int(data["block_num"][trial_idx]),
                        "trial": int(data["trial_num"][trial_idx]),
                        "mode": "geometry-nearest-source",
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
    args.output_trials.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_overall.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_trials, index=False)

    summary = (
        trials.groupby(["session", "mode", "input_layer_session", "selection_metric"], as_index=False)
        .agg(
            n_trials=("PER", "size"),
            mean_PER=("PER", "mean"),
            median_PER=("PER", "median"),
            mean_blank_rate=("blank_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_entropy=("entropy", "mean"),
        )
        .merge(
            selection[["target_session", "selection_metric_value", "abs_days_from_source"]],
            left_on="session",
            right_on="target_session",
            how="left",
        )
        .drop(columns=["target_session"])
    )
    summary.to_csv(args.output_summary, index=False)

    overall = summarize_overall(
        trials=trials,
        summary=summary,
        selected_sessions=selected_sessions,
        native_trials_path=args.native_trials,
        fixed_middle_trials_path=args.fixed_middle_trials,
    )
    overall.to_csv(args.output_overall, index=False)
    plot_weighted_per(overall, args.output_figure)

    print(f"Loaded official GRU checkpoint on {device}.")
    print(f"Selected non-native sources using {args.selection_metric}.")
    print(f"Source candidate mode: {args.source_candidate_mode}. Evaluated {len(selected_sessions)} target sessions.")
    print(f"Wrote {args.output_pairwise}")
    print(f"Wrote {args.output_selection}")
    print(f"Wrote {args.output_trials}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_overall}")
    print(f"Wrote {args.output_figure}")
    if args.eval_type == "val" and not trials.empty:
        print(f"Geometry-nearest phoneme-weighted PER: {weighted_per(trials):.4f}")
        print(f"Geometry-nearest trial-mean PER: {trials['PER'].mean():.4f}")
    print(overall.to_string(index=False))
    print(selection[["target_session", "source_session", "selection_metric_value", "abs_days_from_source"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
