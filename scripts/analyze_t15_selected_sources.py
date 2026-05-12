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

from scripts.run_t15_geometry_source_selection_eval import _as_plain_string, _set_use_amp, weighted_per
from scripts.run_t15_kshot_geometry_source_selection import subset_existing_trials
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


def previous_session_for(target_session: str, ordered_sessions: list[str]) -> str | None:
    target_dt = session_date(target_session)
    previous = [session for session in ordered_sessions if session_date(session) < target_dt]
    return previous[-1] if previous else None


def add_selection_metadata(selection: pd.DataFrame, ordered_sessions: list[str]) -> pd.DataFrame:
    order_index = {session: idx for idx, session in enumerate(ordered_sessions)}
    rows = []
    for row in selection.to_dict(orient="records"):
        target = row["target_session"]
        selected = row["source_session"]
        previous = previous_session_for(target, ordered_sessions)
        target_idx = order_index.get(target, np.nan)
        selected_idx = order_index.get(selected, np.nan)
        previous_idx = order_index.get(previous, np.nan) if previous is not None else np.nan
        rows.append(
            {
                **row,
                "previous_session": previous or "",
                "previous_date": session_date(previous).isoformat() if previous is not None else "",
                "previous_abs_days": int(abs((session_date(target) - session_date(previous)).days))
                if previous is not None
                else np.nan,
                "selected_is_previous": bool(previous is not None and selected == previous),
                "selected_lag_days": int((session_date(target) - session_date(selected)).days),
                "selected_lag_sessions": int(target_idx - selected_idx) if np.isfinite(target_idx) and np.isfinite(selected_idx) else np.nan,
                "previous_lag_sessions": int(target_idx - previous_idx) if np.isfinite(target_idx) and np.isfinite(previous_idx) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def decode_previous_source_trials(
    selection: pd.DataFrame,
    model,
    model_args,
    model_sessions: list[str],
    data_dir: Path,
    csv_path: Path,
    eval_type: str,
    device: torch.device,
    max_trials_per_session: int | None,
) -> pd.DataFrame:
    add_official_model_training_to_path(ROOT)
    from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep

    b2txt_csv_df = pd.read_csv(csv_path)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rows = []
    total_trials = 0
    for row in selection.to_dict(orient="records"):
        if not row.get("previous_session", ""):
            continue
        if max_trials_per_session is None:
            total_trials += int(row["eval_trials"])
        else:
            k = int(row["calibration_trials"])
            total_trials += max(0, min(int(row["num_trials"]), max_trials_per_session) - k)
    data_cache = {}

    with tqdm(total=total_trials, desc="previous-source PER", unit="trial") as progress:
        for row in selection.to_dict(orient="records"):
            previous_session = row.get("previous_session", "")
            if not previous_session:
                continue
            target_session = row["target_session"]
            k = int(row["calibration_trials"])
            if target_session not in data_cache:
                data = load_h5py_file(str(data_dir / target_session / f"data_{eval_type}.hdf5"), b2txt_csv_df)
                if max_trials_per_session is not None:
                    n = min(len(data["neural_features"]), max_trials_per_session)
                    for key, value in list(data.items()):
                        if isinstance(value, list):
                            data[key] = value[:n]
                data_cache[target_session] = data
            data = data_cache[target_session]
            input_layer = model_sessions.index(previous_session)
            for trial_idx in range(k, len(data["neural_features"])):
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
                if eval_type == "val":
                    true_ids = trim_target_sequence(data["seq_class_ids"][trial_idx], data["seq_len"][trial_idx])
                    edit_dist, n_phonemes, per = phoneme_error_rate(true_ids, pred_ids)
                    true_phonemes = phoneme_ids_to_string(true_ids)
                else:
                    edit_dist, n_phonemes, per = np.nan, np.nan, np.nan
                    true_phonemes = ""
                quality = logits_quality_metrics(logits)
                rows.append(
                    {
                        "calibration_trials": k,
                        "session": target_session,
                        "trial_index_within_session": int(trial_idx),
                        "block": int(data["block_num"][trial_idx]),
                        "trial": int(data["trial_num"][trial_idx]),
                        "mode": "previous-source",
                        "input_layer_session": previous_session,
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
    return pd.DataFrame(rows)


def summarize_sessions(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["calibration_trials", "session", "mode", "input_layer_session"], as_index=False)
        .agg(
            n_trials=("PER", "size"),
            mean_PER=("PER", "mean"),
            median_PER=("PER", "median"),
            mean_blank_rate=("blank_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_entropy=("entropy", "mean"),
        )
    )


def method_row(method: str, trials: pd.DataFrame, k: int) -> dict[str, float | int | str]:
    return {
        "calibration_trials": int(k),
        "method": method,
        "weighted_PER": weighted_per(trials),
        "trial_mean_PER": float(trials["PER"].mean()),
        "num_sessions": int(trials["session"].nunique()),
        "num_trials": int(len(trials)),
    }


def build_overall_summary(
    selection: pd.DataFrame,
    native_trials_path: Path,
    fixed_middle_trials_path: Path,
    kshot_trials_path: Path,
    previous_trials: pd.DataFrame,
) -> pd.DataFrame:
    kshot = pd.read_csv(kshot_trials_path)
    rows = []
    for k, selected_k in selection.groupby("calibration_trials"):
        native = subset_existing_trials(native_trials_path, selected_k.rename(columns={"target_session": "session"}))
        fixed = subset_existing_trials(fixed_middle_trials_path, selected_k.rename(columns={"target_session": "session"}))
        kshot_k = kshot[kshot["calibration_trials"] == k]
        previous_k = previous_trials[previous_trials["calibration_trials"] == k]
        if native is not None and not native.empty:
            rows.append(method_row("native-day", native, int(k)))
        if fixed is not None and not fixed.empty:
            rows.append(method_row("fixed_middle_source", fixed, int(k)))
        if not previous_k.empty:
            rows.append(method_row("previous_source", previous_k, int(k)))
        if not kshot_k.empty:
            rows.append(method_row("kshot_geometry_nearest", kshot_k, int(k)))

    overall = pd.DataFrame(rows)
    native_by_k = overall[overall["method"] == "native-day"][["calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "native_weighted_PER"}
    )
    fixed_by_k = overall[overall["method"] == "fixed_middle_source"][["calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "fixed_middle_weighted_PER"}
    )
    previous_by_k = overall[overall["method"] == "previous_source"][["calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "previous_weighted_PER"}
    )
    overall = overall.merge(native_by_k, on="calibration_trials", how="left")
    overall = overall.merge(fixed_by_k, on="calibration_trials", how="left")
    overall = overall.merge(previous_by_k, on="calibration_trials", how="left")
    overall["delta_vs_native_weighted_PER"] = overall["weighted_PER"] - overall["native_weighted_PER"]
    overall["gain_vs_fixed_middle_weighted_PER"] = overall["fixed_middle_weighted_PER"] - overall["weighted_PER"]
    overall["gain_vs_previous_weighted_PER"] = overall["previous_weighted_PER"] - overall["weighted_PER"]
    gap = overall["fixed_middle_weighted_PER"] - overall["native_weighted_PER"]
    overall["recovery_fraction_vs_fixed_middle"] = overall["gain_vs_fixed_middle_weighted_PER"] / gap.replace(0, np.nan)
    return overall


def build_selection_summary(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, frame in selection.groupby("calibration_trials"):
        rows.append(
            {
                "calibration_trials": int(k),
                "num_sessions": int(len(frame)),
                "selected_previous_sessions": int(frame["selected_is_previous"].sum()),
                "selected_previous_fraction": float(frame["selected_is_previous"].mean()),
                "median_selected_lag_days": float(frame["selected_lag_days"].median()),
                "mean_selected_lag_days": float(frame["selected_lag_days"].mean()),
                "median_selected_lag_sessions": float(frame["selected_lag_sessions"].median()),
                "mean_selected_lag_sessions": float(frame["selected_lag_sessions"].mean()),
                "median_previous_lag_days": float(frame["previous_abs_days"].median()),
                "mean_previous_lag_days": float(frame["previous_abs_days"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_session_comparison(
    selection: pd.DataFrame,
    kshot_trials_path: Path,
    previous_summary: pd.DataFrame,
) -> pd.DataFrame:
    kshot_trials = pd.read_csv(kshot_trials_path)
    geometry_summary = summarize_sessions(kshot_trials)
    geometry_summary = geometry_summary[
        ["calibration_trials", "session", "input_layer_session", "n_trials", "mean_PER", "median_PER"]
    ].rename(
        columns={
            "input_layer_session": "geometry_source_session",
            "n_trials": "geometry_n_trials",
            "mean_PER": "geometry_mean_PER",
            "median_PER": "geometry_median_PER",
        }
    )
    previous_summary = previous_summary[
        ["calibration_trials", "session", "input_layer_session", "n_trials", "mean_PER", "median_PER"]
    ].rename(
        columns={
            "input_layer_session": "previous_source_session",
            "n_trials": "previous_n_trials",
            "mean_PER": "previous_mean_PER",
            "median_PER": "previous_median_PER",
        }
    )
    selection_cols = [
        "calibration_trials",
        "target_session",
        "source_session",
        "previous_session",
        "selected_is_previous",
        "selected_lag_days",
        "selected_lag_sessions",
        "previous_abs_days",
        "previous_lag_sessions",
        "selection_metric",
        "selection_metric_value",
    ]
    metadata = selection[selection_cols].rename(columns={"target_session": "session", "source_session": "selected_source_session"})
    comparison = geometry_summary.merge(previous_summary, on=["calibration_trials", "session"], how="inner")
    comparison = comparison.merge(metadata, on=["calibration_trials", "session"], how="left")
    comparison["geometry_minus_previous_mean_PER"] = comparison["geometry_mean_PER"] - comparison["previous_mean_PER"]
    eps = 1e-12
    comparison["geometry_better_than_previous"] = comparison["geometry_minus_previous_mean_PER"] < -eps
    comparison["previous_better_than_geometry"] = comparison["geometry_minus_previous_mean_PER"] > eps
    return comparison.sort_values(["calibration_trials", "session"]).reset_index(drop=True)


def plot_lag_histogram(selection: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(selection["calibration_trials"].unique())
    fig, axes = plt.subplots(len(ks), 1, figsize=(7.2, 2.3 * len(ks)), sharex=True)
    if len(ks) == 1:
        axes = [axes]
    bins = np.arange(0, max(selection["selected_lag_days"].max(), 1) + 8, 7)
    for ax, k in zip(axes, ks, strict=True):
        frame = selection[selection["calibration_trials"] == k]
        ax.hist(frame["selected_lag_days"], bins=bins, color="#7b2cbf", alpha=0.82)
        ax.axvline(frame["previous_abs_days"].median(), color="0.2", linestyle="--", linewidth=1.1, label="median previous lag")
        ax.set_ylabel(f"K={int(k)}\ncount")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Selected source lag in days")
    axes[0].legend(frameon=False)
    fig.suptitle("Geometry-selected past source lags")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_previous_vs_geometry(overall: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    methods = ["native-day", "fixed_middle_source", "previous_source", "kshot_geometry_nearest"]
    labels = ["Native", "Fixed middle", "Previous", "Geometry"]
    colors = ["#2a9d8f", "#457b9d", "#f4a261", "#b23a48"]
    ks = sorted(overall["calibration_trials"].unique())
    x = np.arange(len(ks), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(methods))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for method, label, color, offset in zip(methods, labels, colors, offsets, strict=True):
        values = []
        for k in ks:
            row = overall[(overall["calibration_trials"] == k) & (overall["method"] == method)]
            values.append(float(row["weighted_PER"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={int(k)}" for k in ks])
    ax.set_ylabel("Phoneme-weighted PER on remaining trials")
    ax.set_title("Previous-session vs geometry-selected source")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_selected_source_timeline(selection: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for k, frame in selection.groupby("calibration_trials"):
        target_dates = pd.to_datetime(frame["target_date"])
        source_dates = pd.to_datetime(frame["source_date"])
        ax.scatter(target_dates, source_dates, s=30, alpha=0.78, label=f"K={int(k)}")
    all_dates = pd.to_datetime(pd.concat([selection["target_date"], selection["source_date"]], ignore_index=True))
    ax.plot([all_dates.min(), all_dates.max()], [all_dates.min(), all_dates.max()], color="0.25", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Target session date")
    ax.set_ylabel("Selected source date")
    ax.set_title("Geometry-selected source timeline")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze K-shot selected sources and compare with previous-session baseline.")
    parser.add_argument("--model-path", type=Path, default=Path("data/external/btt-25-gru-pure-baseline-0-0898"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val", "test"], default="val")
    parser.add_argument("--gpu-number", type=int, default=-1)
    parser.add_argument("--max-trials-per-session", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--selection", type=Path, default=Path("results/tables/t15_kshot_geometry_source_selection.csv"))
    parser.add_argument("--kshot-trials", type=Path, default=Path("results/tables/t15_kshot_geometry_source_trial_results.csv"))
    parser.add_argument("--native-trials", type=Path, default=Path("results/tables/t15_decoder_probe_val.csv"))
    parser.add_argument(
        "--fixed-middle-trials",
        type=Path,
        default=Path("results/tables/t15_decoder_probe_cross_day_source_2023_11_26_val.csv"),
    )
    parser.add_argument(
        "--output-selected-summary",
        type=Path,
        default=Path("results/tables/t15_kshot_selected_source_summary.csv"),
    )
    parser.add_argument(
        "--output-selection-annotated",
        type=Path,
        default=Path("results/tables/t15_kshot_selected_sources_annotated.csv"),
    )
    parser.add_argument(
        "--output-previous-trials",
        type=Path,
        default=Path("results/tables/t15_kshot_previous_source_trial_results.csv"),
    )
    parser.add_argument(
        "--output-previous-summary",
        type=Path,
        default=Path("results/tables/t15_kshot_previous_source_session_summary.csv"),
    )
    parser.add_argument(
        "--output-previous-vs-geometry",
        type=Path,
        default=Path("results/tables/t15_kshot_previous_vs_geometry_summary.csv"),
    )
    parser.add_argument(
        "--output-session-comparison",
        type=Path,
        default=Path("results/tables/t15_kshot_previous_vs_geometry_session_comparison.csv"),
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    device = resolve_device(args.gpu_number)
    model, model_args = load_official_gru_decoder(ROOT, args.model_path, device)
    _set_use_amp(model_args, enabled=(device.type == "cuda" and bool(model_args["use_amp"])))
    model_sessions = sorted(list(model_args["dataset"]["sessions"]), key=session_date)

    selection = pd.read_csv(args.selection)
    annotated = add_selection_metadata(selection, model_sessions)
    previous_trials = decode_previous_source_trials(
        selection=annotated,
        model=model,
        model_args=model_args,
        model_sessions=list(model_args["dataset"]["sessions"]),
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        eval_type=args.eval_type,
        device=device,
        max_trials_per_session=args.max_trials_per_session,
    )
    previous_summary = summarize_sessions(previous_trials)
    selected_summary = build_selection_summary(annotated)
    previous_vs_geometry = build_overall_summary(
        selection=annotated,
        native_trials_path=args.native_trials,
        fixed_middle_trials_path=args.fixed_middle_trials,
        kshot_trials_path=args.kshot_trials,
        previous_trials=previous_trials,
    )
    session_comparison = build_session_comparison(
        selection=annotated,
        kshot_trials_path=args.kshot_trials,
        previous_summary=previous_summary,
    )

    for path in [
        args.output_selected_summary,
        args.output_selection_annotated,
        args.output_previous_trials,
        args.output_previous_summary,
        args.output_previous_vs_geometry,
        args.output_session_comparison,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    selected_summary.to_csv(args.output_selected_summary, index=False)
    annotated.to_csv(args.output_selection_annotated, index=False)
    previous_trials.to_csv(args.output_previous_trials, index=False)
    previous_summary.to_csv(args.output_previous_summary, index=False)
    previous_vs_geometry.to_csv(args.output_previous_vs_geometry, index=False)
    session_comparison.to_csv(args.output_session_comparison, index=False)

    plot_lag_histogram(annotated, args.figures_dir / "t15_selected_source_lag_histogram.png")
    plot_previous_vs_geometry(previous_vs_geometry, args.figures_dir / "t15_previous_vs_geometry_per.png")
    plot_selected_source_timeline(annotated, args.figures_dir / "t15_selected_source_timeline.png")

    print(f"Wrote {args.output_selected_summary}")
    print(f"Wrote {args.output_selection_annotated}")
    print(f"Wrote {args.output_previous_trials}")
    print(f"Wrote {args.output_previous_summary}")
    print(f"Wrote {args.output_previous_vs_geometry}")
    print(f"Wrote {args.output_session_comparison}")
    print(f"Wrote {args.figures_dir / 't15_selected_source_lag_histogram.png'}")
    print(f"Wrote {args.figures_dir / 't15_previous_vs_geometry_per.png'}")
    print(f"Wrote {args.figures_dir / 't15_selected_source_timeline.png'}")
    print(selected_summary.to_string(index=False))
    print(previous_vs_geometry.to_string(index=False))
    comparison_counts = (
        session_comparison.groupby("calibration_trials")
        .agg(
            sessions=("session", "size"),
            selected_previous_sessions=("selected_is_previous", "sum"),
            geometry_better_sessions=("geometry_better_than_previous", "sum"),
            previous_better_sessions=("previous_better_than_geometry", "sum"),
            mean_geometry_minus_previous_PER=("geometry_minus_previous_mean_PER", "mean"),
        )
        .reset_index()
    )
    print(comparison_counts.to_string(index=False))


if __name__ == "__main__":
    main()
