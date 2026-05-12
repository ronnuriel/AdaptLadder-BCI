from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_t15_selected_sources import previous_session_for, summarize_sessions
from scripts.run_t15_geometry_source_selection_eval import (
    SELECTION_METRICS,
    compute_session_stats,
    discover_sessions,
    pairwise_geometry_table,
    weighted_per,
)
from scripts.run_t15_kshot_geometry_source_selection import calibration_window_stats, subset_existing_trials
from src.decoder_eval import add_official_model_training_to_path
from src.t15_utils import session_date


def _method_row(method: str, trials: pd.DataFrame, calibration_trials: int) -> dict[str, float | int | str]:
    return {
        "calibration_trials": int(calibration_trials),
        "method": method,
        "weighted_PER": weighted_per(trials),
        "trial_mean_PER": float(trials["PER"].mean()),
        "num_sessions": int(trials["session"].nunique()),
        "num_trials": int(len(trials)),
    }


def _add_trial_key(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        trial_key=list(
            zip(
                frame["calibration_trials"],
                frame["session"],
                frame["trial_index_within_session"],
            )
        )
    )


def compute_kshot_pairwise(
    data_dir: Path,
    csv_path: Path,
    eval_type: str,
    source_stats_split: str,
    calibration_trials: list[int],
    min_eval_trials: int,
    source_candidate_mode: str,
    max_source_frames: int,
    seed: int,
    n_components: int,
    cov_shrinkage: float,
    max_sessions: int | None,
    max_trials_per_session: int | None,
) -> pd.DataFrame:
    add_official_model_training_to_path(ROOT)
    from evaluate_model_helpers import load_h5py_file

    if source_candidate_mode != "past-only":
        raise ValueError("Recency override requires --source-candidate-mode past-only.")

    b2txt_csv_df = pd.read_csv(csv_path)
    model_sessions = sorted(
        [path.name for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("t15.")],
        key=session_date,
    )
    eval_sessions = discover_sessions(data_dir, eval_type, model_sessions)
    source_sessions = discover_sessions(data_dir, source_stats_split, model_sessions)
    sessions = [session for session in eval_sessions if session in source_sessions]
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
        source_sessions = [session for session in source_sessions if session in sessions]

    source_stats = compute_session_stats(
        data_dir=data_dir,
        sessions=source_sessions,
        split=source_stats_split,
        max_frames=max_source_frames,
        seed=seed,
        n_components=n_components,
        shrinkage=cov_shrinkage,
    )

    rows = []
    for session in sessions:
        data = load_h5py_file(str(data_dir / session / f"data_{eval_type}.hdf5"), b2txt_csv_df)
        n_trials = len(data["neural_features"])
        if max_trials_per_session is not None:
            n_trials = min(n_trials, max_trials_per_session)
            data["neural_features"] = data["neural_features"][:n_trials]

        target_date = session_date(session)
        candidate_sources = [
            source_session
            for source_session in source_sessions
            if source_session != session and session_date(source_session) < target_date
        ]
        if not candidate_sources:
            continue

        for k in calibration_trials:
            if n_trials < k + min_eval_trials:
                continue
            target_stats = calibration_window_stats(
                data["neural_features"],
                calibration_trials=k,
                n_components=n_components,
                shrinkage=cov_shrinkage,
            )
            stats = dict(source_stats)
            stats[session] = target_stats
            pairwise = pairwise_geometry_table(
                stats,
                sessions=[session, *candidate_sources],
                allow_native=False,
                source_candidate_mode=source_candidate_mode,
            )
            pairwise = pairwise[pairwise["target_session"] == session].copy()
            if pairwise.empty:
                continue
            pairwise.insert(0, "calibration_trials", int(k))
            pairwise["eval_trials"] = int(n_trials - k)
            pairwise["target_calibration_frames"] = int(target_stats["n_frames_sampled"])
            rows.append(pairwise)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_override_decisions(pairwise: pd.DataFrame, selection_metric: str, alphas: list[float]) -> pd.DataFrame:
    selected = (
        pairwise.sort_values(["calibration_trials", "target_session", selection_metric, "abs_days_from_source", "source_session"])
        .groupby(["calibration_trials", "target_session"], as_index=False)
        .first()
    ).rename(
        columns={
            "source_session": "geometry_source_session",
            selection_metric: "geometry_metric_value",
        }
    )
    selected["selection_metric"] = selection_metric
    previous_rows = []
    ordered_sessions = sorted(
        set(pairwise["target_session"].unique().tolist() + pairwise["source_session"].unique().tolist()),
        key=session_date,
    )
    for row in selected.to_dict(orient="records"):
        target = row["target_session"]
        previous = previous_session_for(target, ordered_sessions)
        if previous is None:
            continue
        previous_frame = pairwise[
            (pairwise["calibration_trials"] == row["calibration_trials"])
            & (pairwise["target_session"] == target)
            & (pairwise["source_session"] == previous)
        ]
        if previous_frame.empty:
            continue
        previous_rows.append(
            {
                "calibration_trials": int(row["calibration_trials"]),
                "target_session": target,
                "previous_source_session": previous,
                "previous_metric_value": float(previous_frame[selection_metric].iloc[0]),
                "previous_lag_days": int(previous_frame["days_from_source"].iloc[0]),
                "previous_lag_sessions": int(
                    ordered_sessions.index(target) - ordered_sessions.index(previous)
                ),
            }
        )
    previous = pd.DataFrame(previous_rows)
    decisions = selected.merge(previous, on=["calibration_trials", "target_session"], how="inner")
    decisions["geometry_lag_days"] = decisions["days_from_source"].astype(int)
    decisions["geometry_lag_sessions"] = [
        int(ordered_sessions.index(target) - ordered_sessions.index(source))
        for target, source in zip(decisions["target_session"], decisions["geometry_source_session"], strict=True)
    ]
    decisions["geometry_is_previous"] = decisions["geometry_source_session"] == decisions["previous_source_session"]
    decisions["geometry_previous_distance_ratio"] = decisions["geometry_metric_value"] / decisions["previous_metric_value"].replace(0, np.nan)

    rows = []
    for row in decisions.to_dict(orient="records"):
        for alpha in alphas:
            use_geometry = (not row["geometry_is_previous"]) and row["geometry_previous_distance_ratio"] < alpha
            rows.append(
                {
                    "calibration_trials": int(row["calibration_trials"]),
                    "target_session": row["target_session"],
                    "target_date": row["target_date"],
                    "alpha": float(alpha),
                    "previous_source_session": row["previous_source_session"],
                    "geometry_source_session": row["geometry_source_session"],
                    "chosen_source_session": row["geometry_source_session"] if use_geometry else row["previous_source_session"],
                    "choice": "geometry_override" if use_geometry else "previous_default",
                    "override_used": bool(use_geometry),
                    "geometry_is_previous": bool(row["geometry_is_previous"]),
                    "previous_metric_value": float(row["previous_metric_value"]),
                    "geometry_metric_value": float(row["geometry_metric_value"]),
                    "geometry_previous_distance_ratio": float(row["geometry_previous_distance_ratio"]),
                    "previous_lag_days": int(row["previous_lag_days"]),
                    "geometry_lag_days": int(row["geometry_lag_days"]),
                    "previous_lag_sessions": int(row["previous_lag_sessions"]),
                    "geometry_lag_sessions": int(row["geometry_lag_sessions"]),
                    "selection_metric": selection_metric,
                }
            )
    return pd.DataFrame(rows)


def build_override_trials(
    decisions: pd.DataFrame,
    geometry_trials: pd.DataFrame,
    previous_trials: pd.DataFrame,
) -> pd.DataFrame:
    geometry_trials = _add_trial_key(geometry_trials)
    previous_trials = _add_trial_key(previous_trials)
    rows = []
    for decision in decisions.to_dict(orient="records"):
        source_trials = geometry_trials if decision["override_used"] else previous_trials
        frame = source_trials[
            (source_trials["calibration_trials"] == decision["calibration_trials"])
            & (source_trials["session"] == decision["target_session"])
        ].copy()
        if frame.empty:
            continue
        frame["mode"] = f"recency_geometry_override_alpha_{decision['alpha']:.2f}"
        frame["override_alpha"] = float(decision["alpha"])
        frame["override_used"] = bool(decision["override_used"])
        frame["previous_source_session"] = decision["previous_source_session"]
        frame["geometry_source_session"] = decision["geometry_source_session"]
        frame["chosen_source_session"] = decision["chosen_source_session"]
        frame["geometry_previous_distance_ratio"] = decision["geometry_previous_distance_ratio"]
        frame["input_layer_session"] = decision["chosen_source_session"]
        rows.append(frame.drop(columns=["trial_key"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_overall_summary(
    decisions: pd.DataFrame,
    override_trials: pd.DataFrame,
    geometry_trials: pd.DataFrame,
    previous_trials: pd.DataFrame,
    native_trials_path: Path,
    fixed_middle_trials_path: Path,
) -> pd.DataFrame:
    rows = []
    selected = decisions[["calibration_trials", "target_session"]].drop_duplicates().rename(columns={"target_session": "session"})
    for k, selected_k in selected.groupby("calibration_trials"):
        native = subset_existing_trials(native_trials_path, selected_k)
        fixed = subset_existing_trials(fixed_middle_trials_path, selected_k)
        selected_sessions = set(selected_k["session"])
        previous_k = previous_trials[
            (previous_trials["calibration_trials"] == k) & (previous_trials["session"].isin(selected_sessions))
        ]
        geometry_k = geometry_trials[
            (geometry_trials["calibration_trials"] == k) & (geometry_trials["session"].isin(selected_sessions))
        ]
        override_k_all = override_trials[override_trials["calibration_trials"] == k]
        if native is not None and not native.empty:
            rows.append(_method_row("native-day", native, int(k)))
        if fixed is not None and not fixed.empty:
            rows.append(_method_row("fixed_middle_source", fixed, int(k)))
        if not previous_k.empty:
            rows.append(_method_row("previous_source", previous_k, int(k)))
        if not geometry_k.empty:
            rows.append(_method_row("kshot_geometry_nearest", geometry_k, int(k)))
        for alpha, override_k in override_k_all.groupby("override_alpha"):
            rows.append(_method_row(f"recency_geometry_override_alpha_{alpha:.2f}", override_k, int(k)))

    summary = pd.DataFrame(rows)
    native = summary[summary["method"] == "native-day"][["calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "native_weighted_PER"}
    )
    fixed = summary[summary["method"] == "fixed_middle_source"][["calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "fixed_middle_weighted_PER"}
    )
    previous = summary[summary["method"] == "previous_source"][["calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "previous_weighted_PER"}
    )
    summary = summary.merge(native, on="calibration_trials", how="left")
    summary = summary.merge(fixed, on="calibration_trials", how="left")
    summary = summary.merge(previous, on="calibration_trials", how="left")
    summary["delta_vs_native_weighted_PER"] = summary["weighted_PER"] - summary["native_weighted_PER"]
    summary["gain_vs_fixed_middle_weighted_PER"] = summary["fixed_middle_weighted_PER"] - summary["weighted_PER"]
    summary["gain_vs_previous_weighted_PER"] = summary["previous_weighted_PER"] - summary["weighted_PER"]
    gap = summary["fixed_middle_weighted_PER"] - summary["native_weighted_PER"]
    summary["recovery_fraction_vs_fixed_middle"] = summary["gain_vs_fixed_middle_weighted_PER"] / gap.replace(0, np.nan)

    override_counts = (
        decisions.groupby(["calibration_trials", "alpha"], as_index=False)
        .agg(overrides_used=("override_used", "sum"), candidate_sessions=("target_session", "nunique"))
        .assign(method=lambda frame: frame["alpha"].map(lambda alpha: f"recency_geometry_override_alpha_{alpha:.2f}"))
    )
    summary = summary.merge(
        override_counts[["calibration_trials", "method", "overrides_used", "candidate_sessions"]],
        on=["calibration_trials", "method"],
        how="left",
    )
    summary["overrides_used"] = summary["overrides_used"].fillna(0).astype(int)
    summary["candidate_sessions"] = summary["candidate_sessions"].fillna(summary["num_sessions"]).astype(int)
    return summary.sort_values(["calibration_trials", "method"]).reset_index(drop=True)


def build_session_summary(override_trials: pd.DataFrame) -> pd.DataFrame:
    if override_trials.empty:
        return pd.DataFrame()
    summary = summarize_sessions(override_trials)
    decisions = override_trials[
        [
            "calibration_trials",
            "session",
            "override_alpha",
            "override_used",
            "previous_source_session",
            "geometry_source_session",
            "chosen_source_session",
            "geometry_previous_distance_ratio",
        ]
    ].drop_duplicates()
    return summary.merge(
        decisions,
        left_on=["calibration_trials", "session", "input_layer_session"],
        right_on=["calibration_trials", "session", "chosen_source_session"],
        how="left",
    )


def plot_override_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(summary["calibration_trials"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(4.4 * len(ks), 4.2), sharey=True)
    if len(ks) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks, strict=True):
        frame = summary[summary["calibration_trials"] == k]
        reference = frame[frame["method"].isin(["native-day", "fixed_middle_source", "previous_source", "kshot_geometry_nearest"])]
        override = frame[frame["method"].str.startswith("recency_geometry_override_alpha_")].copy()
        override["alpha"] = override["method"].str.extract(r"([0-9]+\\.[0-9]+)$").astype(float)
        ax.axhline(float(reference[reference["method"] == "native-day"]["weighted_PER"].iloc[0]), color="#2a9d8f", linewidth=1.4, label="Native")
        ax.axhline(float(reference[reference["method"] == "fixed_middle_source"]["weighted_PER"].iloc[0]), color="#457b9d", linewidth=1.4, label="Fixed middle")
        ax.axhline(float(reference[reference["method"] == "previous_source"]["weighted_PER"].iloc[0]), color="#f4a261", linewidth=1.4, label="Previous")
        ax.axhline(float(reference[reference["method"] == "kshot_geometry_nearest"]["weighted_PER"].iloc[0]), color="#b23a48", linewidth=1.4, label="Always geometry")
        ax.plot(override["alpha"], override["weighted_PER"], marker="o", color="#111111", label="Override")
        for _idx, row in override.iterrows():
            ax.annotate(str(int(row["overrides_used"])), (row["alpha"], row["weighted_PER"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.set_title(f"K={int(k)}")
        ax.set_xlabel("Override alpha")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Phoneme-weighted PER")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Recency-aware geometry override")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recency-aware K-shot geometry source override.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val", "test"], default="val")
    parser.add_argument("--source-stats-split", default="train")
    parser.add_argument("--calibration-trials", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--min-eval-trials", type=int, default=1)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="cov_relative_fro_shift_from_source")
    parser.add_argument("--source-candidate-mode", choices=["past-only"], default="past-only")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--max-source-frames", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--cov-shrinkage", type=float, default=0.05)
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--max-trials-per-session", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--native-trials", type=Path, default=Path("results/tables/t15_decoder_probe_val.csv"))
    parser.add_argument(
        "--fixed-middle-trials",
        type=Path,
        default=Path("results/tables/t15_decoder_probe_cross_day_source_2023_11_26_val.csv"),
    )
    parser.add_argument(
        "--geometry-trials",
        type=Path,
        default=Path("results/tables/t15_kshot_geometry_source_trial_results.csv"),
    )
    parser.add_argument(
        "--previous-trials",
        type=Path,
        default=Path("results/tables/t15_kshot_previous_source_trial_results.csv"),
    )
    parser.add_argument(
        "--output-pairwise",
        type=Path,
        default=Path("results/tables/t15_kshot_recency_geometry_override_pairwise.csv"),
    )
    parser.add_argument(
        "--output-decisions",
        type=Path,
        default=Path("results/tables/t15_kshot_recency_geometry_override_decisions.csv"),
    )
    parser.add_argument(
        "--output-trials",
        type=Path,
        default=Path("results/tables/t15_kshot_recency_geometry_override_trial_results.csv"),
    )
    parser.add_argument(
        "--output-session-summary",
        type=Path,
        default=Path("results/tables/t15_kshot_recency_geometry_override_session_summary.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/t15_kshot_recency_geometry_override_summary.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/figures/t15_kshot_recency_geometry_override_weighted_per.png"),
    )
    args = parser.parse_args()

    pairwise = compute_kshot_pairwise(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        eval_type=args.eval_type,
        source_stats_split=args.source_stats_split,
        calibration_trials=args.calibration_trials,
        min_eval_trials=args.min_eval_trials,
        source_candidate_mode=args.source_candidate_mode,
        max_source_frames=args.max_source_frames,
        seed=args.seed,
        n_components=args.n_components,
        cov_shrinkage=args.cov_shrinkage,
        max_sessions=args.max_sessions,
        max_trials_per_session=args.max_trials_per_session,
    )
    if pairwise.empty:
        raise ValueError("No K-shot pairwise distances were computed.")

    decisions = build_override_decisions(pairwise, args.selection_metric, sorted(args.alpha))
    geometry_trials = pd.read_csv(args.geometry_trials)
    previous_trials = pd.read_csv(args.previous_trials)
    override_trials = build_override_trials(decisions, geometry_trials, previous_trials)
    session_summary = build_session_summary(override_trials)
    summary = build_overall_summary(
        decisions=decisions,
        override_trials=override_trials,
        geometry_trials=geometry_trials,
        previous_trials=previous_trials,
        native_trials_path=args.native_trials,
        fixed_middle_trials_path=args.fixed_middle_trials,
    )

    for path in [
        args.output_pairwise,
        args.output_decisions,
        args.output_trials,
        args.output_session_summary,
        args.output_summary,
        args.output_figure,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    pairwise.to_csv(args.output_pairwise, index=False)
    decisions.to_csv(args.output_decisions, index=False)
    override_trials.to_csv(args.output_trials, index=False)
    session_summary.to_csv(args.output_session_summary, index=False)
    summary.to_csv(args.output_summary, index=False)
    plot_override_summary(summary, args.output_figure)

    print(f"Wrote {args.output_pairwise}")
    print(f"Wrote {args.output_decisions}")
    print(f"Wrote {args.output_trials}")
    print(f"Wrote {args.output_session_summary}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_figure}")
    cols = [
        "calibration_trials",
        "method",
        "weighted_PER",
        "recovery_fraction_vs_fixed_middle",
        "gain_vs_previous_weighted_PER",
        "overrides_used",
        "candidate_sessions",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
