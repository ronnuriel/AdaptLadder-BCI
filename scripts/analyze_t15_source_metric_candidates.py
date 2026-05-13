from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_METRICS = [
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


def select_by_metric(pairwise: pd.DataFrame, metric: str) -> pd.DataFrame:
    selected = (
        pairwise.sort_values(["calibration_trials", "target_session", metric, "abs_days_from_source", "source_session"])
        .groupby(["calibration_trials", "target_session"], as_index=False)
        .first()
    )
    selected["selected_metric_value"] = selected[metric]
    selected = selected.rename(columns={"source_session": "selected_source_session"})
    selected["selection_metric"] = metric
    return selected[
        [
            "calibration_trials",
            "target_session",
            "selected_source_session",
            "selection_metric",
            "selected_metric_value",
            "abs_days_from_source",
            "days_from_source",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize T15 K-shot source candidates selected by different geometry metrics.")
    parser.add_argument(
        "--pairwise",
        type=Path,
        default=Path("results/tables/t15_kshot_recency_geometry_override_pairwise.csv"),
    )
    parser.add_argument(
        "--previous-comparison",
        type=Path,
        default=Path("results/tables/t15_kshot_previous_vs_geometry_session_comparison.csv"),
    )
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument(
        "--output-selection",
        type=Path,
        default=Path("results/tables/_explore_t15_source_metric_candidate_selection.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/_explore_t15_source_metric_candidate_summary.csv"),
    )
    args = parser.parse_args()

    pairwise = pd.read_csv(args.pairwise)
    previous = pd.read_csv(args.previous_comparison)[
        ["calibration_trials", "session", "previous_session", "selected_source_session"]
    ].rename(columns={"session": "target_session", "selected_source_session": "cov_selected_source_session"})

    selections = []
    for metric in args.metrics:
        if metric not in pairwise.columns:
            raise ValueError(f"Metric {metric!r} is missing from {args.pairwise}")
        selections.append(select_by_metric(pairwise, metric))
    selected = pd.concat(selections, ignore_index=True)
    selected = selected.merge(previous, on=["calibration_trials", "target_session"], how="left")
    selected["selected_is_previous"] = selected["selected_source_session"] == selected["previous_session"]
    selected["selected_matches_cov_metric"] = selected["selected_source_session"] == selected["cov_selected_source_session"]
    selected["selected_is_older_nonprevious"] = ~selected["selected_is_previous"]

    summary = (
        selected.groupby(["selection_metric", "calibration_trials"], as_index=False)
        .agg(
            num_targets=("target_session", "nunique"),
            previous_selections=("selected_is_previous", "sum"),
            older_nonprevious_selections=("selected_is_older_nonprevious", "sum"),
            matches_cov_metric=("selected_matches_cov_metric", "sum"),
            median_abs_lag_days=("abs_days_from_source", "median"),
            mean_abs_lag_days=("abs_days_from_source", "mean"),
        )
        .sort_values(["calibration_trials", "selection_metric"])
    )
    summary["previous_fraction"] = summary["previous_selections"] / summary["num_targets"]
    summary["matches_cov_fraction"] = summary["matches_cov_metric"] / summary["num_targets"]

    for path in [args.output_selection, args.output_summary]:
        path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_selection, index=False)
    summary.to_csv(args.output_summary, index=False)

    print(f"Wrote {args.output_selection}")
    print(f"Wrote {args.output_summary}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
