from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def gap_bin(days: float) -> str:
    if days <= 3:
        return "short_0_3"
    if days <= 14:
        return "medium_4_14"
    return "long_15_plus"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze previous-source PER as a function of time gap.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/tables/t15_drift_type_fingerprint_joined.csv"),
        help="Joined T15 source-selection/fingerprint table.",
    )
    parser.add_argument(
        "--output-sessions",
        type=Path,
        default=Path("results/tables/_explore_t15_previous_gap_sessions.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/_explore_t15_previous_gap_summary.csv"),
    )
    parser.add_argument(
        "--output-correlations",
        type=Path,
        default=Path("results/tables/_explore_t15_previous_gap_correlations.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/figures/_explore_t15_previous_gap_per.png"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    keep = [
        "calibration_trials",
        "session",
        "previous_session",
        "previous_abs_days",
        "previous_lag_sessions",
        "previous_weighted_PER",
        "geometry_weighted_PER",
        "geometry_metric_value",
        "previous_metric_value",
        "selected_lag_days",
    ]
    sessions = df[[c for c in keep if c in df.columns]].drop_duplicates().copy()
    sessions["gap_bin"] = sessions["previous_abs_days"].astype(float).map(gap_bin)

    args.output_sessions.parent.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(args.output_sessions, index=False)

    summary = (
        sessions.groupby(["calibration_trials", "gap_bin"], as_index=False)
        .agg(
            n_sessions=("session", "nunique"),
            mean_gap_days=("previous_abs_days", "mean"),
            median_gap_days=("previous_abs_days", "median"),
            mean_previous_PER=("previous_weighted_PER", "mean"),
            median_previous_PER=("previous_weighted_PER", "median"),
            mean_geometry_PER=("geometry_weighted_PER", "mean"),
            median_geometry_PER=("geometry_weighted_PER", "median"),
        )
        .sort_values(["calibration_trials", "mean_gap_days"])
    )
    summary.to_csv(args.output_summary, index=False)

    rows = []
    for k, frame in sessions.groupby("calibration_trials"):
        x = frame["previous_abs_days"].astype(float)
        y = frame["previous_weighted_PER"].astype(float)
        rho, p_value = spearmanr(x, y)
        rows.append(
            {
                "calibration_trials": int(k),
                "spearman_rho_gap_vs_previous_PER": float(rho),
                "p_value": float(p_value),
                "n_sessions": int(len(frame)),
                "min_gap_days": float(x.min()),
                "median_gap_days": float(x.median()),
                "max_gap_days": float(x.max()),
            }
        )
    correlations = pd.DataFrame(rows)
    correlations.to_csv(args.output_correlations, index=False)

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (k, frame) in zip(axes, sessions.groupby("calibration_trials")):
        ax.scatter(frame["previous_abs_days"], frame["previous_weighted_PER"], s=28, alpha=0.8)
        ax.set_title(f"K={int(k)}")
        ax.set_xlabel("Gap from previous source (days)")
        ax.grid(alpha=0.25)
        corr = correlations[correlations["calibration_trials"] == k].iloc[0]
        ax.text(
            0.03,
            0.95,
            f"rho={corr.spearman_rho_gap_vs_previous_PER:.2f}\\np={corr.p_value:.3g}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    axes[0].set_ylabel("Previous-source weighted PER")
    fig.tight_layout()
    fig.savefig(args.output_figure, dpi=200)

    print("Previous-source gap summary:")
    print(summary.to_string(index=False))
    print("\nGap correlations:")
    print(correlations.to_string(index=False))
    print(f"\nWrote {args.output_figure}")


if __name__ == "__main__":
    main()
