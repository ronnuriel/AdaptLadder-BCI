from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISTANCE_METRICS = [
    "mean_shift_from_source",
    "scale_shift_from_source",
    "diag_cov_shift_from_source",
    "cov_relative_fro_shift_from_source",
    "coral_distance_from_source",
    "mean_principal_angle_deg",
    "subspace_chordal_distance",
    "basis_procrustes_error",
    "abs_days_from_source",
]

QUALITY_DIFFS = [
    "confidence_gain",
    "entropy_drop",
    "blank_rate_drop",
]


def load_pairwise(path: Path) -> pd.DataFrame:
    pairwise = pd.read_csv(path)
    required = {"target_session", "source_session", *DISTANCE_METRICS}
    missing = required - set(pairwise.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return pairwise


def add_source_distances(table: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    distance_cols = ["target_session", "source_session", *DISTANCE_METRICS]
    previous = pairwise[distance_cols].rename(
        columns={
            "target_session": "session",
            "source_session": "previous_source_session",
            **{metric: f"previous_{metric}" for metric in DISTANCE_METRICS},
        }
    )
    geometry = pairwise[distance_cols].rename(
        columns={
            "target_session": "session",
            "source_session": "geometry_source_session",
            **{metric: f"geometry_{metric}" for metric in DISTANCE_METRICS},
        }
    )
    out = table.merge(previous, on=["session", "previous_source_session"], how="left")
    out = out.merge(geometry, on=["session", "geometry_source_session"], how="left")
    for metric in DISTANCE_METRICS:
        out[f"geometry_minus_previous_{metric}"] = out[f"geometry_{metric}"] - out[f"previous_{metric}"]
        denom = out[f"previous_{metric}"].replace(0, np.nan)
        out[f"geometry_previous_ratio_{metric}"] = out[f"geometry_{metric}"] / denom
    out["weighted_per_delta_geometry_minus_previous"] = out["geometry_weighted_PER"] - out["previous_weighted_PER"]
    return out


def label_cases(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["case_type"] = np.select(
        [
            frame["selected_is_previous"].astype(bool),
            (~frame["selected_is_previous"].astype(bool)) & frame["geometry_better_weighted"].astype(bool),
            (~frame["selected_is_previous"].astype(bool)) & (~frame["geometry_better_weighted"].astype(bool)),
        ],
        ["geometry_selected_previous", "older_geometry_wins", "older_geometry_loses"],
        default="other",
    )
    return frame


def summarize_fingerprints(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, k_frame in frame.groupby("calibration_trials"):
        for case_type, case_frame in k_frame.groupby("case_type"):
            row = {
                "calibration_trials": int(k),
                "case_type": case_type,
                "n_sessions": int(len(case_frame)),
                "mean_per_delta_geometry_minus_previous": float(case_frame["weighted_per_delta_geometry_minus_previous"].mean()),
                "median_per_delta_geometry_minus_previous": float(case_frame["weighted_per_delta_geometry_minus_previous"].median()),
                "mean_geometry_previous_distance_ratio": float(case_frame["geometry_previous_distance_ratio"].mean()),
                "median_geometry_previous_distance_ratio": float(case_frame["geometry_previous_distance_ratio"].median()),
                "mean_lag_diff_days": float(case_frame["lag_diff_days"].mean()),
            }
            for metric in DISTANCE_METRICS:
                row[f"mean_delta_{metric}"] = float(case_frame[f"geometry_minus_previous_{metric}"].mean())
                row[f"median_ratio_{metric}"] = float(case_frame[f"geometry_previous_ratio_{metric}"].median())
            for metric in QUALITY_DIFFS:
                if metric in case_frame.columns:
                    row[f"mean_{metric}"] = float(case_frame[metric].mean())
                    row[f"median_{metric}"] = float(case_frame[metric].median())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["calibration_trials", "case_type"]).reset_index(drop=True)


def summarize_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    non_previous = frame[~frame["selected_is_previous"].astype(bool)].copy()
    for k, k_frame in non_previous.groupby("calibration_trials"):
        winners = k_frame[k_frame["geometry_better_weighted"].astype(bool)]
        losers = k_frame[~k_frame["geometry_better_weighted"].astype(bool)]
        for metric in DISTANCE_METRICS:
            col = f"geometry_minus_previous_{metric}"
            rows.append(
                {
                    "calibration_trials": int(k),
                    "feature": col,
                    "older_wins_mean": float(winners[col].mean()),
                    "older_loses_mean": float(losers[col].mean()),
                    "mean_difference_wins_minus_loses": float(winners[col].mean() - losers[col].mean()),
                    "older_wins_median": float(winners[col].median()),
                    "older_loses_median": float(losers[col].median()),
                }
            )
        for metric in QUALITY_DIFFS:
            rows.append(
                {
                    "calibration_trials": int(k),
                    "feature": metric,
                    "older_wins_mean": float(winners[metric].mean()),
                    "older_loses_mean": float(losers[metric].mean()),
                    "mean_difference_wins_minus_loses": float(winners[metric].mean() - losers[metric].mean()),
                    "older_wins_median": float(winners[metric].median()),
                    "older_loses_median": float(losers[metric].median()),
                }
            )
    return pd.DataFrame(rows)


def plot_case_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    case_order = ["older_geometry_wins", "older_geometry_loses"]
    colors = {"older_geometry_wins": "#2a9d8f", "older_geometry_loses": "#e76f51"}
    ks = sorted(summary["calibration_trials"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    x = np.arange(len(ks))
    width = 0.34
    for offset, case_type in zip([-width / 2, width / 2], case_order, strict=True):
        frame = summary[summary["case_type"] == case_type]
        n_values = []
        ratio_values = []
        for k in ks:
            row = frame[frame["calibration_trials"] == k]
            n_values.append(float(row["n_sessions"].iloc[0]) if not row.empty else 0.0)
            ratio_values.append(float(row["median_geometry_previous_distance_ratio"].iloc[0]) if not row.empty else np.nan)
        axes[0].bar(x + offset, n_values, width=width, color=colors[case_type], label=case_type.replace("_", " "))
        axes[1].bar(x + offset, ratio_values, width=width, color=colors[case_type], label=case_type.replace("_", " "))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"K={k}" for k in ks])
    axes[0].set_ylabel("Non-previous selections")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].axhline(1.0, color="0.25", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"K={k}" for k in ks])
    axes[1].set_ylabel("Median geometry/previous distance")
    axes[1].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Drift fingerprints of non-recent source choices")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare drift fingerprints when older geometry sources win or lose.")
    parser.add_argument(
        "--session-table",
        type=Path,
        default=Path("results/tables/t15_kshot_learned_gate_session_table.csv"),
    )
    parser.add_argument(
        "--pairwise",
        type=Path,
        default=Path("results/tables/t15_geometry_source_pairwise_distances_past_only.csv"),
    )
    parser.add_argument(
        "--output-joined",
        type=Path,
        default=Path("results/tables/t15_drift_type_fingerprint_joined.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/t15_drift_type_fingerprint_summary.csv"),
    )
    parser.add_argument(
        "--output-predictors",
        type=Path,
        default=Path("results/tables/t15_drift_type_fingerprint_predictors.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/figures/t15_drift_type_fingerprint_summary.png"),
    )
    args = parser.parse_args()

    table = pd.read_csv(args.session_table)
    pairwise = load_pairwise(args.pairwise)
    joined = label_cases(add_source_distances(table, pairwise))
    summary = summarize_fingerprints(joined)
    predictors = summarize_predictors(joined)

    for path in [args.output_joined, args.output_summary, args.output_predictors, args.output_figure]:
        path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output_joined, index=False)
    summary.to_csv(args.output_summary, index=False)
    predictors.to_csv(args.output_predictors, index=False)
    plot_case_summary(summary, args.output_figure)

    print(f"Wrote {args.output_joined}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_predictors}")
    print(f"Wrote {args.output_figure}")
    keep = [
        "calibration_trials",
        "case_type",
        "n_sessions",
        "mean_per_delta_geometry_minus_previous",
        "median_geometry_previous_distance_ratio",
        "mean_delta_cov_relative_fro_shift_from_source",
        "mean_delta_mean_principal_angle_deg",
        "mean_confidence_gain",
        "mean_entropy_drop",
        "mean_blank_rate_drop",
    ]
    print(summary[[col for col in keep if col in summary.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
