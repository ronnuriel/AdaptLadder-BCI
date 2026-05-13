from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FEATURES = [
    "geometry_weighted_PER",
    "geometry_previous_distance_ratio",
    "previous_metric_value",
    "geometry_metric_value",
    "margin_abs",
    "margin_fraction",
    "selected_lag_days",
    "selected_lag_sessions",
    "confidence_gain",
    "entropy_drop",
    "blank_rate_drop",
    "geometry_minus_previous_mean_shift_from_source",
    "geometry_minus_previous_scale_shift_from_source",
    "geometry_minus_previous_cov_relative_fro_shift_from_source",
    "geometry_minus_previous_coral_distance_from_source",
    "geometry_minus_previous_mean_principal_angle_deg",
    "geometry_minus_previous_subspace_chordal_distance",
    "geometry_minus_previous_basis_procrustes_error",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore predictors of previous-source failure.")
    parser.add_argument("--input", type=Path, default=Path("results/tables/t15_drift_type_fingerprint_joined.csv"))
    parser.add_argument(
        "--output-correlations",
        type=Path,
        default=Path("results/tables/_explore_t15_previous_failure_correlations.csv"),
    )
    parser.add_argument(
        "--output-thresholds",
        type=Path,
        default=Path("results/tables/_explore_t15_previous_failure_thresholds.csv"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    rows = []
    for k, k_frame in frame.groupby("calibration_trials"):
        y = k_frame["previous_weighted_PER"].astype(float)
        for feature in [column for column in FEATURES if column in k_frame.columns]:
            x = k_frame[feature].astype(float).replace([np.inf, -np.inf], np.nan)
            ok = x.notna() & y.notna()
            if int(ok.sum()) < 5:
                continue
            rho, p_value = spearmanr(x[ok], y[ok])
            rows.append(
                {
                    "calibration_trials": int(k),
                    "feature": feature,
                    "spearman_rho_vs_previous_PER": float(rho),
                    "abs_rho": float(abs(rho)),
                    "p_value": float(p_value),
                    "n": int(ok.sum()),
                }
            )
    correlations = pd.DataFrame(rows).sort_values(["calibration_trials", "p_value", "feature"])

    threshold_rows = []
    for k, k_frame in frame.groupby("calibration_trials"):
        previous = k_frame["previous_weighted_PER"].astype(float)
        q75 = float(previous.quantile(0.75))
        for threshold_name, threshold in [("PER>15", 0.15), ("PER>18", 0.18), ("worst25", q75)]:
            fail = previous > threshold
            threshold_rows.append(
                {
                    "calibration_trials": int(k),
                    "threshold_name": threshold_name,
                    "threshold_value": float(threshold),
                    "fail_sessions": int(fail.sum()),
                    "total_sessions": int(len(k_frame)),
                    "mean_previous_PER_fail": float(previous[fail].mean()) if fail.any() else np.nan,
                    "mean_previous_PER_ok": float(previous[~fail].mean()) if (~fail).any() else np.nan,
                    "median_previous_PER_fail": float(previous[fail].median()) if fail.any() else np.nan,
                    "median_previous_PER_ok": float(previous[~fail].median()) if (~fail).any() else np.nan,
                }
            )
    thresholds = pd.DataFrame(threshold_rows)

    args.output_correlations.parent.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(args.output_correlations, index=False)
    thresholds.to_csv(args.output_thresholds, index=False)

    print("Top correlations by p-value:")
    print(correlations.groupby("calibration_trials").head(8).to_string(index=False))
    print("\nTop correlations by |rho|:")
    print(correlations.sort_values(["calibration_trials", "abs_rho"], ascending=[True, False]).groupby("calibration_trials").head(8).to_string(index=False))
    print("\nFailure thresholds:")
    print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
