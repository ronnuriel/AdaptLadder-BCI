#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRIC = "cov_relative_fro_shift_from_source"


def weighted_session_per(trials: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = (
        trials.groupby(["calibration_trials", "session"], as_index=False)
        .agg(
            edit_distance=("edit_distance", "sum"),
            num_phonemes=("num_phonemes", "sum"),
            n_eval_trials=("trial", "count"),
        )
    )
    out[f"{prefix}_PER"] = out["edit_distance"] / out["num_phonemes"]
    out = out.rename(
        columns={
            "edit_distance": f"{prefix}_edit_distance",
            "num_phonemes": f"{prefix}_num_phonemes",
            "n_eval_trials": f"{prefix}_eval_trials",
        }
    )
    return out


def merge_distance(
    frame: pd.DataFrame,
    pairwise: pd.DataFrame,
    source_col: str,
    prefix: str,
    metric: str,
) -> pd.DataFrame:
    cols = [
        "calibration_trials",
        "target_session",
        "source_session",
        metric,
        "mean_shift_from_source",
        "scale_shift_from_source",
        "coral_distance_from_source",
        "mean_principal_angle_deg",
        "subspace_chordal_distance",
        "basis_procrustes_error",
    ]
    available = [c for c in cols if c in pairwise.columns]
    distances = pairwise[available].copy()
    rename = {
        "source_session": source_col,
        metric: f"{prefix}_{metric}",
        "mean_shift_from_source": f"{prefix}_mean_shift",
        "scale_shift_from_source": f"{prefix}_scale_shift",
        "coral_distance_from_source": f"{prefix}_coral_distance",
        "mean_principal_angle_deg": f"{prefix}_mean_principal_angle_deg",
        "subspace_chordal_distance": f"{prefix}_subspace_chordal_distance",
        "basis_procrustes_error": f"{prefix}_basis_procrustes_error",
    }
    distances = distances.rename(columns={k: v for k, v in rename.items() if k in distances.columns})
    return frame.merge(
        distances,
        on=["calibration_trials", "target_session", source_col],
        how="left",
        validate="many_to_one",
    )


def winner(row: pd.Series, eps: float = 1e-12) -> str:
    if pd.isna(row["previous_PER"]) or pd.isna(row["geometry_PER"]):
        return "unknown"
    delta = row["geometry_PER"] - row["previous_PER"]
    if abs(delta) <= eps:
        return "tie"
    if delta < 0:
        return "geometry"
    return "previous"


def build_detail(args: argparse.Namespace) -> pd.DataFrame:
    selection = pd.read_csv(args.selection)
    pairwise = pd.read_csv(args.pairwise)
    previous_trials = pd.read_csv(args.previous_trials)
    geometry_trials = pd.read_csv(args.geometry_trials)

    detail = selection.rename(
        columns={
            "target_session": "target_session",
            "source_session": "geometry_source_session",
            "source_date": "geometry_source_date",
            "selection_metric_value": "geometry_selection_metric_value",
            "abs_days_from_source": "geometry_lag_days",
        }
    ).copy()
    if "previous_abs_days" in detail.columns:
        detail = detail.rename(columns={"previous_abs_days": "previous_lag_days"})
    if "selected_is_previous" in detail.columns:
        detail = detail.rename(columns={"selected_is_previous": "same_as_previous"})
    if "target_session" not in detail.columns and "session" in detail.columns:
        detail = detail.rename(columns={"session": "target_session"})

    previous_per = weighted_session_per(previous_trials, "previous").rename(columns={"session": "target_session"})
    geometry_per = weighted_session_per(geometry_trials, "geometry").rename(columns={"session": "target_session"})
    detail = detail.merge(previous_per, on=["calibration_trials", "target_session"], how="left", validate="many_to_one")
    detail = detail.merge(geometry_per, on=["calibration_trials", "target_session"], how="left", validate="many_to_one")

    detail = merge_distance(detail, pairwise, "previous_session", "previous", args.metric)
    detail = merge_distance(detail, pairwise, "geometry_source_session", "geometry", args.metric)

    detail["same_as_previous"] = detail["geometry_source_session"] == detail["previous_session"]
    detail["jumped_older_than_previous"] = (~detail["same_as_previous"]) & (
        detail["geometry_lag_days"] > detail["previous_lag_days"]
    )
    detail["extra_lag_days_vs_previous"] = detail["geometry_lag_days"] - detail["previous_lag_days"]
    detail["geometry_minus_previous_PER"] = detail["geometry_PER"] - detail["previous_PER"]
    detail["winner"] = detail.apply(winner, axis=1)
    detail["geometry_distance_advantage"] = detail[f"previous_{args.metric}"] - detail[f"geometry_{args.metric}"]
    detail["geometry_distance_ratio_vs_previous"] = detail[f"geometry_{args.metric}"] / detail[f"previous_{args.metric}"].replace(0, np.nan)

    ordered = [
        "calibration_trials",
        "target_session",
        "target_date",
        "previous_session",
        "previous_date",
        "geometry_source_session",
        "geometry_source_date",
        "same_as_previous",
        "jumped_older_than_previous",
        "previous_lag_days",
        "geometry_lag_days",
        "extra_lag_days_vs_previous",
        f"previous_{args.metric}",
        f"geometry_{args.metric}",
        "geometry_distance_advantage",
        "geometry_distance_ratio_vs_previous",
        "previous_PER",
        "geometry_PER",
        "geometry_minus_previous_PER",
        "winner",
        "previous_eval_trials",
        "geometry_eval_trials",
    ]
    return detail[[c for c in ordered if c in detail.columns]].sort_values(["calibration_trials", "target_date"])


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for k, group in detail.groupby("calibration_trials"):
        nonprev = group[~group["same_as_previous"]].copy()
        rows.append(
            {
                "calibration_trials": int(k),
                "n_targets": int(group["target_session"].nunique()),
                "selected_previous": int(group["same_as_previous"].sum()),
                "selected_non_previous": int((~group["same_as_previous"]).sum()),
                "selected_previous_fraction": float(group["same_as_previous"].mean()),
                "nonprevious_geometry_wins": int((nonprev["winner"] == "geometry").sum()),
                "nonprevious_previous_wins": int((nonprev["winner"] == "previous").sum()),
                "nonprevious_ties": int((nonprev["winner"] == "tie").sum()),
                "median_geometry_lag_days": float(group["geometry_lag_days"].median()),
                "max_geometry_lag_days": int(group["geometry_lag_days"].max()),
                "median_extra_lag_nonprevious": float(nonprev["extra_lag_days_vs_previous"].median()) if len(nonprev) else np.nan,
                "max_extra_lag_nonprevious": float(nonprev["extra_lag_days_vs_previous"].max()) if len(nonprev) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    nonprevious = detail[~detail["same_as_previous"]].sort_values(
        ["calibration_trials", "target_date", "geometry_source_session"]
    )
    return summary, nonprevious


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare T15 K-shot geometry-selected sources to previous-session sources.")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/tables/t15_kshot_selected_sources_annotated.csv"),
    )
    parser.add_argument(
        "--pairwise",
        type=Path,
        default=Path("results/tables/t15_kshot_recency_geometry_override_pairwise.csv"),
    )
    parser.add_argument(
        "--previous-trials",
        type=Path,
        default=Path("results/tables/t15_kshot_previous_source_trial_results.csv"),
    )
    parser.add_argument(
        "--geometry-trials",
        type=Path,
        default=Path("results/tables/t15_kshot_geometry_source_trial_results.csv"),
    )
    parser.add_argument(
        "--output-detail",
        type=Path,
        default=Path("results/tables/t15_geometry_vs_previous_sources.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/t15_geometry_vs_previous_summary.csv"),
    )
    parser.add_argument(
        "--output-nonprevious",
        type=Path,
        default=Path("results/tables/t15_geometry_nonprevious_selections.csv"),
    )
    args = parser.parse_args()

    detail = build_detail(args)
    summary, nonprevious = summarize(detail)

    for path in [args.output_detail, args.output_summary, args.output_nonprevious]:
        path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_detail, index=False)
    summary.to_csv(args.output_summary, index=False)
    nonprevious.to_csv(args.output_nonprevious, index=False)

    print(f"Wrote {args.output_detail}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_nonprevious}")
    print("\n=== Geometry vs previous summary ===")
    printable = summary.copy()
    printable["selected_previous_fraction"] = (100 * printable["selected_previous_fraction"]).map(lambda v: f"{v:.1f}%")
    print(printable.to_string(index=False))

    print("\n=== Non-previous geometry selections ===")
    cols = [
        "calibration_trials",
        "target_session",
        "previous_session",
        "geometry_source_session",
        "previous_lag_days",
        "geometry_lag_days",
        "extra_lag_days_vs_previous",
        "previous_PER",
        "geometry_PER",
        "winner",
    ]
    preview = nonprevious[[c for c in cols if c in nonprevious.columns]].copy()
    for col in ["previous_PER", "geometry_PER"]:
        if col in preview.columns:
            preview[col] = (100 * preview[col]).map(lambda v: f"{v:.2f}%")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
