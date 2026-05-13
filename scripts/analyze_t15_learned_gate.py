from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "geometry_previous_distance_ratio",
    "margin_fraction",
    "margin_abs",
    "previous_metric_value",
    "geometry_metric_value",
    "selected_lag_days",
    "selected_lag_sessions",
    "lag_diff_days",
    "lag_diff_sessions",
]


def weighted_per(frame: pd.DataFrame, prefix: str) -> float:
    return float(frame[f"{prefix}_edit_distance"].sum() / frame[f"{prefix}_num_phonemes"].sum())


def session_trial_stats(trials: pd.DataFrame, prefix: str) -> pd.DataFrame:
    stats = (
        trials.groupby(["calibration_trials", "session"], as_index=False)
        .agg(
            edit_distance=("edit_distance", "sum"),
            num_phonemes=("num_phonemes", "sum"),
            mean_confidence=("mean_confidence", "mean"),
            entropy=("entropy", "mean"),
            blank_rate=("blank_rate", "mean"),
        )
        .rename(
            columns={
                "edit_distance": f"{prefix}_edit_distance",
                "num_phonemes": f"{prefix}_num_phonemes",
                "mean_confidence": f"{prefix}_mean_confidence",
                "entropy": f"{prefix}_entropy",
                "blank_rate": f"{prefix}_blank_rate",
            }
        )
    )
    stats[f"{prefix}_weighted_PER"] = stats[f"{prefix}_edit_distance"] / stats[f"{prefix}_num_phonemes"]
    return stats


def build_session_table(
    comparison_path: Path,
    previous_trials_path: Path,
    geometry_trials_path: Path,
    override_decisions_path: Path,
) -> pd.DataFrame:
    comparison = pd.read_csv(comparison_path)
    previous_trials = pd.read_csv(previous_trials_path)
    geometry_trials = pd.read_csv(geometry_trials_path)
    decisions = pd.read_csv(override_decisions_path)

    # Alpha does not matter for these features; each alpha has the same distances.
    decision_features = (
        decisions[decisions["alpha"] == decisions["alpha"].max()]
        .rename(columns={"target_session": "session"})
        [
            [
                "calibration_trials",
                "session",
                "previous_metric_value",
                "geometry_metric_value",
                "geometry_previous_distance_ratio",
                "geometry_is_previous",
            ]
        ]
    )

    table = comparison.merge(decision_features, on=["calibration_trials", "session"], how="left")
    table = table.merge(
        session_trial_stats(previous_trials, "previous"),
        on=["calibration_trials", "session"],
        how="left",
    )
    table = table.merge(
        session_trial_stats(geometry_trials, "geometry"),
        on=["calibration_trials", "session"],
        how="left",
    )
    table["geometry_better_weighted"] = table["geometry_weighted_PER"] < table["previous_weighted_PER"]
    table["margin_abs"] = table["previous_metric_value"] - table["geometry_metric_value"]
    table["margin_fraction"] = 1.0 - table["geometry_previous_distance_ratio"]
    table["lag_diff_days"] = table["selected_lag_days"] - table["previous_abs_days"]
    table["lag_diff_sessions"] = table["selected_lag_sessions"] - table["previous_lag_sessions"]
    table["confidence_gain"] = table["geometry_mean_confidence"] - table["previous_mean_confidence"]
    table["entropy_drop"] = table["previous_entropy"] - table["geometry_entropy"]
    table["blank_rate_drop"] = table["previous_blank_rate"] - table["geometry_blank_rate"]
    return table


def candidate_thresholds(values: pd.Series) -> list[float]:
    unique = np.sort(values.dropna().unique())
    if len(unique) == 0:
        return []
    thresholds = list(float(v) for v in unique)
    if len(unique) > 1:
        thresholds.extend(float(v) for v in (unique[:-1] + unique[1:]) / 2.0)
    return thresholds


def evaluate_rule(frame: pd.DataFrame, feature: str, direction: str, threshold: float) -> tuple[float, int]:
    if direction == "<=":
        use_geometry = frame[feature] <= threshold
    elif direction == ">=":
        use_geometry = frame[feature] >= threshold
    else:
        raise ValueError(direction)
    use_geometry = use_geometry & (~frame["selected_is_previous"])
    edit_distance = np.where(use_geometry, frame["geometry_edit_distance"], frame["previous_edit_distance"]).sum()
    num_phonemes = np.where(use_geometry, frame["geometry_num_phonemes"], frame["previous_num_phonemes"]).sum()
    return float(edit_distance / num_phonemes), int(use_geometry.sum())


def fit_stump(train: pd.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    best: tuple[float, int, str, str, float] | None = None
    non_previous = train[~train["selected_is_previous"]]
    if non_previous.empty:
        return {"feature": "none", "direction": "<=", "threshold": np.nan, "train_weighted_PER": weighted_per(train, "previous")}

    for feature in feature_columns:
        for threshold in candidate_thresholds(non_previous[feature]):
            for direction in ["<=", ">="]:
                per, overrides = evaluate_rule(train, feature, direction, threshold)
                candidate = (per, -overrides, feature, direction, threshold)
                if best is None or candidate < best:
                    best = candidate

    assert best is not None
    return {
        "feature": best[2],
        "direction": best[3],
        "threshold": best[4],
        "train_weighted_PER": best[0],
    }


def apply_rule(row: pd.Series, rule: dict[str, object]) -> bool:
    if row["selected_is_previous"] or rule["feature"] == "none":
        return False
    value = row[str(rule["feature"])]
    if pd.isna(value):
        return False
    if rule["direction"] == "<=":
        return bool(value <= float(rule["threshold"]))
    return bool(value >= float(rule["threshold"]))


def leave_one_session_out_gate(session_table: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = []
    for k, frame in session_table.groupby("calibration_trials"):
        for idx, row in frame.iterrows():
            train = frame.drop(idx)
            rule = fit_stump(train, feature_columns)
            use_geometry = apply_rule(row, rule)
            decisions.append(
                {
                    "calibration_trials": int(k),
                    "session": row["session"],
                    "use_geometry": use_geometry,
                    "chosen_source_session": row["geometry_source_session"] if use_geometry else row["previous_source_session"],
                    "geometry_source_session": row["geometry_source_session"],
                    "previous_source_session": row["previous_source_session"],
                    "selected_is_previous": bool(row["selected_is_previous"]),
                    "geometry_better_weighted": bool(row["geometry_better_weighted"]),
                    "rule_feature": rule["feature"],
                    "rule_direction": rule["direction"],
                    "rule_threshold": rule["threshold"],
                    "previous_edit_distance": row["previous_edit_distance"],
                    "previous_num_phonemes": row["previous_num_phonemes"],
                    "geometry_edit_distance": row["geometry_edit_distance"],
                    "geometry_num_phonemes": row["geometry_num_phonemes"],
                }
            )
    decisions_df = pd.DataFrame(decisions)

    rows = []
    for k, decisions_k in decisions_df.groupby("calibration_trials"):
        source_rows = session_table[session_table["calibration_trials"] == k]
        previous_per = weighted_per(source_rows, "previous")
        geometry_per = weighted_per(source_rows, "geometry")
        use_geometry = decisions_k["use_geometry"].to_numpy(dtype=bool)
        gate_edit_distance = np.where(
            use_geometry,
            decisions_k["geometry_edit_distance"],
            decisions_k["previous_edit_distance"],
        ).sum()
        gate_num_phonemes = np.where(
            use_geometry,
            decisions_k["geometry_num_phonemes"],
            decisions_k["previous_num_phonemes"],
        ).sum()
        gate_per = float(gate_edit_distance / gate_num_phonemes)
        rows.append(
            {
                "calibration_trials": int(k),
                "previous_weighted_PER": previous_per,
                "geometry_weighted_PER": geometry_per,
                "learned_gate_weighted_PER": gate_per,
                "gain_vs_previous": previous_per - gate_per,
                "num_sessions": int(len(decisions_k)),
                "nonprevious_candidates": int((~source_rows["selected_is_previous"]).sum()),
                "overrides_used": int(decisions_k["use_geometry"].sum()),
                "correct_overrides": int((decisions_k["use_geometry"] & decisions_k["geometry_better_weighted"]).sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("calibration_trials")
    return decisions_df, summary


def margin_summary(session_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, frame in session_table.groupby("calibration_trials"):
        non_previous = frame[~frame["selected_is_previous"]].copy()
        if len(non_previous) >= 3:
            rho = non_previous[["geometry_previous_distance_ratio", "geometry_minus_previous_mean_PER"]].corr(
                method="spearman"
            ).iloc[0, 1]
        else:
            rho = np.nan
        rows.append(
            {
                "calibration_trials": int(k),
                "nonprevious_candidates": int(len(non_previous)),
                "older_source_wins": int(non_previous["geometry_better_weighted"].sum()),
                "spearman_ratio_vs_mean_per_delta": float(rho) if pd.notna(rho) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_gate(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ks = summary["calibration_trials"].tolist()
    x = np.arange(len(ks))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width, summary["previous_weighted_PER"], width=width, label="previous", color="#e76f51")
    ax.bar(x, summary["geometry_weighted_PER"], width=width, label="geometry", color="#457b9d")
    ax.bar(x + width, summary["learned_gate_weighted_PER"], width=width, label="LOO stump gate", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Phoneme-weighted PER")
    ax.set_title("Recency-aware learned gate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a small recency-aware learned gate for T15 source selection.")
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/tables/t15_kshot_previous_vs_geometry_session_comparison.csv"),
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
        "--override-decisions",
        type=Path,
        default=Path("results/tables/t15_kshot_recency_geometry_override_decisions.csv"),
    )
    parser.add_argument(
        "--output-session-table",
        type=Path,
        default=Path("results/tables/t15_kshot_learned_gate_session_table.csv"),
    )
    parser.add_argument(
        "--output-decisions",
        type=Path,
        default=Path("results/tables/t15_kshot_learned_gate_decisions.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/t15_kshot_learned_gate_summary.csv"),
    )
    parser.add_argument(
        "--output-margin-summary",
        type=Path,
        default=Path("results/tables/t15_kshot_source_selection_margin_summary.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/figures/t15_kshot_learned_gate_per.png"),
    )
    args = parser.parse_args()

    session_table = build_session_table(
        comparison_path=args.comparison,
        previous_trials_path=args.previous_trials,
        geometry_trials_path=args.geometry_trials,
        override_decisions_path=args.override_decisions,
    )
    decisions, summary = leave_one_session_out_gate(session_table, FEATURE_COLUMNS)
    margins = margin_summary(session_table)

    for path in [
        args.output_session_table,
        args.output_decisions,
        args.output_summary,
        args.output_margin_summary,
        args.output_figure,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    session_table.to_csv(args.output_session_table, index=False)
    decisions.to_csv(args.output_decisions, index=False)
    summary.to_csv(args.output_summary, index=False)
    margins.to_csv(args.output_margin_summary, index=False)
    plot_gate(summary, args.output_figure)

    print(f"Wrote {args.output_session_table}")
    print(f"Wrote {args.output_decisions}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_margin_summary}")
    print(f"Wrote {args.output_figure}")
    print(summary.to_string(index=False))
    print(margins.to_string(index=False))


if __name__ == "__main__":
    main()
