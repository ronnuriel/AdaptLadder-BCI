from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "selected_lag_days",
    "selected_lag_sessions",
    "previous_metric_value",
    "geometry_metric_value",
    "geometry_previous_distance_ratio",
    "margin_abs",
    "margin_fraction",
    "previous_mean_confidence",
    "previous_entropy",
    "previous_blank_rate",
    "geometry_minus_previous_mean_principal_angle_deg",
    "geometry_minus_previous_subspace_chordal_distance",
    "geometry_minus_previous_basis_procrustes_error",
    "geometry_minus_previous_cov_relative_fro_shift_from_source",
    "geometry_minus_previous_coral_distance_from_source",
]


def session_stats(trials: pd.DataFrame, method: str, calibration_trials: int) -> pd.DataFrame:
    frame = trials.copy()
    if "calibration_trials" in frame.columns:
        frame = frame[frame["calibration_trials"].astype(int) == int(calibration_trials)].copy()
    if "method" in frame.columns and method in set(frame["method"].astype(str)):
        frame = frame[frame["method"].astype(str) == method].copy()
    out = (
        frame.groupby("session", as_index=False)
        .agg(
            edit_distance=("edit_distance", "sum"),
            num_phonemes=("num_phonemes", "sum"),
        )
    )
    out[f"{method}_PER"] = out["edit_distance"] / out["num_phonemes"]
    return out.rename(
        columns={
            "edit_distance": f"{method}_edit_distance",
            "num_phonemes": f"{method}_num_phonemes",
        }
    )


def weighted_per(frame: pd.DataFrame, method: str) -> float:
    return float(frame[f"{method}_edit_distance"].sum() / frame[f"{method}_num_phonemes"].sum())


def weighted_per_from_choice(frame: pd.DataFrame, choice_col: str) -> float:
    edits = []
    phones = []
    for _, row in frame.iterrows():
        method = row[choice_col]
        edits.append(row[f"{method}_edit_distance"])
        phones.append(row[f"{method}_num_phonemes"])
    return float(np.sum(edits) / np.sum(phones))


def make_thresholds(values: pd.Series) -> list[float]:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return []
    quantiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    thresholds = [float(clean.quantile(q)) for q in quantiles]
    thresholds.extend(float(v) for v in clean.unique() if clean.nunique() <= 8)
    return sorted(set(thresholds))


def summarize_policy(frame: pd.DataFrame, policy_name: str, choice: pd.Series) -> dict[str, object]:
    work = frame.copy()
    work["choice"] = choice.to_numpy()
    per = weighted_per_from_choice(work, "choice")
    previous_per = weighted_per(work, "previous")
    adapter_per = weighted_per(work, "input_layer")
    adapter_count = int((work["choice"] == "input_layer").sum())
    correct_trigger = int(((work["choice"] == "input_layer") & (work["input_layer_PER"] < work["previous_PER"])).sum())
    harmful_trigger = int(((work["choice"] == "input_layer") & (work["input_layer_PER"] >= work["previous_PER"])).sum())
    missed_adapter_win = int(((work["choice"] == "previous") & (work["input_layer_PER"] < work["previous_PER"])).sum())
    return {
        "policy": policy_name,
        "weighted_PER": per,
        "gain_vs_previous": previous_per - per,
        "previous_PER": previous_per,
        "adapter_PER": adapter_per,
        "adapter_triggers": adapter_count,
        "correct_adapter_triggers": correct_trigger,
        "harmful_adapter_triggers": harmful_trigger,
        "missed_adapter_wins": missed_adapter_win,
        "num_sessions": int(len(work)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore previous-vs-calibration trigger policies for T15.")
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=Path("results/tables/t15_drift_type_fingerprint_joined.csv"),
    )
    parser.add_argument(
        "--previous-trials",
        type=Path,
        default=Path("results/tables/t15_kshot_previous_source_trial_results.csv"),
    )
    parser.add_argument(
        "--adapter-trials",
        type=Path,
        default=Path("results/tables/_explore_input_layer_calibration_trials_K20_epochs40.csv"),
    )
    parser.add_argument("--calibration-trials", type=int, default=20)
    parser.add_argument("--adapter-method", default="input_layer")
    parser.add_argument(
        "--output-decisions",
        type=Path,
        default=Path("results/tables/_explore_t15_calibration_trigger_decisions.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/_explore_t15_calibration_trigger_summary.csv"),
    )
    args = parser.parse_args()

    previous = session_stats(pd.read_csv(args.previous_trials), "previous", args.calibration_trials)
    adapter = session_stats(pd.read_csv(args.adapter_trials), args.adapter_method, args.calibration_trials)
    adapter = adapter.rename(
        columns={
            f"{args.adapter_method}_edit_distance": "input_layer_edit_distance",
            f"{args.adapter_method}_num_phonemes": "input_layer_num_phonemes",
            f"{args.adapter_method}_PER": "input_layer_PER",
        }
    )
    features = pd.read_csv(args.fingerprints)
    features = features[features["calibration_trials"].astype(int) == int(args.calibration_trials)].copy()

    table = previous.merge(adapter, on="session", how="inner", validate="one_to_one")
    table = table.merge(features, on="session", how="left", suffixes=("", "_fingerprint"))

    rows: list[dict[str, object]] = []
    decisions = []

    rows.append(summarize_policy(table, "always_previous", pd.Series("previous", index=table.index)))
    rows.append(summarize_policy(table, "always_calibrate", pd.Series("input_layer", index=table.index)))
    oracle_choice = np.where(table["input_layer_PER"] < table["previous_PER"], "input_layer", "previous")
    rows.append(summarize_policy(table, "oracle_previous_vs_calibration", pd.Series(oracle_choice, index=table.index)))

    for feature in [f for f in DEFAULT_FEATURES if f in table.columns]:
        values = table[feature].astype(float).replace([np.inf, -np.inf], np.nan)
        for threshold in make_thresholds(values):
            ok = values.notna()
            for direction, mask in [
                ("high", ok & (values >= threshold)),
                ("low", ok & (values <= threshold)),
            ]:
                choice = pd.Series("previous", index=table.index)
                choice.loc[mask] = "input_layer"
                summary = summarize_policy(table, f"{feature}_{direction}_{threshold:.6g}", choice)
                summary.update({"feature": feature, "direction": direction, "threshold": threshold})
                rows.append(summary)
                decision_frame = table[
                    [
                        "session",
                        "previous_PER",
                        "input_layer_PER",
                        feature,
                    ]
                ].copy()
                decision_frame["policy"] = summary["policy"]
                decision_frame["choice"] = choice
                decisions.append(decision_frame)

    summary = pd.DataFrame(rows).sort_values(["weighted_PER", "adapter_triggers", "policy"])
    decision_table = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    decision_table.to_csv(args.output_decisions, index=False)

    print("Best calibration-trigger policies:")
    print(summary.head(20).to_string(index=False))
    print("\nBaselines:")
    print(summary[summary["policy"].isin(["always_previous", "always_calibrate", "oracle_previous_vs_calibration"])].to_string(index=False))


if __name__ == "__main__":
    main()
