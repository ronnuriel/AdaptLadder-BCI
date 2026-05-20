#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CONDITIONS = [
    "Native-day",
    "Fixed early",
    "Fixed middle",
    "Fixed late",
    "Moment match",
    "Diag. affine K=20",
    "Input layer K=20",
    "Previous source K=20",
    "K-shot geometry K=20",
]


def weighted_session_per(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    required = {"session", "edit_distance", "num_phonemes"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{condition}: missing columns {sorted(missing)}")
    out = (
        frame.groupby("session", as_index=False)
        .agg(
            edit_distance=("edit_distance", "sum"),
            num_phonemes=("num_phonemes", "sum"),
        )
    )
    out["per"] = out["edit_distance"] / out["num_phonemes"]
    out["condition"] = condition
    return out[["session", "condition", "per", "edit_distance", "num_phonemes"]]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_condition_table(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    frames.append(weighted_session_per(load_csv(args.native_trials), "Native-day"))
    frames.append(weighted_session_per(load_csv(args.fixed_early_trials), "Fixed early"))
    frames.append(weighted_session_per(load_csv(args.fixed_middle_trials), "Fixed middle"))
    frames.append(weighted_session_per(load_csv(args.fixed_late_trials), "Fixed late"))

    adaptation = load_csv(args.adaptation_trials)
    frames.append(
        weighted_session_per(
            adaptation[adaptation["adaptation_method"].astype(str) == "moment_match_to_source"].copy(),
            "Moment match",
        )
    )

    affine = load_csv(args.affine_trials)
    frames.append(
        weighted_session_per(
            affine[
                (affine["method"].astype(str) == "diagonal_affine")
                & (affine["calibration_trials"].astype(int) == args.calibration_trials)
            ].copy(),
            f"Diag. affine K={args.calibration_trials}",
        )
    )

    input_layer = load_csv(args.input_layer_trials)
    frames.append(
        weighted_session_per(
            input_layer[
                (input_layer["method"].astype(str) == "input_layer")
                & (input_layer["calibration_trials"].astype(int) == args.calibration_trials)
            ].copy(),
            f"Input layer K={args.calibration_trials}",
        )
    )

    previous = load_csv(args.previous_trials)
    frames.append(
        weighted_session_per(
            previous[previous["calibration_trials"].astype(int) == args.calibration_trials].copy(),
            f"Previous source K={args.calibration_trials}",
        )
    )

    geometry = load_csv(args.geometry_trials)
    frames.append(
        weighted_session_per(
            geometry[geometry["calibration_trials"].astype(int) == args.calibration_trials].copy(),
            f"K-shot geometry K={args.calibration_trials}",
        )
    )

    table = pd.concat(frames, ignore_index=True)
    table["session_short"] = table["session"].str.replace("t15.2025.", "", regex=False)
    table["session_short"] = table["session_short"].str.replace("t15.2024.", "", regex=False)
    table["session_short"] = table["session_short"].str.replace("t15.2023.", "", regex=False)
    return table


def keep_mask(frame: pd.DataFrame, drop_tokens: list[str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for token in drop_tokens:
        mask &= ~frame["session"].astype(str).str.contains(token, regex=False)
    return mask


def summarize(frame: pd.DataFrame, analysis: str, drop_tokens: list[str]) -> pd.DataFrame:
    sub = frame[keep_mask(frame, drop_tokens)].copy()
    rows: list[dict[str, object]] = []
    for condition, group in sub.groupby("condition", sort=False):
        best_idx = group["per"].idxmin()
        worst_idx = group["per"].idxmax()
        rows.append(
            {
                "analysis": analysis,
                "condition": condition,
                "n_sessions": int(group["session"].nunique()),
                "weighted_per": float(group["edit_distance"].sum() / group["num_phonemes"].sum()),
                "mean_session_per": float(group["per"].mean()),
                "median_session_per": float(group["per"].median()),
                "best_per": float(group.loc[best_idx, "per"]),
                "best_session": group.loc[best_idx, "session"],
                "worst_per": float(group.loc[worst_idx, "per"]),
                "worst_session": group.loc[worst_idx, "session"],
            }
        )
    return pd.DataFrame(rows)


def format_percent_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["weighted_per", "mean_session_per", "median_session_per", "best_per", "worst_per"]:
        out[col] = (100 * out[col]).map(lambda value: f"{value:.2f}%")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for whether hard T15 sessions dominate adaptation conclusions."
    )
    parser.add_argument("--calibration-trials", type=int, default=20)
    parser.add_argument("--native-trials", type=Path, default=Path("results/tables/t15_decoder_probe_val.csv"))
    parser.add_argument(
        "--fixed-early-trials",
        type=Path,
        default=Path("results/tables/t15_decoder_probe_cross_day_source_2023_08_13_val.csv"),
    )
    parser.add_argument(
        "--fixed-middle-trials",
        type=Path,
        default=Path("results/tables/t15_decoder_probe_cross_day_source_2023_11_26_val.csv"),
    )
    parser.add_argument(
        "--fixed-late-trials",
        type=Path,
        default=Path("results/tables/t15_decoder_probe_cross_day_source_2025_04_13_val.csv"),
    )
    parser.add_argument(
        "--adaptation-trials",
        type=Path,
        default=Path("results/tables/t15_adaptation_trial_results_source_middle.csv"),
    )
    parser.add_argument(
        "--affine-trials",
        type=Path,
        default=Path("results/tables/t15_affine_calibration_trial_results_source_middle_epochs5.csv"),
    )
    parser.add_argument(
        "--input-layer-trials",
        type=Path,
        default=Path("results/tables/t15_input_layer_calibration_trial_results_source_middle_K20_epochs20.csv"),
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
        "--output-per-session",
        type=Path,
        default=Path("results/tables/t15_per_session_conditions.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/t15_outlier_sensitivity_summary.csv"),
    )
    args = parser.parse_args()

    per_session = build_condition_table(args)
    exclusion_sets = {
        "all_sessions": [],
        "drop_03_30": ["2025.03.30"],
        "drop_03_30_and_03_14": ["2025.03.30", "2025.03.14"],
    }
    summary = pd.concat(
        [summarize(per_session, name, drops) for name, drops in exclusion_sets.items()],
        ignore_index=True,
    )

    args.output_per_session.parent.mkdir(parents=True, exist_ok=True)
    per_session.to_csv(args.output_per_session, index=False)
    summary.to_csv(args.output_summary, index=False)

    print(f"Wrote {args.output_per_session}")
    print(f"Wrote {args.output_summary}")
    print()
    print(format_percent_columns(summary).to_string(index=False))


if __name__ == "__main__":
    main()
