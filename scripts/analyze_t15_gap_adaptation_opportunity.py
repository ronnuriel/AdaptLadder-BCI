from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gap_bin(days: float) -> str:
    if days <= 3:
        return "short_0_3"
    if days <= 14:
        return "medium_4_14"
    return "long_15_plus"


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


def add_policy_row(rows: list[dict[str, object]], frame: pd.DataFrame, gap_name: str, method: str) -> None:
    rows.append(
        {
            "gap_bin": gap_name,
            "method": method,
            "weighted_PER": weighted_per(frame, method),
            "n_sessions": int(len(frame)),
            "n_phonemes": int(frame[f"{method}_num_phonemes"].sum()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare adaptation policies within previous-source gap bins.")
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
        "--geometry-trials",
        type=Path,
        default=Path("results/tables/t15_kshot_geometry_source_trial_results.csv"),
    )
    parser.add_argument(
        "--adapter-trials",
        type=Path,
        default=Path("results/tables/_explore_input_layer_calibration_trials_K20_epochs40.csv"),
    )
    parser.add_argument("--calibration-trials", type=int, default=20)
    parser.add_argument("--adapter-method", default="input_layer")
    parser.add_argument(
        "--output-joined",
        type=Path,
        default=Path("results/tables/_explore_t15_gap_adaptation_opportunity_joined.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/_explore_t15_gap_adaptation_opportunity_summary.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/figures/_explore_t15_gap_adaptation_opportunity.png"),
    )
    args = parser.parse_args()

    previous = session_stats(pd.read_csv(args.previous_trials), "previous", args.calibration_trials)
    geometry = session_stats(pd.read_csv(args.geometry_trials), "geometry", args.calibration_trials)
    adapter_raw = pd.read_csv(args.adapter_trials)
    adapter = session_stats(adapter_raw, args.adapter_method, args.calibration_trials).rename(
        columns={
            f"{args.adapter_method}_edit_distance": "input_layer_edit_distance",
            f"{args.adapter_method}_num_phonemes": "input_layer_num_phonemes",
            f"{args.adapter_method}_PER": "input_layer_PER",
        }
    )
    native = session_stats(adapter_raw, "native-day", args.calibration_trials)
    native = native.rename(
        columns={
            "native-day_edit_distance": "native_edit_distance",
            "native-day_num_phonemes": "native_num_phonemes",
            "native-day_PER": "native_PER",
        }
    )

    fingerprints = pd.read_csv(args.fingerprints)
    fingerprints = fingerprints[fingerprints["calibration_trials"].astype(int) == int(args.calibration_trials)].copy()
    meta = fingerprints[
        [
            "session",
            "previous_session",
            "previous_abs_days",
            "previous_lag_sessions",
            "geometry_source_session",
            "selected_lag_days",
        ]
    ].drop_duplicates()
    meta["gap_bin"] = meta["previous_abs_days"].astype(float).map(gap_bin)

    table = previous.merge(geometry, on="session", how="inner", validate="one_to_one")
    table = table.merge(adapter, on="session", how="inner", validate="one_to_one")
    table = table.merge(native, on="session", how="inner", validate="one_to_one")
    table = table.merge(meta, on="session", how="left", validate="one_to_one")

    oracle_methods = ["previous", "geometry", "input_layer"]
    table["oracle_pga"] = table[[f"{m}_PER" for m in oracle_methods]].idxmin(axis=1).str.replace(
        "_PER", "", regex=False
    )
    table["best_non_native"] = table["oracle_pga"]

    rows: list[dict[str, object]] = []
    for gap_name, frame in table.groupby("gap_bin", sort=False):
        if frame.empty:
            continue
        for method in ["native", "previous", "geometry", "input_layer"]:
            add_policy_row(rows, frame, gap_name, method)
        rows.append(
            {
                "gap_bin": gap_name,
                "method": "oracle_previous_geometry_calibration",
                "weighted_PER": weighted_per_from_choice(frame, "oracle_pga"),
                "n_sessions": int(len(frame)),
                "n_phonemes": int(sum(frame.loc[i, f"{row.oracle_pga}_num_phonemes"] for i, row in frame.iterrows())),
            }
        )
    summary = pd.DataFrame(rows)

    args.output_joined.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_joined, index=False)
    summary.to_csv(args.output_summary, index=False)

    order = ["short_0_3", "medium_4_14", "long_15_plus"]
    method_order = ["native", "previous", "geometry", "input_layer", "oracle_previous_geometry_calibration"]
    plot = summary.pivot(index="gap_bin", columns="method", values="weighted_PER").reindex(order)
    plot = plot[[m for m in method_order if m in plot.columns]]

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    ax = plot.plot(kind="bar", figsize=(9, 4), width=0.82)
    ax.set_ylabel("Weighted PER")
    ax.set_xlabel("Gap from previous source")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(args.output_figure, dpi=200)

    print("Gap-binned adaptation opportunity:")
    print(summary.sort_values(["gap_bin", "weighted_PER"]).to_string(index=False))
    print(f"\nWrote {args.output_figure}")


if __name__ == "__main__":
    main()
