from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GAP_ORDER = ["short_0_3", "medium_4_14", "long_15_plus"]


def weighted_per_from_choice(frame: pd.DataFrame, choice_col: str) -> float:
    edits = []
    phones = []
    for _, row in frame.iterrows():
        method = row[choice_col]
        edits.append(row[f"{method}_edit_distance"])
        phones.append(row[f"{method}_num_phonemes"])
    return float(np.sum(edits) / np.sum(phones))


def add_policy_row(rows: list[dict[str, object]], frame: pd.DataFrame, policy: str, choice_col: str) -> None:
    rows.append(
        {
            "analysis": "K20_fair_remaining_trials",
            "policy": policy,
            "weighted_PER": weighted_per_from_choice(frame, choice_col),
            "n_sessions": int(len(frame)),
            "n_phonemes": int(sum(frame.loc[i, f"{row[choice_col]}_num_phonemes"] for i, row in frame.iterrows())),
        }
    )


def add_gap_rows(rows: list[dict[str, object]], frame: pd.DataFrame, policy: str, choice_col: str) -> None:
    for gap, group in frame.groupby("gap_bin", sort=False):
        rows.append(
            {
                "analysis": "K20_fair_remaining_trials",
                "gap_bin": gap,
                "policy": policy,
                "weighted_PER": weighted_per_from_choice(group, choice_col),
                "n_sessions": int(len(group)),
                "n_phonemes": int(
                    sum(group.loc[i, f"{row[choice_col]}_num_phonemes"] for i, row in group.iterrows())
                ),
            }
        )


def aggregate_trials(trials: pd.DataFrame) -> pd.DataFrame:
    out = (
        trials.groupby(["calibration_trials", "session", "method"], as_index=False)
        .agg(edit_distance=("edit_distance", "sum"), num_phonemes=("num_phonemes", "sum"))
    )
    out["PER"] = out["edit_distance"] / out["num_phonemes"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze gap-aware source/recalibration policies for T15.")
    parser.add_argument(
        "--k20-joined",
        type=Path,
        default=Path("results/tables/_explore_t15_gap_full_residual_l2_3e4_joined.csv"),
        help="Joined K=20 previous/geometry/residual table from gap adaptation analysis.",
    )
    parser.add_argument(
        "--k40-trials",
        type=Path,
        default=Path("results/tables/_explore_t15_full_residual_K40_80_batch10_trials.csv"),
    )
    parser.add_argument(
        "--previous-gap",
        type=Path,
        default=Path("results/tables/_explore_t15_previous_gap_sessions.csv"),
    )
    parser.add_argument(
        "--output-decisions",
        type=Path,
        default=Path("results/tables/_explore_t15_gap_aware_policy_decisions.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/_explore_t15_gap_aware_policy_summary.csv"),
    )
    parser.add_argument(
        "--output-by-gap",
        type=Path,
        default=Path("results/tables/_explore_t15_gap_aware_policy_by_gap.csv"),
    )
    parser.add_argument(
        "--output-k40",
        type=Path,
        default=Path("results/tables/_explore_t15_gap_aware_policy_k40_eligible.csv"),
    )
    parser.add_argument(
        "--output-long-cases",
        type=Path,
        default=Path("results/tables/_explore_t15_gap_aware_long_gap_cases.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/figures/_explore_t15_gap_aware_policy.png"),
    )
    args = parser.parse_args()

    table = pd.read_csv(args.k20_joined)
    # K=20 fair policies: every policy is evaluated on the same post-K=20 trial window.
    decisions = table.copy()
    decisions["choice_previous"] = "previous"
    decisions["choice_geometry"] = "geometry"
    decisions["choice_residual_k20"] = "input_layer"
    decisions["choice_gap_policy_k20"] = np.where(decisions["gap_bin"].eq("short_0_3"), "previous", "input_layer")
    decisions["choice_oracle_pgr"] = decisions[["previous_PER", "geometry_PER", "input_layer_PER"]].idxmin(axis=1).str.replace(
        "_PER", "", regex=False
    )

    policy_specs = [
        ("always_previous", "choice_previous"),
        ("always_geometry", "choice_geometry"),
        ("always_residual_K20", "choice_residual_k20"),
        ("gap_policy_K20_previous_short_residual_else", "choice_gap_policy_k20"),
        ("oracle_previous_geometry_residual", "choice_oracle_pgr"),
    ]
    summary_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    for policy, choice_col in policy_specs:
        add_policy_row(summary_rows, decisions, policy, choice_col)
        add_gap_rows(gap_rows, decisions, policy, choice_col)

    summary = pd.DataFrame(summary_rows)
    by_gap = pd.DataFrame(gap_rows)

    # K=40 eligible comparison: same sessions and same post-K=40 trial window.
    k40 = aggregate_trials(pd.read_csv(args.k40_trials))
    k40 = k40[k40["calibration_trials"].astype(int) == 40].copy()
    gap_meta = pd.read_csv(args.previous_gap)
    gap_meta = gap_meta[gap_meta["calibration_trials"].astype(int) == 20][
        ["session", "previous_session", "previous_abs_days", "previous_lag_sessions", "gap_bin"]
    ].drop_duplicates()
    k40 = k40.merge(gap_meta, on="session", how="left", validate="many_to_one")
    k40_summary_rows = []
    for method, group in k40.groupby("method"):
        k40_summary_rows.append(
            {
                "analysis": "K40_eligible_remaining_trials",
                "method": method,
                "weighted_PER": float(group["edit_distance"].sum() / group["num_phonemes"].sum()),
                "n_sessions": int(group["session"].nunique()),
                "n_phonemes": int(group["num_phonemes"].sum()),
            }
        )
    for (gap, method), group in k40.groupby(["gap_bin", "method"]):
        k40_summary_rows.append(
            {
                "analysis": f"K40_eligible_by_gap:{gap}",
                "method": method,
                "weighted_PER": float(group["edit_distance"].sum() / group["num_phonemes"].sum()),
                "n_sessions": int(group["session"].nunique()),
                "n_phonemes": int(group["num_phonemes"].sum()),
            }
        )
    k40_summary = pd.DataFrame(k40_summary_rows)

    # Long-gap case study, combining K=20 and K=40 where available.
    long = decisions[decisions["gap_bin"].eq("long_15_plus")].copy()
    long_cols = [
        "session",
        "previous_session",
        "previous_abs_days",
        "geometry_source_session",
        "selected_lag_days",
        "native_PER",
        "previous_PER",
        "geometry_PER",
        "input_layer_PER",
        "choice_oracle_pgr",
    ]
    long_cases = long[long_cols].rename(columns={"input_layer_PER": "residual_K20_PER"}).copy()
    k40_wide = k40.pivot_table(index="session", columns="method", values="PER", aggfunc="first").reset_index()
    k40_wide = k40_wide.rename(
        columns={"native-day": "native_K40_PER", "none": "previous_K40_PER", "layer_full": "residual_K40_PER"}
    )
    long_cases = long_cases.merge(k40_wide, on="session", how="left")

    for col in [
        "native_PER",
        "previous_PER",
        "geometry_PER",
        "residual_K20_PER",
        "native_K40_PER",
        "previous_K40_PER",
        "residual_K40_PER",
    ]:
        if col in long_cases.columns:
            long_cases[col] = long_cases[col].astype(float)

    for output in [args.output_decisions, args.output_summary, args.output_by_gap, args.output_k40, args.output_long_cases]:
        output.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(args.output_decisions, index=False)
    summary.to_csv(args.output_summary, index=False)
    by_gap.to_csv(args.output_by_gap, index=False)
    k40_summary.to_csv(args.output_k40, index=False)
    long_cases.to_csv(args.output_long_cases, index=False)

    # Compact figure for K=20 fair policies.
    plot = summary.copy()
    order = [p for p, _ in policy_specs]
    plot["policy"] = pd.Categorical(plot["policy"], categories=order, ordered=True)
    plot = plot.sort_values("policy")
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(plot["policy"].astype(str), plot["weighted_PER"], color="#4c78a8")
    ax.set_ylabel("Weighted PER")
    ax.set_xlabel("Policy")
    ax.set_ylim(0, max(plot["weighted_PER"]) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(args.output_figure, dpi=200)

    print("K=20 fair policy summary:")
    print(summary.sort_values("weighted_PER").to_string(index=False))
    print("\nK=20 by gap:")
    print(by_gap.sort_values(["gap_bin", "weighted_PER"]).to_string(index=False))
    print("\nK=40 eligible:")
    print(k40_summary.sort_values(["analysis", "weighted_PER"]).to_string(index=False))
    print("\nLong-gap cases:")
    display = long_cases.copy()
    per_cols = [c for c in display.columns if c.endswith("_PER")]
    for col in per_cols:
        display[col] = (display[col] * 100).round(2)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
