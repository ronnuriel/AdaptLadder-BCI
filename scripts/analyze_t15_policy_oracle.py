from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def session_stats(trials: pd.DataFrame, method: str) -> pd.DataFrame:
    out = (
        trials.groupby(["calibration_trials", "session"], as_index=False)
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


def load_policy_trials(path: Path, method: str, method_filter: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if method_filter is not None and "method" in frame.columns:
        frame = frame[frame["method"].astype(str) == method_filter].copy()
    return session_stats(frame, method)


def weighted_per(table: pd.DataFrame, method: str) -> float:
    return float(table[f"{method}_edit_distance"].sum() / table[f"{method}_num_phonemes"].sum())


def weighted_per_from_choice(table: pd.DataFrame, choice_col: str) -> float:
    edits = []
    phonemes = []
    for _, row in table.iterrows():
        method = row[choice_col]
        edits.append(row[f"{method}_edit_distance"])
        phonemes.append(row[f"{method}_num_phonemes"])
    return float(np.sum(edits) / np.sum(phonemes))


def add_oracle_counts(row: dict[str, object], frame: pd.DataFrame, choice_col: str, methods: list[str]) -> None:
    counts = frame[choice_col].value_counts().to_dict()
    for method in methods:
        row[f"oracle_uses_{method}"] = int(counts.get(method, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Session-level oracle over previous, geometry, and adapter policies.")
    parser.add_argument("--previous-trials", type=Path, default=Path("results/tables/t15_kshot_previous_source_trial_results.csv"))
    parser.add_argument("--geometry-trials", type=Path, default=Path("results/tables/t15_kshot_geometry_source_trial_results.csv"))
    parser.add_argument(
        "--adapter-trials",
        type=Path,
        default=Path("results/tables/_explore_input_layer_calibration_trials_K5_10_20_epochs20.csv"),
    )
    parser.add_argument("--adapter-method", default="input_layer")
    parser.add_argument("--output-joined", type=Path, default=Path("results/tables/_explore_t15_policy_oracle_joined.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/_explore_t15_policy_oracle_summary.csv"))
    args = parser.parse_args()

    previous = load_policy_trials(args.previous_trials, "previous")
    geometry = load_policy_trials(args.geometry_trials, "geometry")
    adapter = load_policy_trials(args.adapter_trials, "adapter", method_filter=args.adapter_method)

    table = previous.merge(
        geometry[
            [
                "calibration_trials",
                "session",
                "geometry_edit_distance",
                "geometry_num_phonemes",
                "geometry_PER",
            ]
        ],
        on=["calibration_trials", "session"],
        how="inner",
        validate="one_to_one",
    )
    table = table.merge(
        adapter[
            [
                "calibration_trials",
                "session",
                "adapter_edit_distance",
                "adapter_num_phonemes",
                "adapter_PER",
            ]
        ],
        on=["calibration_trials", "session"],
        how="left",
        validate="one_to_one",
    )

    rows = []
    for k, frame in table.groupby("calibration_trials"):
        frame = frame.copy()
        previous_per = weighted_per(frame, "previous")
        geometry_per = weighted_per(frame, "geometry")

        frame["oracle_pg"] = np.where(frame["geometry_PER"] < frame["previous_PER"], "geometry", "previous")
        oracle_pg_per = weighted_per_from_choice(frame, "oracle_pg")
        row: dict[str, object] = {
            "calibration_trials": int(k),
            "num_sessions": int(len(frame)),
            "previous_PER": previous_per,
            "geometry_PER": geometry_per,
            "oracle_previous_geometry_PER": oracle_pg_per,
            "oracle_pg_gain_vs_previous": previous_per - oracle_pg_per,
        }
        add_oracle_counts(row, frame, "oracle_pg", ["previous", "geometry"])

        valid = frame.dropna(subset=["adapter_PER"]).copy()
        if not valid.empty:
            previous_subset_per = weighted_per(valid, "previous")
            geometry_subset_per = weighted_per(valid, "geometry")
            adapter_per = weighted_per(valid, "adapter")
            valid["oracle_pga"] = valid[["previous_PER", "geometry_PER", "adapter_PER"]].idxmin(axis=1).str.replace(
                "_PER", "", regex=False
            )
            oracle_pga_per = weighted_per_from_choice(valid, "oracle_pga")
            row.update(
                {
                    "adapter_sessions": int(len(valid)),
                    "previous_subset_PER": previous_subset_per,
                    "geometry_subset_PER": geometry_subset_per,
                    "adapter_PER": adapter_per,
                    "oracle_previous_geometry_adapter_PER": oracle_pga_per,
                    "oracle_pga_gain_vs_previous_subset": previous_subset_per - oracle_pga_per,
                }
            )
            add_oracle_counts(row, valid, "oracle_pga", ["previous", "geometry", "adapter"])
        rows.append(row)

    summary = pd.DataFrame(rows)
    args.output_joined.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_joined, index=False)
    summary.to_csv(args.output_summary, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
