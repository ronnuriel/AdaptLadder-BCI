from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["calibration_trials", "session", "block", "trial"]


def weighted_per(frame: pd.DataFrame, prefix: str) -> float:
    edits = frame[f"{prefix}_edit_distance"].sum()
    phonemes = frame[f"{prefix}_num_phonemes"].sum()
    return float(edits / phonemes) if phonemes else float("nan")


def add_prefix(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = KEYS + ["edit_distance", "num_phonemes", "PER", "blank_rate", "mean_confidence", "entropy"]
    available = [column for column in keep if column in frame.columns]
    renamed = {
        column: f"{prefix}_{column}"
        for column in available
        if column not in KEYS
    }
    return frame[available].rename(columns=renamed)


def load_policy_trials(path: Path, prefix: str, method: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if method is not None:
        frame = frame[frame["method"] == method].copy()
    return add_prefix(frame, prefix)


def session_table(frame: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (k, session), session_frame in frame.groupby(["calibration_trials", "session"]):
        row: dict[str, object] = {"calibration_trials": int(k), "session": session}
        for method in methods:
            edits = session_frame[f"{method}_edit_distance"].sum()
            phonemes = session_frame[f"{method}_num_phonemes"].sum()
            row[f"{method}_edit_distance"] = float(edits)
            row[f"{method}_num_phonemes"] = float(phonemes)
            row[f"{method}_weighted_PER"] = float(edits / phonemes) if phonemes else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def per_from_session_choice(sessions: pd.DataFrame, method_by_session: np.ndarray) -> float:
    edits = np.zeros(len(sessions), dtype=float)
    phonemes = np.zeros(len(sessions), dtype=float)
    for method in sorted(set(method_by_session)):
        mask = method_by_session == method
        edits[mask] = sessions.loc[mask, f"{method}_edit_distance"].to_numpy(dtype=float)
        phonemes[mask] = sessions.loc[mask, f"{method}_num_phonemes"].to_numpy(dtype=float)
    return float(edits.sum() / phonemes.sum())


def oracle_choice(sessions: pd.DataFrame, methods: list[str]) -> np.ndarray:
    per_columns = [f"{method}_weighted_PER" for method in methods]
    values = sessions[per_columns].to_numpy(dtype=float)
    winner_idx = np.nanargmin(values, axis=1)
    return np.asarray(methods, dtype=object)[winner_idx]


def summarize_oracles(frame: pd.DataFrame, methods: list[str], oracle_name: str, scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sessions = session_table(frame, methods)
    for k, k_sessions in sessions.groupby("calibration_trials"):
        for method in methods:
            rows.append(
                {
                    "scope": scope,
                    "calibration_trials": int(k),
                    "policy": method,
                    "weighted_PER": float(
                        k_sessions[f"{method}_edit_distance"].sum() / k_sessions[f"{method}_num_phonemes"].sum()
                    ),
                    "num_sessions": int(len(k_sessions)),
                }
            )
        choice = oracle_choice(k_sessions, methods)
        rows.append(
            {
                "scope": scope,
                "calibration_trials": int(k),
                "policy": oracle_name,
                "weighted_PER": per_from_session_choice(k_sessions, choice),
                "num_sessions": int(len(k_sessions)),
            }
        )
        counts = pd.Series(choice).value_counts().to_dict()
        for method in methods:
            rows[-1][f"oracle_chose_{method}"] = int(counts.get(method, 0))
    return rows


def add_reference_columns(summary: pd.DataFrame) -> pd.DataFrame:
    previous = summary[summary["policy"] == "previous"][["scope", "calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "previous_weighted_PER"}
    )
    native = summary[summary["policy"] == "native"][["scope", "calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "native_weighted_PER"}
    )
    summary = summary.merge(previous, on=["scope", "calibration_trials"], how="left")
    summary = summary.merge(native, on=["scope", "calibration_trials"], how="left")
    summary["gain_vs_previous"] = summary["previous_weighted_PER"] - summary["weighted_PER"]
    gap = summary["previous_weighted_PER"] - summary["native_weighted_PER"]
    summary["gap_fraction_vs_previous_native"] = summary["gain_vs_previous"] / gap.replace(0, np.nan)
    return summary


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = [
        "native",
        "none",
        "previous",
        "geometry",
        "input_layer",
        "oracle_previous_geometry",
        "oracle_previous_geometry_input_layer",
    ]
    colors = {
        "native": "#2a9d8f",
        "none": "#8d99ae",
        "previous": "#e76f51",
        "geometry": "#457b9d",
        "input_layer": "#f4a261",
        "oracle_previous_geometry": "#264653",
        "oracle_previous_geometry_input_layer": "#6d597a",
    }
    scopes = list(summary["scope"].drop_duplicates())
    fig, axes = plt.subplots(1, len(scopes), figsize=(8.8, 4.2), sharey=True)
    if len(scopes) == 1:
        axes = [axes]
    for ax, scope in zip(axes, scopes, strict=True):
        scope_summary = summary[summary["scope"] == scope]
        present = [policy for policy in order if policy in set(scope_summary["policy"])]
        ks = sorted(scope_summary["calibration_trials"].unique())
        x = np.arange(len(ks))
        width = min(0.11, 0.75 / max(len(present), 1))
        offsets = (np.arange(len(present)) - (len(present) - 1) / 2) * width
        for offset, policy in zip(offsets, present, strict=True):
            values = []
            for k in ks:
                row = scope_summary[(scope_summary["calibration_trials"] == k) & (scope_summary["policy"] == policy)]
                values.append(float(row["weighted_PER"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + offset, values, width=width, label=policy.replace("_", " "), color=colors.get(policy))
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in ks])
        ax.set_title(scope.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Phoneme-weighted PER")
    axes[-1].legend(frameon=False, fontsize=8, ncol=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle policy analysis for T15 source retrieval and adapters.")
    parser.add_argument("--previous-trials", type=Path, default=Path("results/tables/t15_kshot_previous_source_trial_results.csv"))
    parser.add_argument("--geometry-trials", type=Path, default=Path("results/tables/t15_kshot_geometry_source_trial_results.csv"))
    parser.add_argument(
        "--adapter-trials",
        type=Path,
        default=Path("results/tables/_explore_input_layer_calibration_trials_K5_10_20_epochs20.csv"),
    )
    parser.add_argument("--output-trials", type=Path, default=Path("results/tables/_explore_t15_oracle_policy_trials.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/_explore_t15_oracle_policy_summary.csv"))
    parser.add_argument("--output-figure", type=Path, default=Path("results/figures/_explore_t15_oracle_policy_per.png"))
    args = parser.parse_args()

    previous = load_policy_trials(args.previous_trials, "previous")
    geometry = load_policy_trials(args.geometry_trials, "geometry")
    adapter_native = load_policy_trials(args.adapter_trials, "native", method="native-day")
    adapter_none = load_policy_trials(args.adapter_trials, "none", method="none")
    adapter_input = load_policy_trials(args.adapter_trials, "input_layer", method="input_layer")

    prev_geom = previous.merge(geometry, on=KEYS, how="inner", validate="one_to_one")
    rows = summarize_oracles(prev_geom, ["previous", "geometry"], "oracle_previous_geometry", "previous_geometry_full")

    with_adapter = (
        prev_geom.merge(adapter_native, on=KEYS, how="inner", validate="one_to_one")
        .merge(adapter_none, on=KEYS, how="inner", validate="one_to_one")
        .merge(adapter_input, on=KEYS, how="inner", validate="one_to_one")
    )
    rows.extend(summarize_oracles(with_adapter, ["previous", "geometry"], "oracle_previous_geometry", "adapter_intersection"))
    rows.extend(
        summarize_oracles(with_adapter, ["previous", "geometry", "input_layer"], "oracle_previous_geometry_input_layer", "adapter_intersection")
    )
    rows.extend(summarize_oracles(with_adapter, ["native", "previous", "geometry", "input_layer"], "oracle_with_native", "adapter_intersection"))

    summary = add_reference_columns(pd.DataFrame(rows).drop_duplicates(["scope", "calibration_trials", "policy"], keep="last"))
    summary = summary.sort_values(["scope", "calibration_trials", "policy"]).reset_index(drop=True)

    args.output_trials.parent.mkdir(parents=True, exist_ok=True)
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    prev_geom.to_csv(args.output_trials, index=False)
    summary.to_csv(args.output_summary, index=False)
    plot_summary(summary, args.output_figure)

    print(f"Wrote {args.output_trials}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_figure}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
