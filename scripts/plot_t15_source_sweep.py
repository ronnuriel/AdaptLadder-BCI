from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t15_utils import session_date


def weighted_per(trials: pd.DataFrame) -> float:
    return float(trials["edit_distance"].sum() / trials["num_phonemes"].sum())


def source_token(source_session: str) -> str:
    return source_session.replace("t15.", "").replace(".", "_")


def label_sources(sources: list[str]) -> dict[str, str]:
    ordered = sorted(sources, key=session_date)
    if len(ordered) == 1:
        return {ordered[0]: "source"}
    if len(ordered) == 2:
        return {ordered[0]: "early", ordered[1]: "late"}
    labels = {source: f"source_{idx + 1}" for idx, source in enumerate(ordered)}
    labels[ordered[0]] = "early"
    labels[ordered[len(ordered) // 2]] = "middle"
    labels[ordered[-1]] = "late"
    return labels


def find_cross_trial_files(tables_dir: Path) -> list[Path]:
    return sorted(tables_dir.glob("t15_decoder_probe_cross_day_source_*_val.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the T15 early/middle/late source-session sweep.")
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--native-trials", type=Path, default=Path("results/tables/t15_decoder_probe_val.csv"))
    parser.add_argument("--native-summary", type=Path, default=Path("results/tables/t15_decoder_probe_session_summary.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/t15_source_sweep_summary.csv"))
    parser.add_argument("--output-joined", type=Path, default=Path("results/tables/t15_source_sweep_joined.csv"))
    args = parser.parse_args()

    native_trials = pd.read_csv(args.native_trials)
    native = pd.read_csv(args.native_summary)
    native_weighted = weighted_per(native_trials)

    native_by_session = native[
        ["session", "n_trials", "mean_PER", "median_PER", "mean_blank_rate", "mean_confidence", "mean_entropy"]
    ].rename(
        columns={
            "mean_PER": "native_mean_PER",
            "median_PER": "native_median_PER",
            "mean_blank_rate": "native_mean_blank_rate",
            "mean_confidence": "native_mean_confidence",
            "mean_entropy": "native_mean_entropy",
        }
    )

    cross_trial_files = find_cross_trial_files(args.tables_dir)
    if not cross_trial_files:
        raise FileNotFoundError(f"No cross-day trial files found in {args.tables_dir}")

    summary_rows = []
    joined_frames = []
    for trial_path in cross_trial_files:
        trials = pd.read_csv(trial_path)
        source_sessions = sorted(trials["input_layer_session"].dropna().unique())
        if len(source_sessions) != 1:
            raise ValueError(f"Expected one source in {trial_path}, found {source_sessions}")
        source = source_sessions[0]

        summary_path = args.tables_dir / f"t15_decoder_probe_cross_day_source_{source_token(source)}_session_summary.csv"
        if not summary_path.exists():
            # Keep this readable if the filename convention changes.
            token_match = re.search(r"source_(.+)_val\\.csv$", trial_path.name)
            raise FileNotFoundError(f"Missing session summary for {source} ({token_match.group(1) if token_match else trial_path})")
        cross = pd.read_csv(summary_path)

        joined = native_by_session.merge(
            cross[["session", "input_layer_session", "mean_PER", "median_PER", "mean_blank_rate", "mean_confidence", "mean_entropy"]],
            on="session",
            how="inner",
        ).rename(
            columns={
                "input_layer_session": "source_session",
                "mean_PER": "cross_mean_PER",
                "median_PER": "cross_median_PER",
                "mean_blank_rate": "cross_mean_blank_rate",
                "mean_confidence": "cross_mean_confidence",
                "mean_entropy": "cross_mean_entropy",
            }
        )
        joined["target_date"] = joined["session"].map(lambda value: session_date(value).isoformat())
        joined["source_date"] = session_date(source).isoformat()
        joined["days_from_source"] = joined["session"].map(lambda value: (session_date(value) - session_date(source)).days)
        joined["abs_days_from_source"] = joined["days_from_source"].abs()
        joined["delta_mean_PER"] = joined["cross_mean_PER"] - joined["native_mean_PER"]
        joined["delta_median_PER"] = joined["cross_median_PER"] - joined["native_median_PER"]
        joined["ratio_mean_PER"] = joined["cross_mean_PER"] / joined["native_mean_PER"]
        joined_frames.append(joined)

        source_weighted = weighted_per(trials)
        summary_rows.append(
            {
                "source_session": source,
                "weighted_PER": source_weighted,
                "delta_vs_native": source_weighted - native_weighted,
                "trial_mean_PER": float(trials["PER"].mean()),
                "harmed_sessions": int((joined["delta_mean_PER"] > 0).sum()),
                "improved_sessions": int((joined["delta_mean_PER"] < 0).sum()),
                "num_sessions": int(len(joined)),
            }
        )

    source_labels = label_sources([row["source_session"] for row in summary_rows])
    summary = pd.DataFrame(summary_rows).sort_values("source_session").reset_index(drop=True)
    summary["source_label"] = summary["source_session"].map(source_labels)
    summary = summary[
        [
            "source_label",
            "source_session",
            "weighted_PER",
            "delta_vs_native",
            "trial_mean_PER",
            "harmed_sessions",
            "improved_sessions",
            "num_sessions",
        ]
    ]

    joined_all = pd.concat(joined_frames, ignore_index=True)
    joined_all["source_label"] = joined_all["source_session"].map(source_labels)
    joined_all = joined_all.sort_values(["source_session", "target_date"]).reset_index(drop=True)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_joined.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    joined_all.to_csv(args.output_joined, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plot_summary = pd.concat(
        [
            pd.DataFrame([{"label": "native-day", "weighted_PER": native_weighted}]),
            summary.assign(label=summary["source_label"] + "\\n" + summary["source_session"].str.replace("t15.", "", regex=False))[
                ["label", "weighted_PER"]
            ],
        ],
        ignore_index=True,
    )
    ax.bar(plot_summary["label"], plot_summary["weighted_PER"], color=["#2a9d8f", "#457b9d", "#7b2cbf", "#b23a48"])
    ax.set_ylabel("Phoneme-weighted PER")
    ax.set_title("T15 cross-day source sweep")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.figures_dir / "t15_source_sweep_weighted_per.png", dpi=200)
    plt.close(fig)

    heatmap = joined_all.pivot(index="source_label", columns="session", values="delta_mean_PER")
    source_order = summary["source_label"].tolist()
    target_order = sorted(joined_all["session"].unique(), key=session_date)
    heatmap = heatmap.loc[source_order, target_order]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    image = ax.imshow(heatmap.values, aspect="auto", cmap="magma")
    tick_step = max(1, len(target_order) // 12)
    ticks = np.arange(0, len(target_order), tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([target_order[i].replace("t15.", "") for i in ticks], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(source_order)))
    ax.set_yticklabels(source_order)
    ax.set_title("Cross-day PER increase by source and target session")
    fig.colorbar(image, ax=ax, label="Delta mean PER vs native")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "t15_source_sweep_delta_heatmap.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for label, frame in joined_all.groupby("source_label", sort=False):
        ax.scatter(frame["abs_days_from_source"], frame["cross_mean_PER"], alpha=0.78, label=label, s=36)
    ax.axhline(native_weighted, color="0.25", linewidth=1.2, linestyle="--", label="native weighted PER")
    ax.set_xlabel("Absolute days between source and target")
    ax.set_ylabel("Cross-day mean trial PER")
    ax.set_title("Temporal distance and cross-day decoder degradation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.figures_dir / "t15_source_sweep_time_distance_vs_per.png", dpi=200)
    plt.close(fig)

    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_joined}")
    print(f"Wrote {args.figures_dir / 't15_source_sweep_weighted_per.png'}")
    print(f"Wrote {args.figures_dir / 't15_source_sweep_delta_heatmap.png'}")
    print(f"Wrote {args.figures_dir / 't15_source_sweep_time_distance_vs_per.png'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
