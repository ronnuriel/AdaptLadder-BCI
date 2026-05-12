from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t15_utils import session_date


def weighted_per(trials: pd.DataFrame) -> float:
    return float(trials["edit_distance"].sum() / trials["num_phonemes"].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot native-day vs cross-day T15 PER degradation.")
    parser.add_argument("--native-trials", type=Path, default=Path("results/tables/t15_decoder_probe_val.csv"))
    parser.add_argument("--cross-trials", type=Path, default=Path("results/tables/t15_decoder_probe_cross_day_source_2023_08_13_val.csv"))
    parser.add_argument("--native-summary", type=Path, default=Path("results/tables/t15_decoder_probe_session_summary.csv"))
    parser.add_argument(
        "--cross-summary",
        type=Path,
        default=Path("results/tables/t15_decoder_probe_cross_day_source_2023_08_13_session_summary.csv"),
    )
    parser.add_argument("--output-joined", type=Path, default=Path("results/tables/t15_native_vs_cross_day_joined.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/t15_native_vs_cross_day_summary.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    native_trials = pd.read_csv(args.native_trials)
    cross_trials = pd.read_csv(args.cross_trials)
    native = pd.read_csv(args.native_summary)
    cross = pd.read_csv(args.cross_summary)

    source_sessions = sorted(cross["input_layer_session"].dropna().unique())
    if len(source_sessions) != 1:
        raise ValueError(f"Expected one cross-day source session, found: {source_sessions}")
    source_session = source_sessions[0]

    joined = native[
        [
            "session",
            "mode",
            "input_layer_session",
            "n_trials",
            "mean_PER",
            "median_PER",
            "mean_blank_rate",
            "mean_confidence",
            "mean_entropy",
        ]
    ].merge(
        cross[
            [
                "session",
                "mode",
                "input_layer_session",
                "mean_PER",
                "median_PER",
                "mean_blank_rate",
                "mean_confidence",
                "mean_entropy",
            ]
        ],
        on="session",
        suffixes=("_native", "_cross"),
    )
    joined["date"] = joined["session"].map(lambda value: session_date(value).isoformat())
    joined["delta_mean_PER"] = joined["mean_PER_cross"] - joined["mean_PER_native"]
    joined["delta_median_PER"] = joined["median_PER_cross"] - joined["median_PER_native"]
    joined["ratio_mean_PER"] = joined["mean_PER_cross"] / joined["mean_PER_native"]
    joined = joined.sort_values("date").reset_index(drop=True)

    native_weighted = weighted_per(native_trials)
    cross_weighted = weighted_per(cross_trials)
    summary = pd.DataFrame(
        [
            {
                "source_session": source_session,
                "native_weighted_PER": native_weighted,
                "cross_day_weighted_PER": cross_weighted,
                "delta_weighted_PER": cross_weighted - native_weighted,
                "native_trial_mean_PER": float(native_trials["PER"].mean()),
                "cross_day_trial_mean_PER": float(cross_trials["PER"].mean()),
                "sessions_harmed": int((joined["delta_mean_PER"] > 0).sum()),
                "sessions_improved": int((joined["delta_mean_PER"] < 0).sum()),
                "num_sessions": int(len(joined)),
            }
        ]
    )

    args.output_joined.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output_joined, index=False)
    summary.to_csv(args.output_summary, index=False)

    dates = pd.to_datetime(joined["date"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(dates, joined["mean_PER_native"], marker="o", linewidth=1.8, label="Native-day")
    ax.plot(dates, joined["mean_PER_cross"], marker="s", linewidth=1.8, label=f"Cross-day source {source_session}")
    ax.set_xlabel("Session date")
    ax.set_ylabel("Mean trial PER")
    ax.set_title("T15 decoder degradation under cross-day input-layer mismatch")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(args.figures_dir / "t15_native_vs_cross_day_per.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#b23a48" if value > 0 else "#2a9d8f" for value in joined["delta_mean_PER"]]
    ax.bar(dates, joined["delta_mean_PER"], color=colors, width=6)
    ax.axhline(0, color="0.2", linewidth=1)
    ax.set_xlabel("Session date")
    ax.set_ylabel("Cross-day minus native mean PER")
    ax.set_title("Most T15 sessions are harmed by an old source input layer")
    ax.grid(True, axis="y", alpha=0.25)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(args.figures_dir / "t15_cross_day_delta_per_by_session.png", dpi=200)
    plt.close(fig)

    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_joined}")
    print(f"Wrote {args.figures_dir / 't15_native_vs_cross_day_per.png'}")
    print(f"Wrote {args.figures_dir / 't15_cross_day_delta_per_by_session.png'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
