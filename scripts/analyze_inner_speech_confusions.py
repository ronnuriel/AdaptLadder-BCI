from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_named_csv(spec: str) -> tuple[str, pd.DataFrame]:
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        path = spec
        name = Path(path).stem
    return name, pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize inner-speech nearest-behavior confusions.")
    parser.add_argument(
        "--nearest",
        nargs="+",
        default=[
            "binnedTX_go=results/tables/inner_interleaved_binnedTX_go_nearest.csv",
            "spikePow_go=results/tables/inner_interleaved_spikePow_go_nearest.csv",
            "binnedTX_delay=results/tables/inner_interleaved_binnedTX_delay_nearest.csv",
        ],
        help="Entries of the form name=path/to/nearest.csv.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/tables/_explore_inner_speech_confusion_summary.csv"),
    )
    parser.add_argument(
        "--output-confusions",
        type=Path,
        default=Path("results/tables/_explore_inner_speech_confusions.csv"),
    )
    args = parser.parse_args()

    summary_rows = []
    confusion_tables = []
    for spec in args.nearest:
        name, frame = read_named_csv(spec)
        required = {"target_behavior", "source_behavior", "nearest_correct_behavior"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

        total = int(len(frame))
        correct = int(frame["nearest_correct_behavior"].astype(bool).sum())
        chance = 1.0 / frame["target_behavior"].nunique()
        summary_rows.append(
            {
                "setting": name,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else float("nan"),
                "chance": chance,
                "num_behaviors": int(frame["target_behavior"].nunique()),
            }
        )

        confusion = pd.crosstab(frame["target_behavior"], frame["source_behavior"])
        confusion = confusion.reset_index().melt(
            id_vars="target_behavior",
            var_name="retrieved_behavior",
            value_name="count",
        )
        confusion.insert(0, "setting", name)
        confusion_tables.append(confusion)

    summary = pd.DataFrame(summary_rows)
    confusions = pd.concat(confusion_tables, ignore_index=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    confusions.to_csv(args.output_confusions, index=False)

    print("Inner-speech nearest-behavior summary:")
    print(summary.to_string(index=False))
    print("\nNonzero off-diagonal confusions:")
    off_diag = confusions[
        (confusions["target_behavior"] != confusions["retrieved_behavior"]) & (confusions["count"] > 0)
    ]
    print(off_diag.to_string(index=False) if not off_diag.empty else "none")


if __name__ == "__main__":
    main()
