from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drift_metrics import summarize_t15_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the T15 dataset sanity summary table.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/t15_dataset_summary.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_t15_dataset(args.data_dir)
    summary.to_csv(args.output, index=False)

    print(f"Wrote {len(summary)} sessions to {args.output}")
    print(summary.head().to_string(index=False))


if __name__ == "__main__":
    main()
