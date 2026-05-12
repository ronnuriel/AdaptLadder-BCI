from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the T15 adaptation ladder on the cross-day decoder stress test.")
    parser.add_argument("--model-path", type=Path, default=Path("data/external/btt-25-gru-pure-baseline-0-0898"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val", "test"], default="val")
    parser.add_argument("--source-session", required=True)
    parser.add_argument("--adaptations", nargs="+", default=["none", "target_zscore", "moment_match_to_source"])
    parser.add_argument("--gpu-number", type=int, default=-1)
    parser.add_argument("--output-trials", type=Path, default=Path("results/tables/t15_adaptation_trial_results.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/t15_adaptation_session_summary.csv"))
    args = parser.parse_args()

    print("Adaptation ladder CLI scaffold is ready.")
    print(f"Source session: {args.source_session}")
    print(f"Adaptations: {', '.join(args.adaptations)}")
    print("Next step: reuse decoder probe inference and apply input transforms before the source input layer.")


if __name__ == "__main__":
    main()
