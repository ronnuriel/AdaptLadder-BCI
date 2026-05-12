from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drift_metrics import compute_session_stats, drift_metric_table, pairwise_mean_shift, session_mean_pca
from src.plotting import plot_drift_over_time, plot_mean_shift_heatmap, plot_pca_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute T15 session drift metrics and figures.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--split", action="append", default=["train"], help="HDF5 split to include; repeatable.")
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_session_stats(args.data_dir, splits=tuple(args.split))
    metrics = drift_metric_table(stats)
    metrics_path = args.tables_dir / "t15_session_drift_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    pca_df = session_mean_pca(stats)
    pca_df.to_csv(args.tables_dir / "t15_session_mean_pca.csv", index=False)

    sessions, mean_shift = pairwise_mean_shift(stats)
    plot_drift_over_time(metrics, args.figures_dir / "t15_drift_over_time.png")
    plot_pca_sessions(pca_df, args.figures_dir / "t15_pca_sessions.png")
    plot_mean_shift_heatmap(sessions, mean_shift, args.figures_dir / "t15_mean_shift_heatmap.png")

    print(f"Wrote drift metrics for {len(metrics)} sessions to {metrics_path}")
    print(metrics[["session", "days_from_source", "mean_shift_from_source", "scale_shift_from_source"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
