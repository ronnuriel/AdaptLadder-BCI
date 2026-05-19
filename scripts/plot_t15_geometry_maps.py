from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METRICS = {
    "cov_relative_fro_shift_from_source": "Relative covariance shift",
    "mean_principal_angle_deg": "Mean principal angle (deg)",
    "subspace_chordal_distance": "Subspace chordal distance",
    "mean_shift_from_source": "Mean shift",
}


def session_label(session: str) -> str:
    return session.replace("t15.", "")


def ordered_sessions(pairwise: pd.DataFrame) -> list[str]:
    dates = (
        pairwise[["target_session", "target_date"]]
        .rename(columns={"target_session": "session", "target_date": "date"})
        .drop_duplicates()
    )
    return dates.sort_values(["date", "session"])["session"].tolist()


def build_symmetric_matrix(pairwise: pd.DataFrame, metric: str, sessions: list[str]) -> pd.DataFrame:
    forward = pairwise.pivot_table(
        index="target_session",
        columns="source_session",
        values=metric,
        aggfunc="mean",
    )
    matrix = forward.reindex(index=sessions, columns=sessions)

    # The relative covariance distance depends slightly on the denominator
    # source covariance norm. Average both directions to make the visual map
    # read as an undirected "how far apart are these days?" table.
    symmetric = matrix.copy()
    values = matrix.to_numpy(dtype=float)
    stacked = np.stack([values, values.T])
    valid = np.isfinite(stacked)
    sums = np.nansum(stacked, axis=0)
    counts = valid.sum(axis=0)
    sym_values = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    np.fill_diagonal(sym_values, 0.0)
    symmetric.loc[:, :] = sym_values
    return symmetric


def plot_heatmap(matrix: pd.DataFrame, metric: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [session_label(session) for session in matrix.index]

    fig, ax = plt.subplots(figsize=(10.8, 9.2))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="magma", interpolation="nearest")
    ax.set_title(f"T15 cross-day geometry heatmap\n{METRICS[metric]}")
    ax.set_xlabel("Source session")
    ax.set_ylabel("Target session")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(METRICS[metric])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def prepare_scatter_table(pairwise: pd.DataFrame, metric: str) -> pd.DataFrame:
    scatter = pairwise.copy()
    scatter["metric"] = metric
    scatter["metric_value"] = scatter[metric].astype(float)
    scatter["target_label"] = scatter["target_session"].map(session_label)
    scatter["source_label"] = scatter["source_session"].map(session_label)
    return scatter[
        [
            "target_session",
            "source_session",
            "target_date",
            "source_date",
            "days_from_source",
            "abs_days_from_source",
            "metric",
            "metric_value",
            "target_label",
            "source_label",
        ]
    ]


def plot_scatter(scatter: pd.DataFrame, metric: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = scatter["abs_days_from_source"].to_numpy(dtype=float)
    y = scatter["metric_value"].to_numpy(dtype=float)
    rho, p_value = spearmanr(x, y)
    p_label = "<1e-300" if p_value == 0 else f"{p_value:.1e}"

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(x, y, s=18, alpha=0.45, color="#2f6f9f", edgecolors="none")

    if len(np.unique(x)) > 1:
        coeff = np.polyfit(x, y, deg=1)
        line_x = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        ax.plot(line_x, coeff[0] * line_x + coeff[1], color="#b23a48", linewidth=2)

    ax.set_title(f"T15 geometry drift vs time\nSpearman rho={rho:.2f}, p={p_label}")
    ax.set_xlabel("Days between sessions")
    ax.set_ylabel(METRICS[metric])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def summarize_metric(pairwise: pd.DataFrame, metric: str) -> dict[str, float | str | int]:
    scatter = prepare_scatter_table(pairwise, metric)
    rho, p_value = spearmanr(scatter["abs_days_from_source"], scatter["metric_value"])
    return {
        "metric": metric,
        "label": METRICS[metric],
        "num_pairs": int(len(scatter)),
        "spearman_rho_abs_days": float(rho),
        "spearman_p_value": float(p_value),
        "median_metric": float(scatter["metric_value"].median()),
        "min_metric": float(scatter["metric_value"].min()),
        "max_metric": float(scatter["metric_value"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot T15 cross-day geometry heatmaps and drift scatter plots.")
    parser.add_argument(
        "--pairwise",
        type=Path,
        default=Path("results/tables/t15_geometry_source_pairwise_distances.csv"),
        help="Pairwise source-target geometry table.",
    )
    parser.add_argument("--metrics", nargs="+", default=["cov_relative_fro_shift_from_source", "mean_principal_angle_deg"])
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    pairwise = pd.read_csv(args.pairwise)
    pairwise = pairwise[pairwise["target_session"] != pairwise["source_session"]].copy()
    sessions = ordered_sessions(pairwise)

    summary_rows = []
    for metric in args.metrics:
        if metric not in pairwise.columns:
            raise ValueError(f"Metric {metric!r} not found in {args.pairwise}")
        if metric not in METRICS:
            METRICS[metric] = metric

        matrix = build_symmetric_matrix(pairwise, metric, sessions)
        matrix_path = args.tables_dir / f"t15_geometry_heatmap_matrix_{metric}.csv"
        matrix.to_csv(matrix_path)

        scatter = prepare_scatter_table(pairwise, metric)
        scatter_path = args.tables_dir / f"t15_geometry_scatter_{metric}.csv"
        scatter.to_csv(scatter_path, index=False)

        heatmap_path = args.figures_dir / f"t15_geometry_heatmap_{metric}.png"
        scatter_fig_path = args.figures_dir / f"t15_geometry_scatter_{metric}.png"
        plot_heatmap(matrix, metric, heatmap_path)
        plot_scatter(scatter, metric, scatter_fig_path)
        summary_rows.append(summarize_metric(pairwise, metric))

        print(f"Wrote {matrix_path}")
        print(f"Wrote {scatter_path}")
        print(f"Wrote {heatmap_path}")
        print(f"Wrote {scatter_fig_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = args.tables_dir / "t15_geometry_map_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
