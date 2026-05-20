#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_METRIC = "cov_relative_fro_shift_from_source"


def short_date(session: str) -> str:
    return session.replace("t15.", "").replace("2023-", "").replace("2024-", "").replace("2025-", "")


def winner_colors(values: pd.Series) -> list[str]:
    palette = {
        "geometry": "#2ca25f",
        "previous": "#de2d26",
        "tie": "#737373",
        "unknown": "#969696",
    }
    return [palette.get(str(v), "#969696") for v in values]


def winner_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca25f", label="geometry lower PER", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#de2d26", label="previous lower PER", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#737373", label="tie / same source", markersize=8),
    ]


def classic_mds(distance: np.ndarray) -> np.ndarray:
    n = distance.shape[0]
    d2 = distance**2
    center = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * center @ d2 @ center
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = np.maximum(eigvals[:2], 0)
    return eigvecs[:, :2] * np.sqrt(keep)


def build_distance_matrix(pairwise: pd.DataFrame, metric: str) -> tuple[list[str], np.ndarray, pd.Series]:
    sessions = sorted(set(pairwise["target_session"]).union(pairwise["source_session"]))
    index = {session: i for i, session in enumerate(sessions)}
    distance = np.full((len(sessions), len(sessions)), np.nan, dtype=float)
    np.fill_diagonal(distance, 0.0)
    for _, row in pairwise.iterrows():
        i = index[row["target_session"]]
        j = index[row["source_session"]]
        distance[i, j] = float(row[metric])
    # Symmetrize by averaging available directions; some files omit self-pairs.
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            vals = [v for v in [distance[i, j], distance[j, i]] if not np.isnan(v)]
            if vals:
                value = float(np.mean(vals))
                distance[i, j] = value
                distance[j, i] = value
    finite = distance[np.isfinite(distance)]
    fill_value = float(np.nanmax(finite)) if len(finite) else 1.0
    distance = np.nan_to_num(distance, nan=fill_value)
    dates = pd.Series(pd.to_datetime([s.replace("t15.", "") for s in sessions]), index=sessions)
    return sessions, distance, dates


def plot_lag_by_target(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    dates = pd.to_datetime(frame["target_date"])
    colors = winner_colors(frame["winner"])
    markers = np.where(frame["same_as_previous"], "o", "^")
    for marker in sorted(set(markers)):
        mask = markers == marker
        label = "selected previous" if marker == "o" else "selected older state"
        ax.scatter(
            dates[mask],
            frame.loc[mask, "geometry_lag_days"],
            c=np.array(colors, dtype=object)[mask],
            marker=marker,
            s=70,
            edgecolor="white",
            linewidth=0.7,
            label=label,
            alpha=0.92,
        )
    ax.set_title("Geometry-selected source lag by target session")
    ax.set_xlabel("Target session")
    ax.set_ylabel("Geometry source lag (days)")
    ax.grid(True, axis="y", alpha=0.25)
    marker_legend = ax.legend(frameon=False, loc="upper left")
    ax.add_artist(marker_legend)
    ax.legend(handles=winner_legend_handles(), frameon=False, loc="upper right")
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_distance_scatter(frame: pd.DataFrame, metric: str, output: Path) -> None:
    x_col = f"previous_{metric}"
    y_col = f"geometry_{metric}"
    plot = frame.dropna(subset=[x_col, y_col]).copy()
    fig, ax = plt.subplots(figsize=(5.6, 5.1))
    ax.scatter(
        plot[x_col],
        plot[y_col],
        c=winner_colors(plot["winner"]),
        s=np.where(plot["same_as_previous"], 42, 86),
        marker="o",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
    )
    lo = float(min(plot[x_col].min(), plot[y_col].min()))
    hi = float(max(plot[x_col].max(), plot[y_col].max()))
    pad = 0.04 * (hi - lo) if hi > lo else 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#525252", linestyle="--", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title("Previous distance vs. geometry-selected distance")
    ax.set_xlabel("Distance to previous source")
    ax.set_ylabel("Distance to geometry-selected source")
    ax.text(
        0.02,
        0.98,
        "Below diagonal: geometry is closer",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.grid(True, alpha=0.22)
    ax.legend(handles=winner_legend_handles(), frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_state_map(frame: pd.DataFrame, pairwise: pd.DataFrame, metric: str, output: Path) -> None:
    sessions, distance, dates = build_distance_matrix(pairwise, metric)
    coords = classic_mds(distance)
    coord = pd.DataFrame(coords, columns=["x", "y"], index=sessions)
    date_nums = dates.map(pd.Timestamp.toordinal).to_numpy()

    fig, ax = plt.subplots(figsize=(8.2, 5.9))
    scatter = ax.scatter(
        coord["x"],
        coord["y"],
        c=date_nums,
        cmap="viridis",
        s=68,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    tick_ordinals = np.linspace(date_nums.min(), date_nums.max(), 5)
    tick_dates = [pd.Timestamp.fromordinal(int(round(v))) for v in tick_ordinals]
    cbar.set_ticks(tick_ordinals)
    cbar.set_ticklabels([d.strftime("%Y-%m") for d in tick_dates])
    cbar.set_label("Session date")

    nonprev = frame[~frame["same_as_previous"]].copy()
    for _, row in nonprev.iterrows():
        target = row["target_session"]
        previous = row["previous_session"]
        geometry = row["geometry_source_session"]
        if target not in coord.index or previous not in coord.index or geometry not in coord.index:
            continue
        tx, ty = coord.loc[target, ["x", "y"]]
        px, py = coord.loc[previous, ["x", "y"]]
        gx, gy = coord.loc[geometry, ["x", "y"]]
        ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(px, py),
            arrowprops=dict(arrowstyle="->", color="#252525", lw=0.8, alpha=0.35),
            zorder=2,
        )
        ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(gx, gy),
            arrowprops=dict(arrowstyle="->", color="#de2d26", lw=1.2, alpha=0.72),
            zorder=2,
        )
        ax.text(tx, ty, short_date(target)[5:], fontsize=7, ha="left", va="bottom", color="#111111")

    ax.set_title("T15 geometry state map (MDS)")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.grid(True, alpha=0.18)
    ax.text(
        0.01,
        0.01,
        "Black arrows: previous source → target; red arrows: older geometry source → target",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the T15 geometry-vs-previous source-selection story.")
    parser.add_argument("--detail", type=Path, default=Path("results/tables/t15_geometry_vs_previous_sources.csv"))
    parser.add_argument("--pairwise", type=Path, default=Path("results/tables/t15_geometry_source_pairwise_distances.csv"))
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--calibration-trials", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--output-nonprevious",
        type=Path,
        default=Path("results/tables/t15_geometry_non_previous_cases.csv"),
    )
    args = parser.parse_args()

    detail = pd.read_csv(args.detail)
    pairwise = pd.read_csv(args.pairwise)
    frame = detail[detail["calibration_trials"] == args.calibration_trials].copy()
    if frame.empty:
        raise ValueError(f"No rows found for K={args.calibration_trials} in {args.detail}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_nonprevious.parent.mkdir(parents=True, exist_ok=True)
    frame[~frame["same_as_previous"]].to_csv(args.output_nonprevious, index=False)

    lag_path = args.output_dir / "t15_geometry_selected_lag_by_target.png"
    scatter_path = args.output_dir / "t15_geometry_previous_vs_selected_distance.png"
    map_path = args.output_dir / "t15_geometry_state_map_mds.png"
    plot_lag_by_target(frame, lag_path)
    plot_distance_scatter(frame, args.metric, scatter_path)
    plot_state_map(frame, pairwise, args.metric, map_path)

    print(f"Wrote {args.output_nonprevious}")
    print(f"Wrote {lag_path}")
    print(f"Wrote {scatter_path}")
    print(f"Wrote {map_path}")
    summary = frame.groupby(["same_as_previous", "winner"]).size().reset_index(name="count")
    print("\nK =", args.calibration_trials)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
