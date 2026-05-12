from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_drift_over_time(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(metrics["days_from_source"], metrics["mean_shift_from_source"], marker="o", label="Mean shift")
    ax.plot(metrics["days_from_source"], metrics["scale_shift_from_source"], marker="s", label="Scale shift")
    ax.set_xlabel("Days from first session")
    ax.set_ylabel("L2 distance from first session")
    ax.set_title("T15 cross-session drift over time")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pca_sessions(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    points = ax.scatter(pca_df["pc1"], pca_df["pc2"], c=np.arange(len(pca_df)), cmap="viridis", s=42)
    ax.plot(pca_df["pc1"], pca_df["pc2"], color="0.75", linewidth=1)
    ax.set_xlabel("PC1 of session mean")
    ax.set_ylabel("PC2 of session mean")
    ax.set_title("T15 session means in PCA space")
    cbar = fig.colorbar(points, ax=ax)
    cbar.set_label("Session order")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mean_shift_heatmap(sessions: list[str], matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    tick_step = max(1, len(sessions) // 12)
    ticks = np.arange(0, len(sessions), tick_step)
    labels = [sessions[i].replace("t15.", "") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pairwise T15 session mean shift")
    fig.colorbar(image, ax=ax, label="L2 distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
