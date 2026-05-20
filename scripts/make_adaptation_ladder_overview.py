#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse


OUT_DIR = Path("paper_full/figures")


def box(ax, xy, wh, title, body, fc="#ffffff", ec="#1f4e79", title_color="#08306b"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(x + 0.04 * w, y + h - 0.12 * h, title, fontsize=12, weight="bold", color=title_color, va="top")
    ax.text(x + 0.04 * w, y + h - 0.31 * h, body, fontsize=10.0, color="#1a1a1a", va="top", linespacing=1.16)
    return patch


def arrow(ax, start, end, color="#2b6cb0", lw=2.0, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    return patch


def draw_signal(ax, x0, y0, w, h):
    rng = np.random.default_rng(4)
    x = np.linspace(0, 1, 260)
    y = 0.5 + 0.13 * np.sin(17 * np.pi * x) + 0.07 * rng.normal(size=len(x))
    ax.plot(x0 + w * x, y0 + h * y, color="#111111", lw=1.1)
    ax.plot([x0, x0 + w], [y0 + 0.18 * h, y0 + 0.18 * h], color="#555555", lw=0.8)
    ax.plot([x0, x0], [y0 + 0.18 * h, y0 + 0.88 * h], color="#555555", lw=0.8)
    ax.text(x0 + 0.48 * w, y0 + 0.02 * h, "time", fontsize=8, ha="center")


def draw_cloud(ax, center, scale=1.0, color="#6a51a3", seed=0, label=None):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(34, 2)) @ np.array([[0.16, 0.06], [0.02, 0.09]]) * scale
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    ax.scatter(pts[:, 0], pts[:, 1], s=17 * scale, color=color, alpha=0.55, edgecolor="none")
    ax.scatter([center[0]], [center[1]], marker="*", s=210 * scale, color=color, edgecolor="#37135a", linewidth=0.7, zorder=4)
    ax.add_patch(Ellipse(center, 0.46 * scale, 0.24 * scale, angle=18, fill=False, lw=1.2, ls="--", ec=color, alpha=0.75))
    ax.plot([center[0] - 0.2 * scale, center[0] + 0.2 * scale], [center[1] - 0.06 * scale, center[1] + 0.06 * scale], color=color, lw=1.4)
    if label:
        ax.text(center[0], center[1] - 0.22 * scale, label, fontsize=8.5, ha="center", color="#222222")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.6, 6.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Recency-aware input-state selection for cross-session BCI drift", ha="center", va="top", fontsize=20, weight="bold", color="#08306b")
    ax.text(0.5, 0.925, "Keep the GRU decoder frozen; choose or lightly adapt only the input state.", ha="center", va="top", fontsize=12.5, color="#333333")

    box(
        ax,
        (0.035, 0.62),
        (0.19, 0.22),
        "1. New target day",
        "Use the first K trials\nas a short beginning-\nof-day window.",
        fc="#eef6ff",
        ec="#3182bd",
    )
    draw_signal(ax, 0.092, 0.625, 0.12, 0.10)

    box(
        ax,
        (0.285, 0.62),
        (0.20, 0.22),
        "2. Estimate geometry",
        "Feature cloud:\ncovariance and\nsubspace shape.",
        fc="#f7f2ff",
        ec="#756bb1",
        title_color="#4a1486",
    )
    draw_cloud(ax, (0.415, 0.675), 0.22, "#6a51a3", seed=1)

    box(
        ax,
        (0.545, 0.62),
        (0.25, 0.22),
        "3. Compare to past states",
        "Input-state library:\none stored layer\nper past session.",
        fc="#f8fff5",
        ec="#31a354",
        title_color="#006d2c",
    )
    for i, (cx, cy, col) in enumerate(
        [
            (0.645, 0.668, "#3182bd"),
            (0.715, 0.720, "#31a354"),
            (0.765, 0.662, "#fd8d3c"),
        ]
    ):
        draw_cloud(ax, (cx, cy), 0.20, col, seed=10 + i)

    box(
        ax,
        (0.835, 0.62),
        (0.13, 0.22),
        "Input layer",
        "Small session-\nspecific mapping\nbefore the shared\nGRU.",
        fc="#fff7ec",
        ec="#e6550d",
        title_color="#a63603",
    )

    arrow(ax, (0.225, 0.73), (0.285, 0.73))
    arrow(ax, (0.485, 0.73), (0.545, 0.73))
    arrow(ax, (0.795, 0.73), (0.835, 0.73))

    ax.text(0.5, 0.515, "Adaptation ladder: choose the lightest action supported by the evidence", ha="center", va="center", fontsize=15, weight="bold", color="#08306b")

    action_specs = [
        ((0.07, 0.28), (0.23, 0.15), "A. Reuse previous", "If the last state is\nrecent and plausible.", "#238b45", "#edf8e9"),
        ((0.38, 0.28), (0.23, 0.15), "B. Retrieve older state", "If geometry finds a\ncloser stored state.", "#d95f0e", "#fff5eb"),
        ((0.69, 0.28), (0.23, 0.15), "C. Residual calibration", "If stored states are\nnot reliable; update\ninput only.", "#cb181d", "#fff5f0"),
    ]
    for (xy, wh, title, body, ec, fc) in action_specs:
        box(ax, xy, wh, title, body, fc=fc, ec=ec, title_color=ec)
        arrow(ax, (xy[0] + wh[0] / 2, 0.51), (xy[0] + wh[0] / 2, xy[1] + wh[1]), color=ec, lw=1.7)

    box(
        ax,
        (0.23, 0.065),
        (0.54, 0.105),
        "Shared GRU decoder remains frozen",
        "Adaptation acts on input-state selection or a small input transform,\nnot on the recurrent decoder backbone.",
        fc="#eef6ff",
        ec="#3182bd",
    )
    for x in [0.185, 0.495, 0.805]:
        arrow(ax, (x, 0.28), (0.5, 0.17), color="#4d4d4d", lw=1.4, rad=0.08 if x < 0.5 else -0.08)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT_DIR / "adaptation_ladder_overview.png", dpi=240)
    fig.savefig(OUT_DIR / "adaptation_ladder_overview.svg")
    print(f"Wrote {OUT_DIR / 'adaptation_ladder_overview.png'}")
    print(f"Wrote {OUT_DIR / 'adaptation_ladder_overview.svg'}")


if __name__ == "__main__":
    main()
