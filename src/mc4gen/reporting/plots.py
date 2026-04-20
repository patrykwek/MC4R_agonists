"""Matplotlib plots used by analyses and the Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_fig(fig: "plt.Figure", path: Path, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def reward_curve(values: NDArray[np.float32], path: Path, *, title: str = "Reward") -> Path:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(values, linewidth=1)
    ax.set_xlabel("RL step")
    ax.set_ylabel(title)
    ax.grid(alpha=0.3)
    return save_fig(fig, path)


def score_distribution(scores: Mapping[str, NDArray[np.float32]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 3))
    for name, arr in scores.items():
        ax.hist(arr, bins=50, alpha=0.4, label=name)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
    return save_fig(fig, path)


def chemical_space_scatter(
    coords: NDArray[np.float32],
    labels: Sequence[str],
    path: Path,
    *,
    title: str = "Chemical space",
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    seen: dict[str, int] = {}
    palette = plt.get_cmap("tab10")
    for i, lab in enumerate(labels):
        idx = seen.setdefault(lab, len(seen))
        ax.scatter(coords[i, 0], coords[i, 1], s=6, color=palette(idx % 10), alpha=0.6)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=palette(i % 10), label=name)
        for name, i in seen.items()
    ]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    return save_fig(fig, path)


def parity_plot(
    predicted: NDArray[np.float32],
    observed: NDArray[np.float32],
    path: Path,
    *,
    title: str = "Parity",
) -> Path:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(observed, predicted, s=10, alpha=0.6)
    lo = float(min(observed.min(), predicted.min()))
    hi = float(max(observed.max(), predicted.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Observed pKi")
    ax.set_ylabel("Predicted pKi")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return save_fig(fig, path)


def funnel(counts: Mapping[str, int], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = list(counts.keys())
    values = list(counts.values())
    ax.barh(labels, values, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Surviving molecules")
    return save_fig(fig, path)


def heatmap(matrix: NDArray[np.float32], row_labels: Iterable[str], col_labels: Iterable[str], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    img = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(list(col_labels), rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(list(row_labels))
    fig.colorbar(img, ax=ax, shrink=0.8)
    return save_fig(fig, path)
