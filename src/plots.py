"""Figures.

One prediction figure per configuration, plus a training-diagnostics panel.
Deliberately plain: a recessive grid, one y-axis, thin marks, and the
uncertainty band drawn behind everything else so it never hides the data.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots, fixed order — never cycled, never reassigned by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"


def _frame(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def prediction(
    path: str,
    xc: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    truth_xc: Optional[np.ndarray] = None,
    truth: Optional[np.ndarray] = None,
    title: str = "",
    ylabel: str = "coefficient",
) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    _frame(ax)

    ax.fill_between(
        xc, mean - 2 * sigma, mean + 2 * sigma,
        color=SERIES[0], alpha=0.16, linewidth=0, zorder=1,
        label="GP posterior ±2σ",
    )
    ax.plot(xc, mean, color=SERIES[0], linewidth=2.0, zorder=3, label="GP mean")

    if truth is not None and truth_xc is not None:
        ax.plot(
            truth_xc, truth, linestyle="none", marker="o", markersize=4.5,
            markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.3,
            zorder=4, label="withheld measurement",
        )

    ax.set_xlabel("x/c", color=INK_SOFT, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in leg.get_texts():
        text.set_color(INK_SOFT)

    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def diagnostics(path: str, history: Sequence[dict], feature_names: Sequence[str]) -> str:
    """Marginal-likelihood loss and the ARD lengthscales it learned."""
    iters = [h["iter"] for h in history]
    loss = [h["loss"] for h in history]
    ls = np.array([h["lengthscales"] for h in history])  # (steps, dims)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=150)

    ax = axes[0]
    _frame(ax)
    ax.plot(iters, loss, color=SERIES[0], linewidth=2.0)
    ax.set_xlabel("iteration", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("negative marginal log-likelihood", color=INK_SOFT, fontsize=10)
    ax.set_title("Training loss", color=INK, fontsize=11, loc="left", pad=10)

    ax = axes[1]
    _frame(ax)
    labels = ["x/c", *feature_names]
    for j, label in enumerate(labels):
        colour = SERIES[j % len(SERIES)]
        ax.plot(iters, ls[:, j], color=colour, linewidth=2.0, label=label)
        # direct label at the line end — the relief rule for low-contrast slots
        ax.annotate(
            label, xy=(iters[-1], ls[-1, j]), xytext=(4, 0),
            textcoords="offset points", color=colour, fontsize=9,
            va="center", fontweight="medium",
        )
    ax.set_xlabel("iteration", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("ARD lengthscale", color=INK_SOFT, fontsize=10)
    ax.set_title("Learned relevance per feature", color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlim(iters[0], iters[-1] * 1.12)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in leg.get_texts():
        text.set_color(INK_SOFT)

    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def calibration(path: str, records: Sequence[dict]) -> str:
    """Observed ±2σ coverage per withheld configuration against the 0.95 nominal.

    Two bars per configuration: the latent-function band, and the predictive band
    that also carries the measurement error. Only the second is expected to reach
    the nominal value.
    """
    labels = [r["label"] for r in records]
    latent = [r["coverage_latent"] for r in records]
    predictive = [r["coverage_predictive"] for r in records]
    y = np.arange(len(labels))
    h = 0.34

    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(labels) + 2.0), dpi=150)
    _frame(ax)
    ax.barh(y + h / 2 + 0.01, predictive, height=h, color=SERIES[0], zorder=2,
            label="predictive (with measurement error)")
    ax.barh(y - h / 2 - 0.01, latent, height=h, color=SERIES[1], zorder=2,
            label="latent function only")
    ax.axvline(0.95, color=INK, linewidth=1.4, linestyle="--", zorder=3)
    ax.annotate("nominal 0.95", xy=(0.95, len(labels) - 0.35), xytext=(4, 0),
                textcoords="offset points", color=INK, fontsize=9, va="center")
    leg = ax.legend(
        frameon=False, fontsize=9, loc="lower left",
        bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0.0,
    )
    for text in leg.get_texts():
        text.set_color(INK_SOFT)
    ax.set_yticks(y, labels, fontsize=8.5, color=INK_SOFT)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("fraction of withheld points inside ±2σ", color=INK_SOFT, fontsize=10)
    ax.set_title("Calibration on withheld configurations", color=INK, fontsize=11,
                 loc="left", pad=34)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
