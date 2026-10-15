"""
Shared chart style and the two charts we use most (cross-correlation bars and
diffusion curves).  One style for every figure = the project looks professional.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from . import config

# Colour-blind-safe palette; Nasdaq family = blue, S&P family = orange.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"NQ": BLUE, "QQQ": BLUE, "ES": ORANGE, "SPY": ORANGE,
          "LEADER": BLUE, "FOLLOWER": ORANGE}


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.dpi": 110, "savefig.dpi": 160,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "lines.linewidth": 1.6, "legend.frameon": False,
    })


def save(fig, name: str) -> None:
    path = config.FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved figure -> figures/{name}")


def significance_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """|t| needed for significance when we test `n_tests` lags at once.

    Testing 10 lags at the 5% level gives ~40% chance of at least one false
    positive. The Bonferroni correction (alpha / n_tests) fixes that."""
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / 2 / max(n_tests, 1)))


def plot_ccf(ax, ccf: pd.DataFrame, x: str, y: str, title: str | None = None,
             ylim: float | None = None) -> None:
    """Bars = corr(x_t, y_{t+k}). Same-bar (k=0) is grey and clipped so the
    small lagged bars - the interesting part - stay visible."""
    lags = ccf.index.to_numpy()
    corr = ccf["corr"].to_numpy()
    other = lags != 0
    z = significance_threshold(int(other.sum()))
    sig = ccf["t_robust"].abs().to_numpy() > z
    if ylim is None:
        ylim = max(0.02, abs(corr[other]).max() * 1.6) if other.any() else 0.05
    colors = [GRID if k == 0 else (BLUE if s else "#b7d3f6") for k, s in zip(lags, sig)]
    heights = [c if k != 0 else min(c, ylim * 0.98) for k, c in zip(lags, corr)]
    ax.bar(lags, heights, color=colors, width=0.7, edgecolor=SURFACE, linewidth=1)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_ylim(-ylim, ylim)
    ax.text(0.01, 0.97, f"same-bar corr = {ccf.loc[0, 'corr']:.2f} (grey bar, cut off)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK_2)
    ax.text(0.99, 0.02, f"dark = significant (robust |t| > {z:.2f}, Bonferroni)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_xlabel(f"lag k in bars   (k > 0: {x} first,  k < 0: {y} first)")
    ax.set_ylabel(f"corr({x}$_t$, {y}$_{{t+k}}$)")
    ax.set_title(title or f"{x} vs {y}: who moves first?")


def plot_diffusion(ax, curves: dict[str, pd.DataFrame], title: str,
                   ylim: tuple[float, float] | None = None) -> None:
    """Share of the long-run reaction absorbed h bars after a shock.
    Pass the same `ylim` to every panel of a figure so they compare fairly."""
    palette = [BLUE, ORANGE, AQUA, YELLOW]
    for (label, d), color in zip(curves.items(), palette):
        ax.plot(d.index, d["share"] * 100, marker="o", markersize=4, color=color,
                label=f"{label}: {d['share'].iloc[0]:.1%} in same bar")
    ax.axhline(100, color=INK_2, linewidth=0.8, linestyle="--")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("bars after the shock (h)")
    ax.set_ylabel("% of total reaction absorbed")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
