#!/usr/bin/env python3
"""Generate the figures used on the polyband documentation site.

Everything is drawn from synthetic data generated in ``polyband.datasets``,
so this script is fully reproducible and depends on no external file.

Run it from anywhere:

    python docs/make_figures.py

Output goes to ``docs/figures/`` as SVG, which stays sharp at any zoom level
and keeps the page light. The palette matches the site's dark theme.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from polyband import fit_polyband, plot_diagnostics, plot_polyband, select_orders
from polyband.datasets import (
    make_decades,
    make_heavy_tails,
    make_trumpet,
    make_with_errors,
)

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Site palette, from themes_outils/DESIGN_SYSTEM.md
BG = "#0a1322"
TEXT = "#e8eef8"
MUTED = "#a8b4ca"
ACCENT = "#62c2ff"
TRUTH = "#6ce2ad"
WARN = "#ffb066"
LINE = "#2a3b56"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": LINE,
    "axes.titlecolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": LINE,
    "legend.facecolor": "#0e1a2e",
    "legend.edgecolor": LINE,
    "legend.framealpha": 0.9,
    "font.size": 11,
    "axes.titlesize": 11,
    "figure.dpi": 110,
    "svg.fonttype": "none",   # keep text as text, so it stays selectable
})


def save(fig, name):
    fig.tight_layout()
    path = OUT / f"{name}.svg"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parent.parent)}")


def style(ax, xlabel="x", ylabel="y"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ----------------------------------------------------------------------
def fig_hero():
    """The headline picture: trend, nested bands, and the truth on top."""
    data = make_trumpet(n=450, seed=0)
    fit = fit_polyband(data.x, data.y, order_mean=2, order_width=1)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    art = plot_polyband(fit, data.x, data.y, ax=ax, nsigma=(1, 2),
                        color=ACCENT, point_color=MUTED,
                        point_kw=dict(s=22, alpha=0.45))

    grid = fit.grid()
    for sign in (-1, 1):
        ax.plot(grid, data.true_mean(grid) + sign * data.true_sigma(grid),
                color=TRUTH, linestyle="--", linewidth=1.4,
                label="True 1$\\sigma$ width" if sign > 0 else "_nolegend_")

    style(ax)
    ax.legend(handles=art.legend_handles + [ax.lines[-1]], fontsize=9,
              loc="upper left")
    save(fig, "hero")
    return fit


# ----------------------------------------------------------------------
def fig_band_vs_error():
    """The distinction the whole package is built around.

    Left: both quantities drawn on one sample. Right: how each behaves as the
    sample grows, which is the cleanest way to show they are not the same
    thing. The trend error follows the 1/sqrt(N) reference line; the band
    settles on the true width and stays there.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    data = make_trumpet(n=400, seed=4)
    fit = fit_polyband(data.x, data.y, 2, 1)
    grid = fit.grid()
    ax = axes[0]
    ax.scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0,
               label="Data (N = 400)")
    lo, hi = fit.envelope(grid)
    ax.fill_between(grid, lo, hi, color=ACCENT, alpha=0.18, linewidth=0,
                    label="1$\\sigma$ band: spread of the points")
    tlo, thi = fit.trend_band(grid)
    ax.fill_between(grid, tlo, thi, color=WARN, alpha=0.75, linewidth=0,
                    label="1$\\sigma$ error on the trend itself")
    ax.plot(grid, fit.predict(grid), color=ACCENT, linewidth=2.2, label="Trend")
    ax.set_title("Two different bands, one sample", fontsize=10.5)
    style(ax)
    ax.legend(fontsize=8.5, loc="upper left")

    # Right panel: behaviour with sample size, at mid-range x.
    x0 = 5.0
    sizes = np.unique(np.round(np.logspace(np.log10(30), np.log10(20000), 14)
                               ).astype(int))
    widths, errors = [], []
    for n in sizes:
        d = make_trumpet(n=int(n), seed=100 + int(n))
        f = fit_polyband(d.x, d.y, 2, 1)
        widths.append(float(f.scatter(x0)))
        errors.append(float(f.mean_error(x0)))

    ax = axes[1]
    truth = float(data.true_sigma(np.array([x0]))[0])
    ax.plot(sizes, widths, "o-", color=ACCENT, linewidth=2.0, markersize=5,
            label="1$\\sigma$ band half-width")
    ax.axhline(truth, color=TRUTH, linestyle="--", linewidth=1.4,
               label=f"true width at x = {x0:g}")
    ax.plot(sizes, errors, "s-", color=WARN, linewidth=2.0, markersize=4.5,
            label="error on the trend")
    ref = np.asarray(errors, dtype=float)[0] * np.sqrt(sizes[0] / sizes)
    ax.plot(sizes, ref, color=WARN, linestyle=":", linewidth=1.3,
            label=r"$1/\sqrt{N}$ reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Adding data shrinks one and not the other", fontsize=10.5)
    style(ax, xlabel="sample size N", ylabel=f"half-width at x = {x0:g}")
    ax.legend(fontsize=8.5, loc="lower left")
    save(fig, "band_vs_error")


# ----------------------------------------------------------------------
def fig_orders():
    """What the two orders actually control, shown side by side."""
    data = make_trumpet(n=350, seed=0)
    combos = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4), sharex=True, sharey=True)
    for ax, (om, ow) in zip(axes.ravel(), combos):
        fit = fit_polyband(data.x, data.y, order_mean=om, order_width=ow)
        grid = fit.grid()
        ax.scatter(data.x, data.y, s=12, alpha=0.35, color=MUTED, linewidths=0)
        lo, hi = fit.envelope(grid)
        ax.fill_between(grid, lo, hi, color=ACCENT, alpha=0.20, linewidth=0)
        ax.plot(grid, fit.predict(grid), color=ACCENT, linewidth=2.0)
        ax.plot(grid, data.true_mean(grid), color=TRUTH, linestyle="--",
                linewidth=1.2)
        ax.set_title(f"order_mean={om}, order_width={ow}    "
                     f"BIC={fit.bic:.0f}", fontsize=9.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    save(fig, "orders")


# ----------------------------------------------------------------------
def fig_robust():
    """Gaussian against Student-t on a contaminated sample."""
    data = make_heavy_tails(n=600, seed=1, contamination=0.09)
    gauss = fit_polyband(data.x, data.y, 2, 0)
    robust = fit_polyband(data.x, data.y, 2, 0, nu=4)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, fit, name in ((axes[0], gauss, "Gaussian likelihood (default)"),
                          (axes[1], robust, "Student-t likelihood, nu=4")):
        grid = fit.grid()
        ax.scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0)
        lo, hi = fit.envelope(grid)
        ax.fill_between(grid, lo, hi, color=ACCENT, alpha=0.20, linewidth=0,
                        label="fitted 1$\\sigma$ band")
        ax.plot(grid, fit.predict(grid), color=ACCENT, linewidth=2.1)
        for sign in (-1, 1):
            ax.plot(grid, data.true_mean(grid) + sign * data.true_sigma(grid),
                    color=TRUTH, linestyle="--", linewidth=1.4,
                    label="true width of the clean component"
                          if sign > 0 else "_nolegend_")
        width = float(fit.scatter(5.0))
        ax.set_title(f"{name}\nfitted width = {width:.2f} "
                     f"(true 0.60)", fontsize=10)
        style(ax, ylabel="y" if fit is gauss else "")
    axes[0].legend(fontsize=8.5, loc="upper left")
    save(fig, "robust")


# ----------------------------------------------------------------------
def fig_logspace():
    """Why a multiplicative quantity has to be fitted in log space."""
    data = make_decades(n=400, seed=2)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    linear = fit_polyband(data.x, data.y, 2, 1)
    grid = linear.grid()
    axes[0].scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0)
    lo, hi = linear.envelope(grid)
    axes[0].fill_between(grid, lo, hi, color=WARN, alpha=0.20, linewidth=0)
    axes[0].plot(grid, linear.predict(grid), color=WARN, linewidth=2.1)
    axes[0].axhline(0, color=LINE, linewidth=1.0)
    axes[0].set_title("Fitted in y: band runs below zero,\nscatter is skewed",
                      fontsize=10)
    style(axes[0])

    logfit = fit_polyband(data.x, data.y, 2, 1, log_y=True)
    grid = logfit.grid()
    axes[1].scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0)
    lo, hi = logfit.envelope(grid)
    axes[1].fill_between(grid, lo, hi, color=ACCENT, alpha=0.20, linewidth=0)
    axes[1].plot(grid, logfit.predict(grid), color=ACCENT, linewidth=2.1)
    axes[1].set_yscale("log")
    axes[1].set_title("Fitted with log_y=True: band tracks\nthe multiplicative "
                      "spread", fontsize=10)
    style(axes[1])
    save(fig, "logspace")


# ----------------------------------------------------------------------
def fig_yerr():
    """Separating intrinsic scatter from measurement noise."""
    data = make_with_errors(n=400, seed=3)
    naive = fit_polyband(data.x, data.y, 1, 0)
    aware = fit_polyband(data.x, data.y, 1, 0, yerr=data.yerr)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    grid = naive.grid()
    ax.errorbar(data.x, data.y, yerr=data.yerr, fmt="o", markersize=3,
                color=MUTED, alpha=0.45, elinewidth=0.8, capsize=0,
                linestyle="none", label="Data with their error bars")

    lo, hi = naive.envelope(grid)
    ax.fill_between(grid, lo, hi, color=WARN, alpha=0.16, linewidth=0,
                    label=f"Band ignoring yerr  (width {naive.scatter(5.0):.2f})")
    lo, hi = aware.envelope(grid)
    ax.fill_between(grid, lo, hi, color=ACCENT, alpha=0.26, linewidth=0,
                    label=f"Intrinsic band from yerr  (width {aware.scatter(5.0):.2f})")
    ax.plot(grid, aware.predict(grid), color=ACCENT, linewidth=2.1,
            label="Trend")
    for sign in (-1, 1):
        ax.plot(grid, data.true_mean(grid) + sign * data.true_sigma(grid),
                color=TRUTH, linestyle="--", linewidth=1.4,
                label="True intrinsic width (0.45)" if sign > 0 else "_nolegend_")
    style(ax)
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "yerr")


# ----------------------------------------------------------------------
def fig_bins():
    """The binned alternative, and why a continuous fit beats it."""
    data = make_trumpet(n=220, seed=8)
    edges = np.arange(0, 11, 1.0)
    centres, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (data.x >= lo) & (data.x < hi)
        if sel.sum() > 1:
            centres.append(0.5 * (lo + hi))
            means.append(float(np.mean(data.y[sel])))
            stds.append(float(np.std(data.y[sel], ddof=1)))
    centres = np.array(centres)
    means = np.array(means)
    stds = np.array(stds)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)

    axes[0].scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0)
    axes[0].fill_between(centres, means - stds, means + stds, color=WARN,
                         alpha=0.20, linewidth=0, step="mid")
    axes[0].plot(centres, means, "o-", color=WARN, linewidth=1.8, markersize=5)
    axes[0].set_title("Binned mean $\\pm$ standard deviation\n"
                      "jumpy, bin-width dependent, no value between bins",
                      fontsize=10)
    style(axes[0])

    fit = fit_polyband(data.x, data.y, 2, 1)
    grid = fit.grid()
    axes[1].scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0)
    lo, hi = fit.envelope(grid)
    axes[1].fill_between(grid, lo, hi, color=ACCENT, alpha=0.20, linewidth=0)
    axes[1].plot(grid, fit.predict(grid), color=ACCENT, linewidth=2.1)
    for sign in (-1, 1):
        axes[1].plot(grid, data.true_mean(grid) + sign * data.true_sigma(grid),
                     color=TRUTH, linestyle="--", linewidth=1.3)
        axes[0].plot(grid, data.true_mean(grid) + sign * data.true_sigma(grid),
                     color=TRUTH, linestyle="--", linewidth=1.3)
    axes[1].set_title("polyband: smooth, continuous,\nrecovers the true width "
                      "(dashed)", fontsize=10)
    style(axes[1], ylabel="")
    save(fig, "bins_vs_polyband")


# ----------------------------------------------------------------------
def fig_diagnostics():
    """The four-panel calibration check."""
    data = make_trumpet(n=800, seed=5)
    fit = fit_polyband(data.x, data.y, 2, 1)
    fig = plot_diagnostics(fit, data.x, data.y, color=ACCENT)
    for ax in fig.axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    save(fig, "diagnostics")


# ----------------------------------------------------------------------
def fig_underfit_width():
    """What a too-rigid width polynomial does to the residuals."""
    data = make_trumpet(n=900, seed=6)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)

    for ax, ow, title in ((axes[0], 0, "order_width=0: too rigid"),
                          (axes[1], 1, "order_width=1: correct")):
        fit = fit_polyband(data.x, data.y, 2, ow)
        z = fit.zscore(data.x, data.y)
        ax.scatter(data.x, z, s=14, alpha=0.35, color=MUTED, linewidths=0)
        for k, ls in ((1, "--"), (2, ":")):
            for sign in (-1, 1):
                ax.axhline(sign * k, color=ACCENT, linestyle=ls, linewidth=1.2)
        ax.axhline(0, color=ACCENT, linewidth=1.5)
        inside = float(np.mean(np.abs(z) < 1)) * 100
        ax.set_title(f"{title}\n{inside:.0f}% inside 1$\\sigma$ "
                     f"(expected 68%)", fontsize=10)
        style(ax, ylabel="standardised residual" if ow == 0 else "")
    save(fig, "underfit_width")


if __name__ == "__main__":
    print("Generating polyband documentation figures")
    fit = fig_hero()
    fig_band_vs_error()
    fig_orders()
    fig_robust()
    fig_logspace()
    fig_yerr()
    fig_bins()
    fig_diagnostics()
    fig_underfit_width()

    print("\nHero fit summary, quoted on the page:")
    print(fit.summary())
    data = make_trumpet(n=450, seed=0)
    best, table = select_orders(data.x, data.y, 4, 2)
    print("\nBIC ranking, top 5:")
    for om, ow, val in table[:5]:
        print(f"  order_mean={om} order_width={ow}  BIC={val:.1f}")
    print("\nCoverage of the hero fit:")
    for k, obs, exp in fit.coverage(data.x, data.y):
        print(f"  {k:.0f} sigma: {obs:.3f} observed vs {exp:.3f} expected")
