#!/usr/bin/env python3
"""Driving the matplotlib layer: styling, overlays, subplots, custom legends.

    python examples/05_matplotlib_integration.py

Writes matplotlib_integration.pdf next to this file.

plot_polyband() never touches the figure-level state. It draws on the axis you
give it, adds nothing to the legend by itself, and returns every artist it
created so you can restyle after the fact. That makes it safe to call several
times on the same axis, which is how you overlay two fits.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polyband import fit_polyband, plot_polyband, polyband_plot
from polyband.datasets import make_heavy_tails, make_trumpet

data = make_trumpet(n=350, seed=21)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

# ----------------------------------------------------------------------
# 1. One-liner. Fits and draws with sensible defaults.
# ----------------------------------------------------------------------
art = polyband_plot(data.x, data.y, order_mean=2, order_width=1, ax=axes[0, 0])
axes[0, 0].legend(handles=art.legend_handles, fontsize=8)
axes[0, 0].set_title("polyband_plot: fit and draw in one call", fontsize=10)

# ----------------------------------------------------------------------
# 2. Restyling through the keyword dictionaries, and after the fact through
#    the returned artists.
# ----------------------------------------------------------------------
fit = fit_polyband(data.x, data.y, 2, 1)
art = plot_polyband(
    fit, data.x, data.y, ax=axes[0, 1],
    nsigma=(1, 2, 3),
    color="#c77dff",
    point_kw=dict(s=14, alpha=0.5, color="#4cc9f0", marker="^"),
    trend_kw=dict(linewidth=3.0, linestyle="-."),
    band_alpha=0.30,
)
# Every artist is reachable afterwards.
art.trend.set_zorder(10)
for band in art.bands:
    band.set_edgecolor("#c77dff")
    band.set_linewidth(0.6)
axes[0, 1].legend(handles=art.legend_handles, fontsize=8)
axes[0, 1].set_title("Restyled, with three nested levels", fontsize=10)

# ----------------------------------------------------------------------
# 3. Two fits on one axis. label_prefix keeps the legend readable.
# ----------------------------------------------------------------------
contaminated = make_heavy_tails(n=500, seed=1, contamination=0.09)
gaussian = fit_polyband(contaminated.x, contaminated.y, 2, 0)
robust = fit_polyband(contaminated.x, contaminated.y, 2, 0, nu=4)

art_g = plot_polyband(gaussian, contaminated.x, contaminated.y, ax=axes[1, 0],
                      color="tab:red", label_prefix="Gaussian: ")
art_r = plot_polyband(robust, ax=axes[1, 0], show_points=False,
                      color="tab:green", label_prefix="Student-t: ")
axes[1, 0].legend(handles=art_g.legend_handles + art_r.legend_handles,
                  fontsize=8)
axes[1, 0].set_title("Two fits overlaid on one axis", fontsize=10)

# ----------------------------------------------------------------------
# 4. The band against the uncertainty on the trend, plus a deliberate
#    extrapolation drawn as a dashed continuation.
# ----------------------------------------------------------------------
ax = axes[1, 1]
art = plot_polyband(fit, data.x, data.y, ax=ax, show_trend_error=True,
                    color="tab:orange")

# Curves stop at the data by default. To show an extrapolation, draw it
# separately and style it so nobody mistakes it for a constrained region.
outside = np.linspace(fit.x_max, fit.x_max + 2.5, 60)
ax.plot(outside, fit.predict(outside), color="tab:orange", linestyle=":",
        linewidth=2.0, label="Extrapolation (unconstrained)")
lo, hi = fit.envelope(outside)
ax.fill_between(outside, lo, hi, color="tab:orange", alpha=0.07, linewidth=0)
ax.legend(handles=art.legend_handles + [ax.lines[-1]], fontsize=8)
ax.set_title("Scatter band, trend error, and an honest extrapolation",
             fontsize=10)

for ax in axes.ravel():
    ax.set_xlabel("x")
    ax.set_ylabel("y")

out = Path(__file__).resolve().parent / "matplotlib_integration.pdf"
fig.tight_layout()
fig.savefig(out)
print(f"Saved: {out}")
