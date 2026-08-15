#!/usr/bin/env python3
"""The shortest useful polyband script, written to be read start to finish.

    python examples/01_quickstart.py

Writes quickstart.pdf next to this file.

Unlike the other examples this one builds its x and y by hand rather than
importing a ready-made dataset, so that every number entering the fit is
visible on the page. Swap step 1 for your own two arrays and the rest works
unchanged.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polyband import fit_polyband, plot_polyband

# ----------------------------------------------------------------------
# 1. The data
#
# Replace this whole section with your own x and y. They only have to be
# two 1-D arrays of the same length; NaN and inf are dropped for you, in
# whole rows, so a catalogue with gaps can go straight in.
# ----------------------------------------------------------------------
rng = np.random.default_rng(0)

# Where the measurements sit along the x axis. Nothing requires them to be
# sorted or evenly spaced, and polyband never bins them.
x = rng.uniform(0, 10, 400)

# The curve the points scatter around. Here a parabola that bends back down,
# which is why order_mean=2 below.
trend = 3.0 + 1.1 * x - 0.075 * x ** 2

# How wide that scatter is at each x. This is the part that makes the package
# worth using: the width is not constant, it grows by a factor of 16 across
# the range. A single global standard deviation would be too wide on the left
# and too narrow on the right, simultaneously.
sigma = np.exp(-1.2 + 0.28 * x)
print(f"true width: {sigma.min():.2f} at x=0, {sigma.max():.2f} at x=10")

# One Gaussian draw per point, each with its own local width.
y = trend + rng.normal(0.0, sigma)

# ----------------------------------------------------------------------
# 2. The fit
#
# The two degrees are independent, and that is the whole point of the
# package. Pick each one from what you expect that curve to look like:
#
#   order_mean=2   the trend is a parabola             -> degree 2
#   order_width=1  sigma is exp(linear), so ln(sigma)
#                  is a straight line                  -> degree 1
#
# Both polynomials are fitted at the same time, by maximum likelihood, not
# in two passes. That is what lets the trend be weighted by the local
# scatter: points in the noisy part of the range pull less on the mean.
#
# If you have no idea which degrees to use, see 02_choosing_orders.py.
# ----------------------------------------------------------------------
fit = fit_polyband(x, y, order_mean=2, order_width=1)

# ----------------------------------------------------------------------
# 3. Reading the result
# ----------------------------------------------------------------------
print()
print(fit.summary())

print()
# The trend at a single x, in the units of y.
print(f"trend at x=5            : {fit.predict(5.0):.3f}")
# Half-width of the 1-sigma band there. This describes where the POINTS are,
# not how well the curve is known; those are different quantities and
# fit.mean_error() is the other one.
print(f"band half-width at x=5  : {fit.scatter(5.0):.3f}")
# The two edges of the band, ready to plot or to compare a value against.
lo, hi = fit.envelope(5.0)
print(f"1-sigma band at x=5     : {lo:.3f} to {hi:.3f}")
# How unusual a specific measurement is, in units of the LOCAL width. This is
# the question the whole package exists to answer.
print(f"y=12.3 at x=5 sits at   : {fit.zscore(5.0, 12.3):+.2f} sigma")

# ----------------------------------------------------------------------
# 4. Checking it
#
# A band nobody has checked is decoration, not a measurement. About 68% of
# the points belong inside 1 sigma and 95% inside 2. Systematic over-coverage
# at 1 sigma, with 2 and 3 sigma on target, means tails heavier than Gaussian,
# which is what the nu argument is for (see 03_robust_and_errors.py).
# ----------------------------------------------------------------------
print("\ncoverage:")
for nsigma, observed, expected in fit.coverage(x, y):
    print(f"  {nsigma:.0f} sigma: {observed:.1%} observed, {expected:.1%} expected")

# Here the truth is known, so the fit can be graded against it directly.
# On real data this is exactly the comparison you cannot make, which is why
# the coverage check above matters.
print(f"\nfitted width at x=5: {fit.scatter(5.0):.3f}   "
      f"true width: {np.exp(-1.2 + 0.28 * 5.0):.3f}")

# ----------------------------------------------------------------------
# 5. Drawing it
#
# plot_polyband draws on the axis you hand it, never calls show() or
# legend() by itself, and returns every artist it created so you can
# restyle afterwards. See 05_matplotlib_integration.py for the rest.
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.0))

# nsigma=(1, 2) nests two bands, the wider one more transparent.
artists = plot_polyband(fit, x, y, ax=ax, nsigma=(1, 2))

# Overlay the truth, which is only possible because we generated the data.
grid = fit.grid()
for sign in (-1, 1):
    ax.plot(grid, 3.0 + 1.1 * grid - 0.075 * grid ** 2
            + sign * np.exp(-1.2 + 0.28 * grid),
            color="tab:green", linestyle="--", linewidth=1.3,
            label="True width" if sign > 0 else "_nolegend_")

ax.legend(handles=artists.legend_handles + [ax.lines[-1]], fontsize=9)
ax.set_xlabel("x")
ax.set_ylabel("y")

out = Path(__file__).resolve().parent / "quickstart.pdf"
fig.tight_layout()
fig.savefig(out)
print(f"\nSaved: {out}")
