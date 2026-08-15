#!/usr/bin/env python3
"""Fitting a quantity that spans orders of magnitude.

    python examples/04_log_space.py

Writes log_space.pdf next to this file.

When the scatter is multiplicative rather than additive, y itself is the wrong
variable to fit. Set log_y=True and polyband works on log10(y) internally
while still handing results back in the units of y, so nothing downstream has
to know about the transform.

The one place the transform stays visible is the width: fit.scatter() returns
dex, because a band that is symmetric in log space is asymmetric in linear
space and there is no single number to report otherwise. Use fit.envelope()
for the edges, which does come back in y units.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polyband import fit_polyband
from polyband.datasets import make_decades

# ----------------------------------------------------------------------
# 1. A quantity that spans decades, with multiplicative scatter
# ----------------------------------------------------------------------
data = make_decades(n=400, seed=2)
print(f"y spans {data.y.min():.2g} to {data.y.max():.2g}, "
      f"a factor of {data.y.max() / data.y.min():.0f}")

# ----------------------------------------------------------------------
# 2. The same data fitted both ways, so the two can be compared directly
# ----------------------------------------------------------------------
# Identical degrees, identical data: the ONLY difference is log_y.
linear = fit_polyband(data.x, data.y, 2, 1)
log = fit_polyband(data.x, data.y, 2, 1, log_y=True)

print(f"\nfitted in y      : width at x=5 is {linear.scatter(5.0):.2f} "
      f"{linear.width_units}")
print(f"fitted in log10(y): width at x=5 is {log.scatter(5.0):.3f} "
      f"{log.width_units}, i.e. a factor of {10 ** log.scatter(5.0):.2f}")

# ----------------------------------------------------------------------
# 3. Why the linear fit is not merely worse but impossible
# ----------------------------------------------------------------------
# A Gaussian band in y is symmetric, so for a positive quantity with large
# scatter its lower edge has to cross zero somewhere. No choice of degree
# fixes that; only changing the variable does.
grid = np.linspace(0, 10, 200)
below_zero = (linear.envelope(grid)[0] < 0).mean()
print(f"\nthe linear-space band dips below zero over {below_zero:.0%} of the "
      f"x range, which is impossible for a positive quantity")

# ----------------------------------------------------------------------
# 4. The log-space band is properly calibrated
# ----------------------------------------------------------------------
print("\ncoverage of the log-space band:")
for nsigma, observed, expected in log.coverage(data.x, data.y):
    print(f"  {nsigma:.0f} sigma: {observed:.1%} observed, {expected:.1%} expected")

# ----------------------------------------------------------------------
# 5. Draw both, with a log y axis on the right so the band reads as a strip
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, fit, title in ((axes[0], linear, "log_y=False"),
                       (axes[1], log, "log_y=True")):
    grid = fit.grid()
    ax.scatter(data.x, data.y, s=16, alpha=0.3, color="tab:blue", linewidths=0)
    lo, hi = fit.envelope(grid)
    ax.fill_between(grid, lo, hi, color="tab:orange", alpha=0.2, linewidth=0)
    ax.plot(grid, fit.predict(grid), color="tab:orange", linewidth=2.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
axes[0].axhline(0, color="0.4", linewidth=1.0)
axes[1].set_yscale("log")

out = Path(__file__).resolve().parent / "log_space.pdf"
fig.tight_layout()
fig.savefig(out)
print(f"\nSaved: {out}")

# ----------------------------------------------------------------------
# 6. Nothing downstream needs to know about the transform
# ----------------------------------------------------------------------
# zscore takes y in its original units and handles the log internally, so an
# external object can be compared to the population without you redoing the
# arithmetic. Note the asymmetry the log space implies: a factor of 10 above
# the trend and a factor of 10 below are the same distance in sigma.
for value in (3.0, 30.0):
    print(f"a point at x=5, y={value:g} sits at "
          f"{float(log.zscore(5.0, value)):+.2f} sigma")
