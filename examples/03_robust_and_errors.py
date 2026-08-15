#!/usr/bin/env python3
"""Two ways the default Gaussian fit can mislead, and the fix for each.

    python examples/03_robust_and_errors.py

Writes robust_and_errors.pdf next to this file.

Case 1: a handful of outliers inflate the band everywhere. Fix: nu=4, a
Student-t likelihood, which stops letting the far points set the width.

Case 2: the points carry measurement errors, so the observed spread is wider
than the intrinsic spread you actually want to describe. Fix: pass yerr, and
the fitted width becomes the intrinsic component alone.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polyband import fit_polyband, plot_polyband
from polyband.datasets import make_heavy_tails, make_with_errors

fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

# ----------------------------------------------------------------------
# Case 1: outliers
# ----------------------------------------------------------------------
# 9% of these 600 points are drawn from a component six times broader. The
# clean component has a constant width of 0.60, which is what a robust fit
# should recover; the standard deviation of the full sample is not that
# number and is not what you want.
contaminated = make_heavy_tails(n=600, seed=1, contamination=0.09)

# Same data, same degrees, one argument apart. Under the Gaussian the far
# points enter the width equation as z^2, which is unbounded, so a handful of
# them set the width for the whole x range. nu=4 replaces that by a weight
# that saturates: a point at 10 sigma counts 4.8 instead of 100, so it can no
# longer dictate the answer.
#
# The cost, worth knowing: a Student-t width is a scale parameter, not a
# standard deviation. On clean Gaussian data nu=4 returns about 0.83 of the
# true sd, so do not read envelope(x, 1) as a 68% interval without checking
# coverage() first.
gaussian = fit_polyband(contaminated.x, contaminated.y, 2, 0)
robust = fit_polyband(contaminated.x, contaminated.y, 2, 0, nu=4)

truth = float(contaminated.true_sigma(np.array([5.0]))[0])
print("Case 1, contaminated sample")
print(f"  true width of the clean component : {truth:.2f}")
print(f"  Gaussian fit                      : {gaussian.scatter(5.0):.2f}")
print(f"  Student-t fit, nu=4               : {robust.scatter(5.0):.2f}")

plot_polyband(gaussian, contaminated.x, contaminated.y, ax=axes[0],
              color="tab:red", label_prefix="Gaussian ")
plot_polyband(robust, ax=axes[0], color="tab:green", show_points=False,
              label_prefix="Student-t ")
axes[0].set_title("Outliers: nu=4 recovers the clean width", fontsize=10)
axes[0].legend(fontsize=8)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# ----------------------------------------------------------------------
# Case 2: known measurement errors
# ----------------------------------------------------------------------
# Each point carries its own uncertainty, growing with x. The spread you can
# see is the intrinsic spread and the measurement noise added in quadrature:
#     observed^2 = intrinsic^2 + yerr^2
# Usually it is the intrinsic part you want to describe, since the
# measurement part says more about your instrument than about the population.
measured = make_with_errors(n=1500, seed=3)

# Without yerr the band absorbs both terms and reports the observed spread.
naive = fit_polyband(measured.x, measured.y, 1, 0)

# With yerr the sum above is done inside the likelihood, per point, so the
# fitted width is the intrinsic component alone. Note this is a subtraction
# in disguise: if the error bars are overstated the intrinsic width is driven
# towards zero, so a suspiciously small answer means check the error bars.
aware = fit_polyband(measured.x, measured.y, 1, 0, yerr=measured.yerr)

truth = float(measured.true_sigma(np.array([5.0]))[0])
print("\nCase 2, points with measurement errors")
print(f"  true intrinsic width : {truth:.2f}")
print(f"  ignoring yerr        : {naive.scatter(5.0):.2f}  (absorbs the errors)")
print(f"  passing yerr         : {aware.scatter(5.0):.2f}")

axes[1].errorbar(measured.x, measured.y, yerr=measured.yerr, fmt="o",
                 markersize=3, alpha=0.35, color="tab:blue", elinewidth=0.8,
                 linestyle="none", label="Data")
plot_polyband(naive, ax=axes[1], color="tab:red", show_points=False,
              label_prefix="observed spread ")
plot_polyband(aware, ax=axes[1], color="tab:green", show_points=False,
              label_prefix="intrinsic spread ")
axes[1].set_title("Measurement errors: yerr isolates the intrinsic scatter",
                  fontsize=10)
axes[1].legend(fontsize=8)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

out = Path(__file__).resolve().parent / "robust_and_errors.pdf"
fig.tight_layout()
fig.savefig(out)
print(f"\nSaved: {out}")
