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
contaminated = make_heavy_tails(n=600, seed=1, contamination=0.09)
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
measured = make_with_errors(n=1500, seed=3)
naive = fit_polyband(measured.x, measured.y, 1, 0)
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
