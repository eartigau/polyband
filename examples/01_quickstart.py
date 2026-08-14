#!/usr/bin/env python3
"""The shortest useful polyband script: fit, plot, save.

    python examples/01_quickstart.py

Writes quickstart.pdf next to this file.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from polyband import fit_polyband, plot_polyband
from polyband.datasets import make_trumpet

# Any (x, y) arrays would do here; this one comes with its ground truth so the
# fit can be checked against it.
data = make_trumpet(n=400, seed=0)

# order_mean is the degree of the trend, order_width the degree of ln(sigma).
# They are independent: a curved trend can perfectly well have a band whose
# width grows linearly, which is exactly this case.
fit = fit_polyband(data.x, data.y, order_mean=2, order_width=1)

print(fit.summary())

# The band should contain about 68% of the points at 1 sigma. Always worth
# checking: a band nobody has checked is just a decoration.
print("\ncoverage:")
for nsigma, observed, expected in fit.coverage(data.x, data.y):
    print(f"  {nsigma:.0f} sigma: {observed:.1%} observed, {expected:.1%} expected")

fig, ax = plt.subplots(figsize=(8.5, 5.0))
artists = plot_polyband(fit, data.x, data.y, ax=ax, nsigma=(1, 2))
ax.legend(handles=artists.legend_handles, fontsize=9)
ax.set_xlabel("x")
ax.set_ylabel("y")

out = Path(__file__).resolve().parent / "quickstart.pdf"
fig.tight_layout()
fig.savefig(out)
print(f"\nSaved: {out}")
