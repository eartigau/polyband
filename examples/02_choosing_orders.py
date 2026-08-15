#!/usr/bin/env python3
"""Letting BIC pick the two polynomial degrees, and checking the choice.

    python examples/02_choosing_orders.py

Writes choosing_orders.pdf next to this file.

The rule of thumb: treat a BIC difference below about 2 as a tie and keep the
simpler model. An over-flexible width polynomial is not a harmless extra
degree of freedom, it produces a band that wiggles to chase noise and then
misstates how unusual any given point is.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polyband import fit_polyband, plot_diagnostics, select_orders
from polyband.datasets import make_trumpet

# ----------------------------------------------------------------------
# 1. Data whose true degrees are known, so the answer can be graded
# ----------------------------------------------------------------------
# make_trumpet generates a quadratic trend (order_mean=2) with a width that
# is an exponential of a linear function, i.e. ln(sigma) linear in x
# (order_width=1). A correct selection should land on exactly that pair.
data = make_trumpet(n=500, seed=12)

# ----------------------------------------------------------------------
# 2. Let BIC scan the grid
# ----------------------------------------------------------------------
# Every combination from (0, 0) to (4, 2) is fitted and scored, 15 fits in
# all. Combinations that fail are skipped rather than raising. `best` comes
# back already fitted, so there is nothing to redo afterwards.
best, table = select_orders(data.x, data.y, max_order_mean=4, max_order_width=2)

# `table` is (order_mean, order_width, criterion), best first. Print the
# delta against the winner rather than the raw values: only differences
# mean anything, and a delta under about 2 is a tie.
print("order_mean  order_width       BIC   delta")
best_bic = table[0][2]
for order_mean, order_width, bic in table[:8]:
    print(f"{order_mean:10d}  {order_width:11d}  {bic:8.1f}  {bic - best_bic:6.1f}")

print(f"\nSelected: order_mean={best.order_mean}, order_width={best.order_width}")
print(f"Truth for this dataset: order_mean=2, order_width=1")

# ----------------------------------------------------------------------
# 3. Check the winner is actually adequate
# ----------------------------------------------------------------------
# The information criterion says which candidate is preferred; it cannot say
# whether the preferred one is any good, because it only ever compares the
# models you offered it. The residual diagnostics answer that second
# question. Both matter, and they are not interchangeable.
fig = plot_diagnostics(best, data.x, data.y)
out = Path(__file__).resolve().parent / "choosing_orders.pdf"
fig.savefig(out)
print(f"\nSaved: {out}")

# ----------------------------------------------------------------------
# 4. Quantify the funnel instead of eyeballing it
# ----------------------------------------------------------------------
# A width polynomial of too low an order leaves a funnel in the standardised
# residuals. The fit forces their spread to be 1 ON AVERAGE whatever the
# degree, so the average tells you nothing; what breaks is the spread within
# each slice of x. order_width=0 below is visibly wrong slice by slice while
# still averaging to 1, which is exactly the failure to watch for.
print("\nspread of standardised residuals, by quarter of the x range:")
for order_width in (0, 1):
    fit = fit_polyband(data.x, data.y, order_mean=2, order_width=order_width)
    z = fit.zscore(data.x, data.y)
    edges = np.percentile(data.x, [0, 25, 50, 75, 100])
    spreads = [np.std(z[(data.x >= lo) & (data.x <= hi)])
               for lo, hi in zip(edges[:-1], edges[1:])]
    formatted = "  ".join(f"{s:.2f}" for s in spreads)
    print(f"  order_width={order_width}: {formatted}")
