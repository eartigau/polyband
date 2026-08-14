# polyband

**Fit the scatter, not just the mean.**

`polyband` fits a smooth trend through a scatter plot *and* a smooth
description of how wide the scatter is around it, with the degree of each
polynomial chosen independently.

Documentation and worked examples: **https://eartigau.github.io/polyband/**
(bilingual, English and French)

![polyband](docs/figures/en/hero.svg)

## What problem this solves

You have a cloud of points and you want two things: where the middle of the
cloud goes, and how thick the cloud is. The usual approaches each fall short.

- **Binning in x** gives a jumpy answer that depends on the bin width, says
  nothing between bin centres, and wastes points near the edges.
- **A polynomial fit plus the covariance matrix** answers a different
  question. The covariance gives the uncertainty on the *fitted curve*, which
  shrinks like `1/sqrt(N)`. The thickness of the cloud does not shrink at all
  as you collect more data.
- **A polynomial fit plus a single global RMS** works only if the scatter is
  the same everywhere, which is usually the first thing that fails.

`polyband` fits both polynomials at once by maximum likelihood:

```
y = P_mean(x) + noise,   noise ~ N(0, s(x))
ln s(x) = P_width(x)
```

Because the two are fitted together, the trend is automatically weighted by
the local scatter: points in the noisy part of the x range pull less on the
mean than points in the quiet part.

## Install

```bash
pip install git+https://github.com/eartigau/polyband.git
```

Or, to work on it locally:

```bash
git clone https://github.com/eartigau/polyband.git
cd polyband
pip install -e ".[dev]"
pytest
```

Requires Python 3.9+, numpy and scipy. matplotlib is needed only for the
plotting helpers.

## Quick start

```python
import matplotlib.pyplot as plt
from polyband import fit_polyband, plot_polyband

fit = fit_polyband(x, y, order_mean=2, order_width=1)

print(fit.summary())
print(fit.predict(5.0))       # trend at x = 5
print(fit.scatter(5.0))       # half-width of the 1-sigma band at x = 5
print(fit.zscore(5.0, 12.3))  # how unusual a point is, in sigma

art = plot_polyband(fit, x, y, nsigma=(1, 2))
art.ax.legend(handles=art.legend_handles)
plt.show()
```

`order_mean` is the degree of the trend, `order_width` the degree of
`ln(sigma)`. They are independent: a curved trend with a constant-width band
(`2, 0`) is an ordinary combination, and so is a straight trend whose scatter
grows (`1, 1`).

Not sure which degrees to use? Let BIC decide:

```python
from polyband import select_orders

fit, table = select_orders(x, y, max_order_mean=4, max_order_width=2)
print(fit.order_mean, fit.order_width)
```

Treat a BIC difference below about 2 as a tie and keep the simpler model.

## The band and the trend error are different things

This is the distinction the package is built around, so it is worth stating
plainly:

| | what it describes | behaviour as N grows |
|---|---|---|
| `fit.envelope(x)` | where the **points** lie | converges to the true spread |
| `fit.trend_band(x)` | where the **curve** lies | shrinks like `1/sqrt(N)` |

![band versus trend error](docs/figures/en/band_vs_error.svg)

If you are asking "is this object unusual compared to the population?", you
want the first one. If you are asking "how well do I know the mean relation?",
you want the second.

## Options

| Argument | What it does |
|---|---|
| `order_mean`, `order_width` | Degrees of the two polynomials, independent |
| `log_y=True` | Fit `log10(y)`, for quantities spanning decades or with multiplicative scatter. Results come back in the units of y |
| `yerr=...` | Per-point measurement errors, so the fitted width is the *intrinsic* scatter rather than the observed spread |
| `nu=4` | Student-t likelihood, for samples with outliers |
| `dof_correction=True` | Undo the downward bias of maximum-likelihood widths. On by default |
| `n_bootstrap=300` | Estimate the coefficient covariance by resampling instead of from the Hessian |

## Always check the band

A band nobody has checked is a decoration. Two lines:

```python
for nsigma, observed, expected in fit.coverage(x, y):
    print(f"{nsigma:.0f} sigma: {observed:.1%} observed, {expected:.1%} expected")
```

and, for the full picture, `plot_diagnostics(fit, x, y)`: standardised
residuals against x, their distribution, a quantile-quantile plot, and a
coverage curve. A funnel shape in the first panel means `order_width` is too
low; a wave means `order_mean` is.

## Examples

Runnable scripts in [`examples/`](examples/):

1. `01_quickstart.py`: fit, check coverage, plot
2. `02_choosing_orders.py`: BIC selection and residual diagnostics
3. `03_robust_and_errors.py`: outliers with `nu`, measurement errors with `yerr`
4. `04_log_space.py`: quantities spanning orders of magnitude
5. `05_matplotlib_integration.py`: styling, overlays, subplots, extrapolation

All of them run on synthetic data from `polyband.datasets`, which ships the
ground truth alongside each sample so a fitted band can be compared against
the width it was actually drawn from.

## License

MIT
