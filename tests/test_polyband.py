"""Tests for polyband.

The important ones are the recovery tests: generate data from a known trend
and a known width, then check the fit gets both back. A band that looks
plausible but is systematically too narrow is worse than no band at all, so
calibration is tested explicitly rather than assumed.
"""

import numpy as np
import pytest

from polyband import fit_polyband, select_orders
from polyband.datasets import (
    make_decades,
    make_heavy_tails,
    make_trumpet,
    make_with_errors,
)


# ----------------------------------------------------------------------
# Recovery of the generating model
# ----------------------------------------------------------------------
def test_recovers_trend_and_width():
    data = make_trumpet(n=3000, seed=7)
    fit = fit_polyband(data.x, data.y, order_mean=2, order_width=1)

    xs = np.linspace(1.0, 9.0, 9)
    assert np.allclose(fit.predict(xs), data.true_mean(xs), rtol=0.06, atol=0.05)
    assert np.allclose(fit.scatter(xs), data.true_sigma(xs), rtol=0.12)


def test_band_is_calibrated():
    """About 68% and 95% of points should fall inside the 1 and 2 sigma bands."""
    data = make_trumpet(n=4000, seed=11)
    fit = fit_polyband(data.x, data.y, order_mean=2, order_width=1)

    for k, observed, expected in fit.coverage(data.x, data.y, nsigma=(1, 2)):
        assert abs(observed - expected) < 0.02, f"{k} sigma: {observed} vs {expected}"


def test_width_has_no_residual_x_dependence():
    """Standardised residuals must have unit spread across the whole x range."""
    data = make_trumpet(n=4000, seed=13)
    fit = fit_polyband(data.x, data.y, order_mean=2, order_width=1)
    z = fit.zscore(data.x, data.y)

    for lo, hi in ((0, 2.5), (2.5, 5), (5, 7.5), (7.5, 10)):
        chunk = z[(data.x >= lo) & (data.x < hi)]
        assert abs(np.std(chunk) - 1.0) < 0.12, f"x in [{lo}, {hi}): std={np.std(chunk)}"


def test_constant_width_is_recovered_as_constant():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10, 2000)
    y = 1.0 + 0.5 * x + rng.normal(0, 0.7, x.size)
    fit = fit_polyband(x, y, order_mean=1, order_width=1)

    # The linear term of ln(sigma) should be consistent with zero.
    xs = np.linspace(0, 10, 5)
    assert np.allclose(fit.scatter(xs), 0.7, rtol=0.1)


# ----------------------------------------------------------------------
# The trend error and the band are different things
# ----------------------------------------------------------------------
def test_mean_error_shrinks_with_n_but_band_does_not():
    small = make_trumpet(n=200, seed=5)
    large = make_trumpet(n=5000, seed=5)

    fit_small = fit_polyband(small.x, small.y, 2, 1)
    fit_large = fit_polyband(large.x, large.y, 2, 1)

    x0 = 5.0
    # Trend uncertainty falls roughly like 1/sqrt(N), so a 25-fold increase in
    # sample size should shrink it by a factor of several.
    assert fit_large.mean_error(x0) < 0.5 * fit_small.mean_error(x0)
    # The band converges to the population width instead of shrinking.
    assert abs(fit_large.scatter(x0) - fit_small.scatter(x0)) < 0.3 * fit_small.scatter(x0)
    # And it is much wider than the trend uncertainty.
    assert fit_large.scatter(x0) > 5 * fit_large.mean_error(x0)


def test_hessian_covariance_matches_bootstrap():
    data = make_trumpet(n=500, seed=17)
    hess_fit = fit_polyband(data.x, data.y, 2, 1)
    boot_fit = fit_polyband(data.x, data.y, 2, 1, n_bootstrap=200, random_state=0)

    xs = np.linspace(1, 9, 5)
    ratio = hess_fit.mean_error(xs) / boot_fit.mean_error(xs)
    assert np.all(ratio > 0.7) and np.all(ratio < 1.4)


# ----------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------
def test_log_y_round_trip():
    data = make_decades(n=2000, seed=19)
    fit = fit_polyband(data.x, data.y, 2, 1, log_y=True)

    xs = np.linspace(1, 9, 5)
    assert np.allclose(fit.predict(xs), data.true_mean(xs), rtol=0.1)
    assert np.allclose(fit.scatter(xs), data.true_sigma(xs), rtol=0.2)
    # Envelope must come back in the original units, and bracket the trend.
    lo, hi = fit.envelope(xs)
    assert np.all(lo < fit.predict(xs)) and np.all(fit.predict(xs) < hi)
    assert fit.width_units == "dex"


def test_student_t_resists_outliers():
    data = make_heavy_tails(n=1500, seed=23, contamination=0.10)
    gauss = fit_polyband(data.x, data.y, 2, 0)
    robust = fit_polyband(data.x, data.y, 2, 0, nu=4)

    truth = float(data.true_sigma(np.array([5.0]))[0])
    # The Gaussian fit is inflated by the contamination; the robust one is not.
    assert gauss.scatter(5.0) > 1.4 * truth
    assert abs(robust.scatter(5.0) - truth) < 0.35 * truth


def test_yerr_separates_intrinsic_scatter():
    data = make_with_errors(n=3000, seed=29)
    with_err = fit_polyband(data.x, data.y, 1, 0, yerr=data.yerr)
    without = fit_polyband(data.x, data.y, 1, 0)

    truth = float(data.true_sigma(np.array([5.0]))[0])
    assert abs(with_err.scatter(5.0) - truth) < 0.2 * truth
    # Ignoring the errors inflates the width, since it then absorbs them.
    assert without.scatter(5.0) > with_err.scatter(5.0)


def test_dof_correction_inflates_width():
    data = make_trumpet(n=40, seed=31)
    corrected = fit_polyband(data.x, data.y, 2, 1, dof_correction=True)
    raw = fit_polyband(data.x, data.y, 2, 1, dof_correction=False)
    assert corrected.scatter(5.0) > raw.scatter(5.0)


# ----------------------------------------------------------------------
# Order selection
# ----------------------------------------------------------------------
def test_select_orders_finds_the_truth():
    data = make_trumpet(n=2000, seed=37)
    best, table = select_orders(data.x, data.y, max_order_mean=4, max_order_width=2)
    assert (best.order_mean, best.order_width) == (2, 1)
    assert table[0][:2] == (2, 1)
    assert len(table) == 15


def test_select_orders_prefers_constant_width_when_true():
    rng = np.random.default_rng(41)
    x = rng.uniform(0, 10, 1500)
    y = 2 + 0.4 * x + rng.normal(0, 0.5, x.size)
    best, _ = select_orders(x, y, max_order_mean=3, max_order_width=2)
    assert best.order_width == 0


# ----------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------
def test_non_finite_points_are_dropped():
    data = make_trumpet(n=300, seed=43)
    y = data.y.copy()
    y[:5] = np.nan
    y[5:8] = np.inf
    fit = fit_polyband(data.x, y, 2, 1)
    assert fit.n_points == 292


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="need more than"):
        fit_polyband([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], order_mean=2, order_width=1)


def test_identical_x_raises():
    with pytest.raises(ValueError, match="identical"):
        fit_polyband(np.ones(50), np.arange(50.0), order_mean=1, order_width=0)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="same shape"):
        fit_polyband(np.arange(10.0), np.arange(9.0))


def test_log_y_drops_non_positive():
    data = make_decades(n=200, seed=47)
    y = data.y.copy()
    y[:4] = -1.0
    fit = fit_polyband(data.x, y, 2, 1, log_y=True)
    assert fit.n_points == 196


def test_scalar_and_array_shapes():
    data = make_trumpet(n=200, seed=53)
    fit = fit_polyband(data.x, data.y, 2, 1)

    assert np.ndim(fit.predict(5.0)) == 0
    assert np.ndim(fit.mean_error(5.0)) == 0
    assert fit.predict(np.linspace(1, 9, 7)).shape == (7,)
    assert fit.mean_error(np.linspace(1, 9, 7)).shape == (7,)
    lo, hi = fit.envelope(np.linspace(1, 9, 7))
    assert lo.shape == hi.shape == (7,)


def test_grid_and_in_range():
    data = make_trumpet(n=200, seed=59)
    fit = fit_polyband(data.x, data.y, 2, 1)

    g = fit.grid(n=50)
    assert g.size == 50 and np.all(fit.in_range(g))
    padded = fit.grid(n=50, extrapolate=0.1)
    assert padded.min() < fit.x_min and padded.max() > fit.x_max
    assert not np.all(fit.in_range(padded))


def test_summary_and_information_criteria():
    data = make_trumpet(n=300, seed=61)
    fit = fit_polyband(data.x, data.y, 2, 1)

    assert fit.converged
    assert np.isfinite(fit.aic) and np.isfinite(fit.bic)
    assert fit.bic > fit.aic  # BIC penalises harder at N > 7
    text = fit.summary()
    assert "order_mean=2" in text and "order_width=1" in text


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def test_plotting_creates_artists():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from polyband import plot_diagnostics, plot_polyband, polyband_plot

    data = make_trumpet(n=200, seed=67)
    fig, ax = plt.subplots()
    fit = fit_polyband(data.x, data.y, 2, 1)
    art = plot_polyband(fit, data.x, data.y, ax=ax, nsigma=(1, 2),
                        show_trend_error=True)

    assert art.points is not None and art.trend is not None
    assert len(art.bands) == 2 and art.trend_band is not None
    assert len(art.legend_handles) == 5
    plt.close(fig)

    fig, ax = plt.subplots()
    art2 = polyband_plot(data.x, data.y, ax=ax)
    assert art2.fit.n_points == len(data)
    plt.close(fig)

    fig = plot_diagnostics(fit, data.x, data.y)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_log_y_plot_sets_log_scale():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from polyband import plot_polyband

    data = make_decades(n=200, seed=71)
    fit = fit_polyband(data.x, data.y, 2, 1, log_y=True)
    fig, ax = plt.subplots()
    plot_polyband(fit, data.x, data.y, ax=ax)
    assert ax.get_yscale() == "log"
    plt.close(fig)
