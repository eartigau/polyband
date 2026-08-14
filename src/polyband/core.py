"""Joint maximum-likelihood fitting of a polynomial trend and a polynomial band.

The model
---------
For data ``(x, y)``::

    y = P_mean(x) + noise,    noise ~ N(0, s(x))
    ln s(x) = P_width(x)

``P_mean`` has degree ``order_mean`` and ``P_width`` has degree
``order_width``. The two degrees are independent: a curved trend with a
constant-width band is a perfectly ordinary combination, and so is a straight
trend whose scatter grows.

Both polynomials are fitted at the same time by maximum likelihood, not in two
passes. That matters: the trend is then automatically weighted by the local
scatter, so points in the noisy part of the x range pull less on the mean than
points in the quiet part.

Three design choices worth knowing about
----------------------------------------
1. **The band is the scatter of the points, not the error on the trend.**
   These are different quantities and the difference is large. The error on
   the trend comes from the coefficient covariance matrix and shrinks like
   1/sqrt(N); the band does not shrink at all as you add data, it converges to
   the true population spread. Use :meth:`PolyBandFit.scatter` for the former
   and :meth:`PolyBandFit.mean_error` for the latter.

2. **The width polynomial describes ln(sigma), not sigma or sigma squared.**
   Fitting sigma squared with an ordinary polynomial lets the variance go
   negative, which then has to be clipped, and it weights outliers by their
   fourth power. Working in ln(sigma) makes positivity automatic and gives the
   width parameters the behaviour of a scale family.

3. **Maximum-likelihood widths are biased low**, by roughly
   sqrt((N - p) / N). ``dof_correction=True`` (the default) undoes this, the
   same way dividing by N-1 rather than N does for a plain standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
    from scipy.special import gammaln
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a hard dependency in practice
    _HAS_SCIPY = False

__all__ = ["PolyBandFit", "fit_polyband", "select_orders"]


# For z ~ N(0, sigma):  E[ln|z|] = ln(sigma) - (gamma_Euler + ln 2) / 2,
# the second term being about -0.6352. Used to de-bias the starting guess of
# the width polynomial, which is a fit to ln|residual|.
_LOG_ABS_NORMAL_OFFSET = -0.5 * (np.euler_gamma + np.log(2.0))


@dataclass
class PolyBandFit:
    """Result of a joint trend-and-band fit. Returned by :func:`fit_polyband`.

    The polynomial coefficients are stored in a rescaled variable
    ``t = (x - x_offset) / x_scale`` mapping the fitted range onto [-1, 1].
    You almost never need them directly: call the methods, which take raw x.

    Attributes
    ----------
    coeff_mean, coeff_width : ndarray
        Coefficients of ``P_mean`` and of ``P_width = ln sigma``, in ``t``,
        highest power first (the numpy convention).
    x_offset, x_scale : float
        The rescaling applied to x before fitting.
    x_min, x_max : float
        Range spanned by the fitted data. Beyond it you are extrapolating;
        :meth:`in_range` and the ``extrapolate`` argument of the plotting
        helpers exist to keep that honest.
    log_y : bool
        Whether the fit was done on log10(y). If so, :meth:`predict` and
        :meth:`envelope` return values back in the original units of y, while
        :meth:`scatter` stays in dex.
    cov : ndarray or None
        Covariance of the concatenated parameters ``[coeff_mean, coeff_width]``,
        from the inverse Hessian at the optimum, or from the bootstrap when
        ``n_bootstrap`` was used.
    n_points : int
        Number of finite points actually fitted.
    log_likelihood : float
        Log likelihood at the optimum, for :attr:`aic` / :attr:`bic`.
    nu : float or None
        Degrees of freedom of the Student-t likelihood; ``None`` for Gaussian.
    """

    coeff_mean: np.ndarray
    coeff_width: np.ndarray
    x_offset: float
    x_scale: float
    x_min: float
    x_max: float
    log_y: bool = False
    cov: Optional[np.ndarray] = None
    n_points: int = 0
    log_likelihood: float = np.nan
    nu: Optional[float] = None
    dof_corrected: bool = True
    converged: bool = True
    message: str = ""
    _has_yerr: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def order_mean(self) -> int:
        """Degree of the trend polynomial."""
        return len(self.coeff_mean) - 1

    @property
    def order_width(self) -> int:
        """Degree of the ln(sigma) polynomial."""
        return len(self.coeff_width) - 1

    @property
    def n_params(self) -> int:
        """Total number of free parameters."""
        return len(self.coeff_mean) + len(self.coeff_width)

    @property
    def width_units(self) -> str:
        """``'dex'`` when the fit was done in log10(y), else ``'y units'``."""
        return "dex" if self.log_y else "y units"

    def _t(self, x) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.x_offset) / self.x_scale

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------
    def predict(self, x):
        """Central trend at ``x``, in the original units of y.

        If the fit used ``log_y=True`` this returns ``10 ** P_mean(x)``, so it
        is directly plottable against the data.
        """
        mu = np.polyval(self.coeff_mean, self._t(x))
        return 10 ** mu if self.log_y else mu

    def predict_fit_space(self, x):
        """Central trend in the space the fit was performed in.

        Identical to :meth:`predict` unless ``log_y=True``, in which case this
        returns log10 of the trend.
        """
        return np.polyval(self.coeff_mean, self._t(x))

    def scatter(self, x):
        """Half-width of the 1-sigma band at ``x``, in :attr:`width_units`.

        With ``log_y=True`` this is a dispersion in dex, i.e. a multiplicative
        factor of ``10 ** scatter`` on the trend, which is why it is not
        returned in y units: a log-space band is asymmetric in linear space.
        """
        return np.exp(np.polyval(self.coeff_width, self._t(x)))

    def envelope(self, x, nsigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Lower and upper edges of the ``nsigma`` band, in the units of y.

        This is where the *points* live, not where the trend lives. About 68%
        of the sample should fall inside ``nsigma=1``; check it with
        :meth:`coverage`.
        """
        mu = np.polyval(self.coeff_mean, self._t(x))
        sig = self.scatter(x)
        lo, hi = mu - nsigma * sig, mu + nsigma * sig
        return (10 ** lo, 10 ** hi) if self.log_y else (lo, hi)

    def mean_error(self, x):
        """1-sigma uncertainty on the trend itself, in :attr:`width_units`.

        Propagated from the coefficient covariance matrix. This is the narrow
        band that tightens as the sample grows, as opposed to
        :meth:`envelope`, which does not. Returns NaN if no covariance is
        available.
        """
        scalar = np.ndim(x) == 0
        if self.cov is None:
            out = np.full(np.atleast_1d(np.asarray(x, dtype=float)).shape, np.nan)
            return float(out[0]) if scalar else out
        t = np.atleast_1d(self._t(x))
        design = np.vander(t, len(self.coeff_mean))
        cov_mean = self.cov[: len(self.coeff_mean), : len(self.coeff_mean)]
        var = np.einsum("ij,jk,ik->i", design, cov_mean, design)
        out = np.sqrt(np.clip(var, 0.0, None))
        return float(out[0]) if scalar else out

    def trend_band(self, x, nsigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Confidence band on the trend, in the units of y.

        The counterpart of :meth:`envelope` for the mean curve.
        """
        mu = np.polyval(self.coeff_mean, self._t(x))
        err = np.atleast_1d(self.mean_error(x))
        lo, hi = mu - nsigma * err, mu + nsigma * err
        return (10 ** lo, 10 ** hi) if self.log_y else (lo, hi)

    def zscore(self, x, y):
        """Standardised residual ``(y - trend) / sigma(x)``.

        The natural way to ask how unusual a point is, since it accounts for
        the band being wider in some parts of the x range than others. Handles
        the log transform automatically.
        """
        y = np.asarray(y, dtype=float)
        y_fit = np.log10(y) if self.log_y else y
        return (y_fit - self.predict_fit_space(x)) / self.scatter(x)

    def in_range(self, x):
        """Boolean mask of the ``x`` values inside the fitted range."""
        x = np.asarray(x, dtype=float)
        return (x >= self.x_min) & (x <= self.x_max)

    def grid(self, n: int = 400, extrapolate: float = 0.0) -> np.ndarray:
        """Convenient x grid spanning the fitted range.

        ``extrapolate`` extends it by that fraction of the range on each side.
        The default of 0 keeps curves strictly inside the data.
        """
        span = self.x_max - self.x_min
        pad = extrapolate * span
        return np.linspace(self.x_min - pad, self.x_max + pad, n)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def coverage(self, x, y, nsigma: Sequence[float] = (1.0, 2.0, 3.0)):
        """Fraction of points inside the band, against the Gaussian expectation.

        Returns a list of ``(nsigma, observed_fraction, expected_fraction)``.
        A band that is well specified should track the expectation closely;
        systematic over-coverage at 1 sigma usually means the residuals are
        heavier-tailed than Gaussian, which is what ``nu`` is for.
        """
        from math import erf, sqrt

        z = np.abs(self.zscore(x, y))
        z = z[np.isfinite(z)]
        return [(float(k), float(np.mean(z < k)), float(erf(k / sqrt(2.0))))
                for k in nsigma]

    @property
    def aic(self) -> float:
        """Akaike information criterion, lower is better."""
        return 2 * self.n_params - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        """Bayesian information criterion, lower is better.

        Penalises extra polynomial degrees more heavily than :attr:`aic`, which
        is usually what you want when choosing orders: an over-flexible width
        polynomial produces a band that wiggles to chase noise.
        """
        return self.n_params * np.log(self.n_points) - 2 * self.log_likelihood

    def summary(self) -> str:
        """Multi-line human-readable description of the fit."""
        like = "Gaussian" if self.nu is None else f"Student-t (nu = {self.nu:g})"
        lines = [
            f"PolyBandFit  order_mean={self.order_mean}  "
            f"order_width={self.order_width}  N={self.n_points}",
            f"  x range     : {self.x_min:.6g} to {self.x_max:.6g}",
            f"  fitted in   : {'log10(y)' if self.log_y else 'y'}",
            f"  likelihood  : {like}",
            f"  log L       : {self.log_likelihood:.4g}",
            f"  AIC / BIC   : {self.aic:.6g} / {self.bic:.6g}",
            "  x, trend, band half-width "
            f"[{self.width_units}]:",
        ]
        xs = np.linspace(self.x_min, self.x_max, 5)
        for xi, mi, si in zip(xs, self.predict(xs), self.scatter(xs)):
            lines.append(f"    {xi:12.6g} {mi:12.6g} {si:12.6g}")
        if not self.converged:
            lines.append(f"  WARNING: optimiser did not converge: {self.message}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"PolyBandFit(order_mean={self.order_mean}, "
                f"order_width={self.order_width}, N={self.n_points}, "
                f"log_y={self.log_y})")


# ----------------------------------------------------------------------
# Likelihood
# ----------------------------------------------------------------------
def _neg_log_like(params, design_mean, design_width, y, yvar, nu):
    """Negative log likelihood of the heteroscedastic polynomial model."""
    n_mean = design_mean.shape[1]
    mu = design_mean @ params[:n_mean]
    # Clipping keeps exp() from overflowing while the optimiser explores.
    log_sig = np.clip(design_width @ params[n_mean:], -30.0, 30.0)
    var = np.exp(2.0 * log_sig)
    if yvar is not None:
        var = var + yvar
    resid2 = (y - mu) ** 2

    if nu is None:
        return 0.5 * float(np.sum(resid2 / var + np.log(var)))

    const = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(nu * np.pi)
    ll = const - 0.5 * np.log(var) - 0.5 * (nu + 1.0) * np.log1p(resid2 / (nu * var))
    return -float(np.sum(ll))


def _numeric_hessian(func: Callable, params, args, rel_step: float = 1e-4):
    """Central-difference Hessian of ``func`` at ``params``.

    The inverse Hessian accumulated by BFGS is not trustworthy here, because
    the Nelder-Mead pre-solve usually leaves the optimiser already at the
    minimum and BFGS then exits on a precision-loss flag having built almost
    no curvature information. At these parameter counts an explicit numerical
    Hessian costs nothing and gives a covariance matrix that actually matches
    a bootstrap.
    """
    params = np.asarray(params, dtype=float)
    n = params.size
    step = rel_step * np.maximum(np.abs(params), 1.0)
    hess = np.empty((n, n))
    f0 = func(params, *args)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = step[i], step[j]
            if i == j:
                hess[i, i] = (func(params + ei, *args) - 2 * f0
                              + func(params - ei, *args)) / step[i] ** 2
            else:
                val = (func(params + ei + ej, *args) - func(params + ei - ej, *args)
                       - func(params - ei + ej, *args) + func(params - ei - ej, *args))
                hess[i, j] = hess[j, i] = val / (4 * step[i] * step[j])
    return hess


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def fit_polyband(
    x: Sequence[float],
    y: Sequence[float],
    order_mean: int = 2,
    order_width: int = 1,
    yerr: Optional[Sequence[float]] = None,
    log_y: bool = False,
    nu: Optional[float] = None,
    dof_correction: bool = True,
    n_bootstrap: int = 0,
    random_state: Optional[int] = None,
) -> PolyBandFit:
    """Fit a polynomial trend and a polynomial-width band to ``(x, y)``.

    Parameters
    ----------
    x, y : array_like
        The data. Any point whose x, y or yerr is NaN or infinite is dropped,
        silently and as a whole row, so the three arrays stay aligned. See the
        Notes section below.
    order_mean : int, default 2
        Degree of the trend polynomial. 0 is a constant, 1 a straight line.
    order_width : int, default 1
        Degree of the ``ln sigma`` polynomial, independent of ``order_mean``.
        Use 0 for a band of constant width, 1 for one that widens or narrows
        steadily. Rarely worth going above 2: an over-flexible width chases
        noise, and :func:`select_orders` will usually tell you so.
    yerr : array_like, optional
        Per-point measurement uncertainties. When supplied, the fitted width
        is the *intrinsic* scatter, with ``yerr`` added in quadrature inside
        the likelihood, so the band separates real spread from measurement
        noise. Leave as None to have the band absorb both.
    log_y : bool, default False
        Fit ``log10(y)`` instead of ``y``. Use it when y spans orders of
        magnitude or when its scatter is multiplicative rather than additive.
        Non-positive y values are dropped. Results come back in the original
        units of y, except :meth:`PolyBandFit.scatter`, which stays in dex.
    nu : float, optional
        Degrees of freedom of a Student-t likelihood. Values of 3 to 5 make
        the fit strongly resistant to outliers, at the cost of the band no
        longer being the plain standard deviation. ``None`` gives the Gaussian
        case, where the band is exactly the 1-sigma dispersion.
    dof_correction : bool, default True
        Inflate the fitted width by ``sqrt(N / (N - n_params))`` to undo the
        downward bias of maximum-likelihood variances. Matters for small
        samples, negligible for large ones.
    n_bootstrap : int, default 0
        When positive, estimate the covariance by resampling this many times
        instead of from the Hessian. Slower, but makes no assumption that the
        likelihood is locally quadratic. A few hundred is usually plenty.
    random_state : int, optional
        Seed for the bootstrap resampling.

    Returns
    -------
    PolyBandFit

    Raises
    ------
    ValueError
        If there are not more finite points than free parameters, or if all x
        values are identical.

    Notes
    -----
    **Non-finite data.** NaN and inf are treated identically and are removed
    row-wise: a point is used only if its x, its y and, when supplied, its
    yerr are all finite. With ``log_y=True`` the requirement ``y > 0`` is added,
    and with ``yerr`` the requirement ``yerr >= 0``; ``yerr == 0`` is kept and
    means an exact measurement. Removal is silent, so data with gaps can be
    passed straight in, and :attr:`PolyBandFit.n_points` reports how many
    points actually entered the fit. If fewer points survive than there are
    free parameters, a ``ValueError`` is raised rather than a meaningless fit
    returned.

    The methods of the returned :class:`PolyBandFit` propagate non-finite
    inputs rather than filtering them: ``predict(np.nan)`` is NaN, and
    :meth:`PolyBandFit.in_range` is False for NaN and for inf.
    :meth:`PolyBandFit.coverage` is the exception and drops non-finite
    residuals, since a fraction computed over NaN would be meaningless.

    Examples
    --------
    >>> import numpy as np
    >>> from polyband import fit_polyband
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 10, 500)
    >>> y = 2 + 0.5 * x + rng.normal(0, 0.2 + 0.1 * x)
    >>> fit = fit_polyband(x, y, order_mean=1, order_width=1)
    >>> lo, hi = fit.envelope(fit.grid())
    """
    if not _HAS_SCIPY:
        raise ImportError("polyband requires scipy; pip install scipy")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")

    # Non-finite handling. Every usability condition is accumulated into one
    # boolean mask, which is then applied to x, y and yerr in the same
    # operation. Doing it array by array would be the classic way to silently
    # misalign the three, so it is deliberately a single mask:
    #   - np.isfinite is False for NaN, +inf and -inf alike, so both kinds of
    #     bad value are caught by the same test;
    #   - log_y additionally drops y <= 0, which has no logarithm. NaN > 0 is
    #     already False, so that test does not need its own NaN guard;
    #   - a point with a non-finite or negative yerr is dropped rather than
    #     silently treated as having no error, since the two mean different
    #     things. yerr == 0 is legitimate and kept: it says the measurement is
    #     exact, and the likelihood handles it.
    # Dropping is silent by design, so that a catalogue with gaps can be passed
    # straight in. What survived is recorded in ``n_points``, and comparing it
    # against len(x) is the way to find out how much was thrown away.
    good = np.isfinite(x) & np.isfinite(y)
    if log_y:
        good &= y > 0
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        if yerr.shape != x.shape:
            raise ValueError("yerr must have the same shape as x and y")
        good &= np.isfinite(yerr) & (yerr >= 0)

    x, y = x[good], y[good]
    yerr_used = None
    if yerr is not None:
        yerr_used = yerr[good]

    if log_y:
        if yerr_used is not None:
            # Propagate a symmetric error into dex to first order.
            yerr_used = yerr_used / (y * np.log(10.0))
        y = np.log10(y)

    n = x.size
    n_params = (order_mean + 1) + (order_width + 1)
    if n <= n_params:
        raise ValueError(
            f"need more than {n_params} usable points for orders "
            f"({order_mean}, {order_width}), got {n}"
        )

    # Rescale x onto [-1, 1]. A Vandermonde matrix built from raw values in
    # the thousands is hopelessly ill-conditioned by order 3.
    x_offset = 0.5 * (x.max() + x.min())
    x_scale = 0.5 * (x.max() - x.min())
    if x_scale == 0:
        raise ValueError("all x values are identical, nothing to fit")
    t = (x - x_offset) / x_scale

    design_mean = np.vander(t, order_mean + 1)
    design_width = np.vander(t, order_width + 1)
    yvar = None if yerr_used is None else yerr_used ** 2

    # Starting guess: ordinary least squares for the trend, then a fit to
    # ln|residual| for the width, de-biased by E[ln|z|] - ln(sigma).
    init_mean = np.polyfit(t, y, order_mean)
    resid = y - np.polyval(init_mean, t)
    floor = 1e-8 * max(float(np.std(resid)), 1e-12)
    init_width = np.polyfit(t, np.log(np.abs(resid) + floor), order_width)
    init_width[-1] -= _LOG_ABS_NORMAL_OFFSET
    p0 = np.concatenate([init_mean, init_width])

    args = (design_mean, design_width, y, yvar, nu)
    nll0 = _neg_log_like(p0, *args)

    # Nelder-Mead first: robust from a rough start, and the likelihood is
    # mildly non-quadratic in the width parameters. BFGS then polishes.
    res = minimize(_neg_log_like, p0, args=args, method="Nelder-Mead",
                   options={"maxiter": 20000, "maxfev": 20000,
                            "xatol": 1e-10, "fatol": 1e-10})
    res = minimize(_neg_log_like, res.x, args=args, method="BFGS",
                   options={"maxiter": 10000, "gtol": 1e-8})

    # BFGS routinely reports precision loss when handed an already-converged
    # start, so judge on the objective rather than on the status flag.
    converged = bool(np.isfinite(res.fun) and res.fun <= nll0)

    coeff_mean = np.asarray(res.x[: order_mean + 1], dtype=float).copy()
    coeff_width = np.asarray(res.x[order_mean + 1:], dtype=float).copy()

    if dof_correction:
        coeff_width[-1] += 0.5 * np.log(n / (n - n_params))

    cov = None
    if n_bootstrap > 0:
        rng = np.random.default_rng(random_state)
        draws: List[np.ndarray] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            sub = minimize(
                _neg_log_like, res.x,
                args=(design_mean[idx], design_width[idx], y[idx],
                      None if yvar is None else yvar[idx], nu),
                method="BFGS", options={"maxiter": 2000},
            )
            if np.all(np.isfinite(sub.x)):
                draws.append(sub.x)
        if len(draws) > n_params:
            cov = np.cov(np.array(draws).T)
    else:
        try:
            cov = np.linalg.inv(_numeric_hessian(_neg_log_like, res.x, args))
            if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) < 0):
                cov = None
        except np.linalg.LinAlgError:
            cov = None

    return PolyBandFit(
        coeff_mean=coeff_mean,
        coeff_width=coeff_width,
        x_offset=float(x_offset),
        x_scale=float(x_scale),
        x_min=float(x.min()),
        x_max=float(x.max()),
        log_y=bool(log_y),
        cov=cov,
        n_points=int(n),
        log_likelihood=float(-res.fun),
        nu=nu,
        dof_corrected=bool(dof_correction),
        converged=converged,
        message=str(res.message),
        _has_yerr=yerr_used is not None,
    )


def select_orders(
    x,
    y,
    max_order_mean: int = 4,
    max_order_width: int = 2,
    criterion: str = "bic",
    verbose: bool = False,
    **kwargs,
) -> Tuple[PolyBandFit, List[Tuple[int, int, float]]]:
    """Scan polynomial degrees and return the best pair by AIC or BIC.

    Every combination from ``(0, 0)`` up to ``(max_order_mean,
    max_order_width)`` is fitted and scored. Combinations that fail to fit are
    skipped rather than raising.

    Prefer ``'bic'`` unless you have a reason not to: it penalises extra
    degrees more firmly, and the failure mode you want to avoid is a width
    polynomial flexible enough to chase noise. Treat differences below about
    2 as a tie and take the simpler model.

    Parameters
    ----------
    x, y : array_like
        The data, as for :func:`fit_polyband`.
    max_order_mean, max_order_width : int
        Highest degrees to try.
    criterion : {'bic', 'aic'}
        Which information criterion to minimise.
    verbose : bool
        Print the ranking as it is computed.
    **kwargs
        Passed straight through to :func:`fit_polyband` (``log_y``, ``nu``,
        ``yerr`` and so on), so the scan is done under the same assumptions as
        the final fit.

    Returns
    -------
    best : PolyBandFit
        The winning fit, ready to use.
    table : list of (order_mean, order_width, criterion_value)
        All successful combinations, best first.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError("criterion must be 'aic' or 'bic'")

    scored = []
    for om in range(max_order_mean + 1):
        for ow in range(max_order_width + 1):
            try:
                fit = fit_polyband(x, y, order_mean=om, order_width=ow, **kwargs)
            except (ValueError, np.linalg.LinAlgError):
                continue
            value = getattr(fit, criterion)
            if np.isfinite(value):
                scored.append((value, om, ow, fit))

    if not scored:
        raise RuntimeError("no order combination could be fitted")

    scored.sort(key=lambda r: r[0])
    table = [(om, ow, val) for val, om, ow, _ in scored]

    if verbose:
        print(f"order_mean  order_width  {criterion.upper()}")
        for om, ow, val in table:
            print(f"{om:10d}  {ow:11d}  {val:12.4g}")

    return scored[0][3], table
