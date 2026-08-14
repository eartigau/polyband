"""Synthetic datasets for demonstrations and tests.

Everything here is generated from a seeded random number generator, so the
examples and the documentation figures are reproducible and nothing depends on
data that may not be yours to distribute.

Each generator returns a :class:`SyntheticData`, which carries the ground
truth alongside the sample. Being able to compare a fitted band against the
width it was actually drawn from is the whole point: it is the only honest way
to show that a method works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

__all__ = [
    "SyntheticData",
    "make_trumpet",
    "make_heavy_tails",
    "make_decades",
    "make_with_errors",
]


@dataclass
class SyntheticData:
    """A synthetic sample together with the truth it was drawn from.

    Attributes
    ----------
    x, y : ndarray
        The sample.
    true_mean, true_sigma : callable
        Functions of x giving the generating trend and the generating width.
        Evaluate them on a grid to overlay the truth on a fitted band.
    yerr : ndarray or None
        Per-point measurement uncertainties, when the generator produced any.
        In that case ``true_sigma`` is the *intrinsic* width, and the observed
        scatter is the two added in quadrature.
    log_y : bool
        Whether this dataset is meant to be fitted with ``log_y=True``.
    description : str
        One line saying what the dataset is for.
    """

    x: np.ndarray
    y: np.ndarray
    true_mean: Callable[[np.ndarray], np.ndarray]
    true_sigma: Callable[[np.ndarray], np.ndarray]
    yerr: Optional[np.ndarray] = None
    log_y: bool = False
    description: str = ""

    def __len__(self) -> int:
        return int(self.x.size)

    def as_tuple(self):
        """``(x, y)``, or ``(x, y, yerr)`` when uncertainties are present."""
        return (self.x, self.y) if self.yerr is None else (self.x, self.y, self.yerr)


def make_trumpet(n: int = 400, seed: int = 0, x_range=(0.0, 10.0)) -> SyntheticData:
    """A curved trend whose scatter widens steadily with x.

    The archetypal case for this package, and the one a constant-width band
    gets wrong in both directions at once: too wide on the left, too narrow on
    the right. The generating width is a true exponential of a linear
    function, which is exactly the family ``order_width=1`` represents, so a
    correct implementation should recover it closely.

    Parameters
    ----------
    n : int
        Number of points.
    seed : int
        Seed for the generator.
    x_range : tuple
        Range the points are drawn uniformly from.

    Returns
    -------
    SyntheticData
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, n)

    def mean(v):
        v = np.asarray(v, dtype=float)
        return 3.0 + 1.10 * v - 0.075 * v ** 2

    def sigma(v):
        v = np.asarray(v, dtype=float)
        return np.exp(-1.2 + 0.28 * v)

    y = mean(x) + rng.normal(0.0, sigma(x))
    return SyntheticData(
        x=x, y=y, true_mean=mean, true_sigma=sigma,
        description="Curved trend, width growing exponentially with x",
    )


def make_heavy_tails(
    n: int = 400,
    seed: int = 1,
    contamination: float = 0.08,
    outlier_scale: float = 6.0,
    x_range=(0.0, 10.0),
) -> SyntheticData:
    """A clean trend with a minority of far-flung outliers.

    A fraction ``contamination`` of the points is drawn with a width inflated
    by ``outlier_scale``. Under a Gaussian likelihood those few points drag the
    band open across the whole x range; a Student-t likelihood (``nu=4``, say)
    recovers the width of the well-behaved majority instead.

    ``true_sigma`` is the width of the clean component, which is what a robust
    fit should converge to, not the standard deviation of the full sample.

    Parameters
    ----------
    n : int
        Number of points.
    seed : int
        Seed for the generator.
    contamination : float
        Fraction of points drawn from the broad component.
    outlier_scale : float
        How much wider that component is.
    x_range : tuple
        Range the points are drawn uniformly from.

    Returns
    -------
    SyntheticData
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, n)

    def mean(v):
        v = np.asarray(v, dtype=float)
        return 2.0 + 0.9 * v - 0.06 * v ** 2

    def sigma(v):
        v = np.asarray(v, dtype=float)
        return np.full_like(v, 0.6)

    scale = np.where(rng.random(n) < contamination, outlier_scale, 1.0)
    y = mean(x) + rng.normal(0.0, sigma(x) * scale)
    return SyntheticData(
        x=x, y=y, true_mean=mean, true_sigma=sigma,
        description=f"Constant-width trend contaminated with "
                    f"{contamination:.0%} broad outliers",
    )


def make_decades(n: int = 350, seed: int = 2, x_range=(0.0, 10.0)) -> SyntheticData:
    """A quantity spanning orders of magnitude, with multiplicative scatter.

    Fit this one with ``log_y=True``. In linear space the scatter is wildly
    heteroscedastic and strongly skewed, and no polynomial band can describe
    it; in log space it is a tidy constant-width strip. ``true_mean`` and
    ``true_sigma`` are given in the linear and log10 spaces respectively, which
    is the same convention :class:`~polyband.PolyBandFit` uses.

    Parameters
    ----------
    n : int
        Number of points.
    seed : int
        Seed for the generator.
    x_range : tuple
        Range the points are drawn uniformly from.

    Returns
    -------
    SyntheticData
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, n)

    def log_mean(v):
        v = np.asarray(v, dtype=float)
        return 2.4 - 0.34 * v + 0.012 * v ** 2

    def mean(v):
        return 10 ** log_mean(v)

    def sigma(v):
        v = np.asarray(v, dtype=float)
        return 0.16 + 0.020 * v

    y = 10 ** (log_mean(x) + rng.normal(0.0, sigma(x)))
    return SyntheticData(
        x=x, y=y, true_mean=mean, true_sigma=sigma, log_y=True,
        description="Quantity spanning three decades with multiplicative scatter",
    )


def make_with_errors(
    n: int = 300,
    seed: int = 3,
    x_range=(0.0, 10.0),
) -> SyntheticData:
    """Intrinsic scatter blurred by known, unequal measurement errors.

    Each point carries its own uncertainty, larger on average at high x. The
    observed spread is the intrinsic width and the measurement error added in
    quadrature. Passing ``yerr`` to the fit recovers the intrinsic component
    alone; leaving it out gives a band describing the observed spread, which
    is a different and usually less interesting quantity.

    Parameters
    ----------
    n : int
        Number of points.
    seed : int
        Seed for the generator.
    x_range : tuple
        Range the points are drawn uniformly from.

    Returns
    -------
    SyntheticData
        With ``yerr`` populated. ``true_sigma`` is the intrinsic width.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, n)

    def mean(v):
        v = np.asarray(v, dtype=float)
        return 1.5 + 0.75 * v

    def sigma(v):
        v = np.asarray(v, dtype=float)
        return np.full_like(v, 0.45)

    # Measurement errors themselves scatter from point to point, and grow with
    # x, so that neither the intrinsic nor the measured term dominates
    # everywhere.
    yerr = rng.lognormal(mean=np.log(0.25 + 0.12 * x), sigma=0.3)
    y = mean(x) + rng.normal(0.0, np.hypot(sigma(x), yerr))
    return SyntheticData(
        x=x, y=y, true_mean=mean, true_sigma=sigma, yerr=yerr,
        description="Constant intrinsic scatter behind growing measurement errors",
    )
