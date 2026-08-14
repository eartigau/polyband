"""Matplotlib integration for polyband.

Two entry points cover almost everything:

- :func:`plot_polyband` draws a finished fit onto an axis, or onto the current
  axis if you do not give it one.
- :func:`polyband_plot` fits and draws in a single call, for when you just
  want to look at your data.

Both return a :class:`BandArtists` holding every artist they created, so
anything can be restyled afterwards without digging through ``ax.get_lines()``.

Nothing here calls ``plt.show()`` or ``plt.style.use()``. The helpers draw on
the axis you give them and respect whatever style you have set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import PolyBandFit, fit_polyband

__all__ = ["BandArtists", "plot_polyband", "polyband_plot", "plot_diagnostics"]


@dataclass
class BandArtists:
    """Artists created by :func:`plot_polyband`, for later restyling.

    Attributes
    ----------
    fit : PolyBandFit
        The fit that was drawn, handy when :func:`polyband_plot` created it.
    points : matplotlib artist or None
        The scatter of the data, if it was drawn.
    trend : matplotlib line or None
        The central trend curve.
    bands : list
        One filled region per entry of ``nsigma``, widest first.
    trend_band : matplotlib artist or None
        The confidence band on the trend itself.
    ax : matplotlib axis
        The axis everything was drawn on.
    """

    fit: PolyBandFit
    ax: Any
    points: Any = None
    trend: Any = None
    bands: List[Any] = field(default_factory=list)
    trend_band: Any = None

    @property
    def legend_handles(self) -> List[Any]:
        """Every artist that carries a label, in a sensible legend order."""
        candidates = [self.points, self.trend, self.trend_band, *self.bands]
        out = []
        for artist in candidates:
            if artist is None:
                continue
            label = getattr(artist, "get_label", lambda: "")()
            if label and not label.startswith("_"):
                out.append(artist)
        return out


def _sigma_label(k: float, log_y: bool) -> str:
    ktxt = f"{k:g}"
    return rf"${ktxt}\sigma$ band"


def plot_polyband(
    fit: PolyBandFit,
    x: Optional[Sequence[float]] = None,
    y: Optional[Sequence[float]] = None,
    ax: Any = None,
    nsigma: Sequence[float] = (1.0,),
    color: str = "tab:orange",
    point_color: str = "tab:blue",
    n: int = 400,
    extrapolate: float = 0.0,
    show_points: bool = True,
    show_trend: bool = True,
    show_band: bool = True,
    show_trend_error: bool = False,
    band_alpha: float = 0.20,
    trend_error_alpha: float = 0.45,
    labels: bool = True,
    label_prefix: str = "",
    point_kw: Optional[Dict[str, Any]] = None,
    trend_kw: Optional[Dict[str, Any]] = None,
    band_kw: Optional[Dict[str, Any]] = None,
) -> BandArtists:
    """Draw a fitted trend and its band on a matplotlib axis.

    Parameters
    ----------
    fit : PolyBandFit
        A fit from :func:`polyband.fit_polyband`.
    x, y : array_like, optional
        The underlying data. Needed only to draw the points; the curves come
        entirely from ``fit``.
    ax : matplotlib axis, optional
        Target axis. Defaults to the current axis.
    nsigma : sequence of float, default (1.0,)
        Band levels to draw. Passing ``(1, 2)`` gives nested bands with the
        outer one more transparent, which reads well on a dense scatter.
    color : str
        Colour of the trend and of the bands.
    point_color : str
        Colour of the data points.
    n : int
        Number of samples along the curves.
    extrapolate : float, default 0
        Extend the curves by this fraction of the fitted x range on each side.
        The default of 0 stops them exactly where the data stop, which is the
        honest choice: a polynomial band has no business being drawn where
        nothing constrains it. Set it deliberately if you need to show an
        extrapolation, and consider styling that part with ``trend_kw``.
    show_points, show_trend, show_band : bool
        Toggle each element.
    show_trend_error : bool, default False
        Also draw the confidence band on the trend itself. Off by default
        because it is a different quantity from the scatter band and putting
        both up without explanation invites confusion. Turn it on when the
        distinction is the point you are making.
    band_alpha, trend_error_alpha : float
        Opacities of the filled regions.
    labels : bool
        Attach legend labels to the artists. The function never calls
        ``ax.legend()`` itself; use ``artists.legend_handles``.
    label_prefix : str
        Prepended to every label, useful when overlaying two fits.
    point_kw, trend_kw, band_kw : dict, optional
        Extra keyword arguments forwarded to ``ax.scatter``, ``ax.plot`` and
        ``ax.fill_between``, overriding the defaults above.

    Returns
    -------
    BandArtists

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from polyband import fit_polyband, plot_polyband
    >>> fit = fit_polyband(x, y, order_mean=2, order_width=1)   # doctest: +SKIP
    >>> art = plot_polyband(fit, x, y, nsigma=(1, 2))           # doctest: +SKIP
    >>> art.ax.legend(handles=art.legend_handles)               # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    artists = BandArtists(fit=fit, ax=ax)
    curve_x = fit.grid(n=n, extrapolate=extrapolate)

    if show_points and x is not None and y is not None:
        kw = dict(s=26, alpha=0.25, color=point_color, linewidths=0,
                  label=f"{label_prefix}Data" if labels else "_nolegend_")
        kw.update(point_kw or {})
        artists.points = ax.scatter(np.asarray(x), np.asarray(y), **kw)

    if show_band:
        # Widest first, so narrower bands land on top. The alpha rises as the
        # bands narrow, so the innermost one reads as the densest both on the
        # plot and in the legend swatches.
        levels = sorted(nsigma, reverse=True)
        for i, k in enumerate(levels):
            lo, hi = fit.envelope(curve_x, nsigma=k)
            kw = dict(color=color, alpha=band_alpha / (len(levels) - i),
                      linewidth=0,
                      label=(f"{label_prefix}{_sigma_label(k, fit.log_y)}"
                             if labels else "_nolegend_"))
            kw.update(band_kw or {})
            artists.bands.append(ax.fill_between(curve_x, lo, hi, **kw))

    if show_trend_error and fit.cov is not None:
        lo, hi = fit.trend_band(curve_x, nsigma=1.0)
        artists.trend_band = ax.fill_between(
            curve_x, lo, hi, color=color, alpha=trend_error_alpha, linewidth=0,
            label=(f"{label_prefix}Trend uncertainty" if labels else "_nolegend_"),
        )

    if show_trend:
        kw = dict(color=color, linewidth=2.2,
                  label=f"{label_prefix}Trend" if labels else "_nolegend_")
        kw.update(trend_kw or {})
        artists.trend, = ax.plot(curve_x, fit.predict(curve_x), **kw)

    if fit.log_y:
        ax.set_yscale("log")

    return artists


def polyband_plot(
    x,
    y,
    order_mean: int = 2,
    order_width: int = 1,
    ax: Any = None,
    fit_kw: Optional[Dict[str, Any]] = None,
    **plot_kw,
) -> BandArtists:
    """Fit and draw in one call.

    A thin convenience wrapper: everything in ``fit_kw`` goes to
    :func:`polyband.fit_polyband`, everything else to :func:`plot_polyband`.

    Examples
    --------
    >>> from polyband import polyband_plot
    >>> art = polyband_plot(x, y, order_mean=2, order_width=1)   # doctest: +SKIP
    >>> print(art.fit.summary())                                 # doctest: +SKIP
    """
    fit = fit_polyband(x, y, order_mean=order_mean, order_width=order_width,
                       **(fit_kw or {}))
    return plot_polyband(fit, x=x, y=y, ax=ax, **plot_kw)


def plot_diagnostics(fit: PolyBandFit, x, y, fig=None, color: str = "tab:orange"):
    """Four-panel check that the band is actually calibrated.

    A band is only meaningful if the right fraction of points falls inside it,
    at every x. The panels are:

    1. **Standardised residuals against x.** Should look like a structureless
       strip of constant width between the dashed lines. A funnel shape means
       the width polynomial has too low an order; a wave means the trend does.
    2. **Histogram of the standardised residuals**, against a unit Gaussian.
    3. **Quantile-quantile plot.** Points off the diagonal at the ends are
       heavy tails, which is the signal to try a Student-t likelihood.
    4. **Coverage against the Gaussian expectation.** The curve should sit on
       the diagonal.

    Parameters
    ----------
    fit : PolyBandFit
    x, y : array_like
        The data the fit was made from.
    fig : matplotlib figure, optional
        Figure to draw into. A new 2x2 figure is made if omitted.
    color : str
        Accent colour for the reference curves.

    Returns
    -------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = fit.zscore(x, y)
    good = np.isfinite(z)
    x, z = x[good], z[good]

    if fig is None:
        fig, _ = plt.subplots(2, 2, figsize=(10.5, 7.5))
    axes = np.asarray(fig.axes).ravel()[:4]
    ax1, ax2, ax3, ax4 = axes

    # 1. residuals vs x
    ax1.scatter(x, z, s=18, alpha=0.35, color="tab:blue", linewidths=0)
    for k, style in ((1, "--"), (2, ":")):
        for sign in (-1, 1):
            ax1.axhline(sign * k, color=color, linestyle=style, linewidth=1.2)
    ax1.axhline(0, color=color, linewidth=1.6)
    ax1.set_xlabel("x")
    ax1.set_ylabel("standardised residual")
    ax1.set_title("Residuals vs x", fontsize=10)

    # 2. histogram vs unit gaussian
    ax2.hist(z, bins=max(10, int(np.sqrt(z.size))), density=True,
             color="tab:blue", alpha=0.45, edgecolor="none")
    zz = np.linspace(-4, 4, 200)
    ax2.plot(zz, np.exp(-0.5 * zz ** 2) / np.sqrt(2 * np.pi), color=color,
             linewidth=2.0, label="unit Gaussian")
    ax2.set_xlabel("standardised residual")
    ax2.set_ylabel("density")
    ax2.set_title("Residual distribution", fontsize=10)
    ax2.legend(fontsize=8)

    # 3. QQ plot
    from scipy.stats import norm

    order = np.sort(z)
    probs = (np.arange(1, order.size + 1) - 0.5) / order.size
    ax3.plot(norm.ppf(probs), order, ".", color="tab:blue", markersize=4,
             alpha=0.6)
    lim = float(np.nanmax(np.abs(order))) * 1.05
    ax3.plot([-lim, lim], [-lim, lim], color=color, linewidth=1.6)
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set_xlabel("theoretical quantile")
    ax3.set_ylabel("observed quantile")
    ax3.set_title("Quantile-quantile", fontsize=10)

    # 4. coverage
    levels = np.linspace(0.1, 3.0, 40)
    observed = [float(np.mean(np.abs(z) < k)) for k in levels]
    expected = [float(norm.cdf(k) - norm.cdf(-k)) for k in levels]
    ax4.plot(expected, observed, color="tab:blue", linewidth=2.0)
    ax4.plot([0, 1], [0, 1], color=color, linewidth=1.6, linestyle="--")
    for k in (1.0, 2.0):
        e = float(norm.cdf(k) - norm.cdf(-k))
        o = float(np.mean(np.abs(z) < k))
        ax4.plot([e], [o], "o", color=color, markersize=6)
        ax4.annotate(f"{k:g}$\\sigma$", (e, o), textcoords="offset points",
                     xytext=(6, -10), fontsize=9, color=color)
    ax4.set_xlabel("expected fraction inside")
    ax4.set_ylabel("observed fraction inside")
    ax4.set_title("Band coverage", fontsize=10)

    fig.tight_layout()
    return fig
