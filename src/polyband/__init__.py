"""polyband: polynomial regression for the mean relation and for its scatter.

A polynomial regression that determines two curves at once: the mean
relation running through a scatter plot, and the envelope describing how
wide the scatter around it is, with the degree of each polynomial chosen
independently.

Quick start
-----------
>>> import matplotlib.pyplot as plt
>>> from polyband import fit_polyband, plot_polyband
>>> from polyband.datasets import make_trumpet
>>> data = make_trumpet(seed=0)
>>> fit = fit_polyband(data.x, data.y, order_mean=2, order_width=1)
>>> art = plot_polyband(fit, data.x, data.y, nsigma=(1, 2))
>>> _ = art.ax.legend(handles=art.legend_handles)

The band returned by ``fit.envelope()`` is the spread of the *points*. The
uncertainty on the trend itself is a separate and much narrower thing,
available as ``fit.mean_error()``. Confusing the two is the mistake this
package exists to make hard.
"""

from .core import PolyBandFit, fit_polyband, select_orders
from .plotting import BandArtists, plot_diagnostics, plot_polyband, polyband_plot

__version__ = "0.1.0"

__all__ = [
    "PolyBandFit",
    "fit_polyband",
    "select_orders",
    "plot_polyband",
    "polyband_plot",
    "plot_diagnostics",
    "BandArtists",
    "__version__",
]
