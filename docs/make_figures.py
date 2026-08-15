#!/usr/bin/env python3
"""Generate the figures used on the polyband documentation site.

Everything is drawn from synthetic data generated in ``polyband.datasets``,
so this script is fully reproducible and depends on no external file.

Run it from anywhere:

    python docs/make_figures.py

Each figure is written twice, once per site language, to
``docs/figures/en/`` and ``docs/figures/fr/``. The data and the fits are
identical between the two; only the text differs. The palette matches the
site's dark theme.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool

from scipy.optimize import minimize

from polyband import fit_polyband, plot_diagnostics, plot_polyband, select_orders
from polyband.core import _neg_log_like
from polyband.datasets import (
    make_decades,
    make_heavy_tails,
    make_trumpet,
    make_with_errors,
)

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent

# Site palette, from themes_outils/DESIGN_SYSTEM.md
BG = "#0a1322"
TEXT = "#e8eef8"
MUTED = "#a8b4ca"
ACCENT = "#62c2ff"
TRUTH = "#6ce2ad"
WARN = "#ffb066"
LINE = "#2a3b56"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": LINE,
    "axes.titlecolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": LINE,
    "legend.facecolor": "#0e1a2e",
    "legend.edgecolor": LINE,
    "legend.framealpha": 0.9,
    "font.size": 11,
    "axes.titlesize": 11,
    "figure.dpi": 110,
    "svg.fonttype": "none",   # keep text as text, so it stays selectable
})

NB = " "   # non-breaking space, for French punctuation spacing

# ----------------------------------------------------------------------
# Strings. Panel titles, axis labels and legends, in both site languages.
# Identifiers that name code (order_mean, log_y, yerr, nu) stay as they are.
# ----------------------------------------------------------------------
STRINGS = {
    "en": {
        "data": "Data",
        "data_n": "Data (N = {n})",
        "trend": "Trend",
        "true_width": "True 1$\\sigma$ width",
        "band_1s": "1$\\sigma$ band",

        "bve_left_title": "Two different bands, one sample",
        "bve_band": "1$\\sigma$ band: spread of the points",
        "bve_trend_err": "1$\\sigma$ error on the trend itself",
        "bve_right_title": "Adding data shrinks one and not the other",
        "bve_halfwidth": "1$\\sigma$ band half-width",
        "bve_true_at": "true width at x = {x}",
        "bve_error": "error on the trend",
        "bve_ref": r"$1/\sqrt{N}$ reference",
        "bve_xlabel": "sample size N",
        "bve_ylabel": "half-width at x = {x}",

        "robust_gauss": "Gaussian likelihood (default)",
        "robust_student": "Student-t likelihood, nu = 4",
        "robust_width": "fitted width = {w} (true {t})",
        "robust_band": "fitted 1$\\sigma$ band",
        "robust_true": "true width of the clean component",

        "log_linear": "Fitted in y: band runs below zero,\nscatter is skewed",
        "log_log": "Fitted with log_y=True: band tracks\nthe multiplicative spread",

        "yerr_data": "Data with their error bars",
        "yerr_ignoring": "Band ignoring yerr  (width {w})",
        "yerr_intrinsic": "Intrinsic band from yerr  (width {w})",
        "yerr_true": "True intrinsic width ({t})",

        "bins_left": "Binned mean $\\pm$ standard deviation\n"
                     "jumpy, bin-width dependent, no value between bins",
        "bins_right": "polyband: smooth, continuous,\n"
                      "recovers the true width (dashed)",

        "diag_resid_title": "Residuals vs x",
        "diag_resid_y": "standardised residual",
        "diag_dist_title": "Residual distribution",
        "diag_density": "density",
        "diag_gaussian": "unit Gaussian",
        "diag_qq_title": "Quantile-quantile",
        "diag_qq_x": "theoretical quantile",
        "diag_qq_y": "observed quantile",
        "diag_cov_title": "Band coverage",
        "diag_cov_x": "expected fraction inside",
        "diag_cov_y": "observed fraction inside",

        "orders_under": "too rigid",
        "orders_right": "correct",
        "orders_over": "too flexible",
        "orders_best": "best",
        "orders_ylab_mean": "y\n\nsweeping the TREND degree\n(width held correct)",
        "orders_ylab_width": "y\n\nsweeping the WIDTH degree\n(trend held correct)",

        "uf_rigid": "order_width=0: too rigid",
        "uf_correct": "order_width=1: correct",
        "uf_inside": "{p}% inside 1$\\sigma$ (expected 68%)",

        "opt_a_title": "What one point costs the fit",
        "opt_a_x": "$\\sigma_{\\rm model}\\,/\\,\\sigma_{\\rm true}$ "
                   "at that point",
        "opt_a_y": "expected cost in $-\\ln L$ (nats)",
        "opt_a_chi2": "$\\frac{1}{2}\\,z^2$ term: punishes a band too narrow",
        "opt_a_logs": "$\\ln\\sigma$ term: punishes a band too wide",
        "opt_a_total": "sum: one minimum, at the true width",
        "opt_b_title": "A band that under- and over-estimates,\n"
                       "keeping the average width right",
        "opt_b_y": "$\\sigma_{\\rm model}\\,/\\,\\sigma_{\\rm true}$",
        "opt_b_tilt": "tilt {a}",
        "opt_b_fit": "what polyband actually returns",
        "opt_c_title": "The compensation never pays",
        "opt_c_x": "tilt applied to $\\ln\\sigma$",
        "opt_c_y": "excess $-\\ln L$ after refitting\neverything else",
        "opt_c_n": "N = {n} points",

        "og_frac": "{f} outliers, {s}$\\times$ wider",
        "og_scale": "10% outliers, {s}$\\times$ wider",
        "og_gauss": "Gaussian: {w}",
        "og_student": "Student-t nu=4: {w}",
        "og_true": "true width = {w}",
        "og_outliers": "outliers",
        "og_gauss_band": "Gaussian band",
        "og_student_band": "Student-t band",
        "og_offscale": "{n} points outside the frame",

        "os_a_title": "How deviant they are stops mattering",
        "os_a_x": "outlier width, in units of the true width",
        "os_b_title": "How many there are never stops mattering",
        "os_b_x": "fraction of outliers",
        "os_y": "fitted width / true width",
        "os_gauss": "Gaussian",
        "os_nu": "nu = {v}",
        "os_ok": "correct answer",
        "os_c_title": "What robustness costs on clean data",
        "os_c_x": "nu",
        "os_c_scale": "fitted width / true sd",
        "os_c_k": "band multiple giving 68.3% coverage",
    },
    "fr": {
        "data": "Données",
        "data_n": "Données (N = {n})",
        "trend": "Tendance",
        "true_width": "Largeur vraie à 1$\\sigma$",
        "band_1s": "Enveloppe à 1$\\sigma$",

        "bve_left_title": "Deux enveloppes différentes, un seul échantillon",
        "bve_band": f"Enveloppe à 1$\\sigma${NB}: dispersion des points",
        "bve_trend_err": "Erreur à 1$\\sigma$ sur la tendance elle-même",
        "bve_right_title": "Ajouter des données rétrécit l'une et pas l'autre",
        "bve_halfwidth": "demi-largeur de l'enveloppe à 1$\\sigma$",
        "bve_true_at": "largeur vraie à x = {x}",
        "bve_error": "erreur sur la tendance",
        "bve_ref": r"référence en $1/\sqrt{N}$",
        "bve_xlabel": "taille de l'échantillon N",
        "bve_ylabel": "demi-largeur à x = {x}",

        "robust_gauss": "Vraisemblance gaussienne (par défaut)",
        "robust_student": "Vraisemblance de Student, nu = 4",
        "robust_width": "largeur ajustée = {w} (vraie {t})",
        "robust_band": "enveloppe ajustée à 1$\\sigma$",
        "robust_true": "largeur vraie de la composante propre",

        "log_linear": f"Ajusté en y{NB}: l'enveloppe passe sous zéro,\n"
                      "la dispersion est asymétrique",
        "log_log": f"Ajusté avec log_y=True{NB}: l'enveloppe suit\n"
                   "la dispersion multiplicative",

        "yerr_data": "Données avec leurs barres d'erreur",
        "yerr_ignoring": "Enveloppe sans yerr  (largeur {w})",
        "yerr_intrinsic": "Enveloppe intrinsèque avec yerr  (largeur {w})",
        "yerr_true": "Largeur intrinsèque vraie ({t})",

        "bins_left": "Moyenne $\\pm$ écart-type par intervalle\n"
                     "en dents de scie, dépend de la largeur, "
                     "rien entre les centres",
        "bins_right": f"polyband{NB}: lisse et continu,\n"
                      "retrouve la largeur vraie (pointillés)",

        "diag_resid_title": "Résidus en fonction de x",
        "diag_resid_y": "résidu standardisé",
        "diag_dist_title": "Distribution des résidus",
        "diag_density": "densité",
        "diag_gaussian": "gaussienne réduite",
        "diag_qq_title": "Quantile-quantile",
        "diag_qq_x": "quantile théorique",
        "diag_qq_y": "quantile observé",
        "diag_cov_title": "Couverture de l'enveloppe",
        "diag_cov_x": "fraction attendue à l'intérieur",
        "diag_cov_y": "fraction observée à l'intérieur",

        "orders_under": "trop rigide",
        "orders_right": "correct",
        "orders_over": "trop souple",
        "orders_best": "meilleur",
        "orders_ylab_mean": "y\n\ndegré de la TENDANCE balayé\n(largeur maintenue correcte)",
        "orders_ylab_width": "y\n\ndegré de la LARGEUR balayé\n(tendance maintenue correcte)",

        "uf_rigid": f"order_width=0{NB}: trop rigide",
        "uf_correct": f"order_width=1{NB}: correct",
        "uf_inside": f"{{p}}{NB}% dans 1$\\sigma$ (68{NB}% attendu)",

        "opt_a_title": "Ce qu'un seul point coûte à l'ajustement",
        "opt_a_x": "$\\sigma_{\\rm modèle}\\,/\\,\\sigma_{\\rm vrai}$ "
                   "en ce point",
        "opt_a_y": "coût attendu en $-\\ln L$ (nats)",
        "opt_a_chi2": f"terme $\\frac{{1}}{{2}}\\,z^2${NB}: pénalise "
                      "une enveloppe trop étroite",
        "opt_a_logs": f"terme $\\ln\\sigma${NB}: pénalise "
                      "une enveloppe trop large",
        "opt_a_total": f"somme{NB}: un seul minimum, à la largeur vraie",
        "opt_b_title": "Une enveloppe qui sous-estime puis surestime,\n"
                       "en gardant la largeur moyenne juste",
        "opt_b_y": "$\\sigma_{\\rm modèle}\\,/\\,\\sigma_{\\rm vrai}$",
        "opt_b_tilt": "inclinaison {a}",
        "opt_b_fit": "ce que polyband renvoie réellement",
        "opt_c_title": "La compensation n'est jamais rentable",
        "opt_c_x": "inclinaison imposée à $\\ln\\sigma$",
        "opt_c_y": "excès de $-\\ln L$ après réajustement\nde tout le reste",
        "opt_c_n": "N = {n} points",

        "og_frac": f"{{f}} d'aberrants, {{s}}$\\times$ plus larges",
        "og_scale": f"10{NB}% d'aberrants, {{s}}$\\times$ plus larges",
        "og_gauss": f"Gaussienne{NB}: {{w}}",
        "og_student": f"Student nu=4{NB}: {{w}}",
        "og_true": "largeur vraie = {w}",
        "og_outliers": "aberrants",
        "og_gauss_band": "enveloppe gaussienne",
        "og_student_band": "enveloppe Student",
        "og_offscale": "{n} points hors cadre",

        "os_a_title": "Leur degré de déviance cesse de compter",
        "os_a_x": "largeur des aberrants, en unités de la largeur vraie",
        "os_b_title": "Leur nombre ne cesse jamais de compter",
        "os_b_x": "fraction d'aberrants",
        "os_y": "largeur ajustée / largeur vraie",
        "os_gauss": "Gaussienne",
        "os_nu": "nu = {v}",
        "os_ok": "bonne réponse",
        "os_c_title": "Ce que la robustesse coûte sur des données propres",
        "os_c_x": "nu",
        "os_c_scale": "largeur ajustée / écart-type vrai",
        "os_c_k": f"multiple de l'enveloppe donnant 68,3{NB}% de couverture",
    },
}

LANG = "en"
OUT = DOCS / "figures" / LANG


def T(key, **kw):
    """Look up a string in the current language, formatting any fields."""
    text = STRINGS[LANG][key]
    return text.format(**kw) if kw else text


def num(value, fmt="{:.2f}"):
    """Format a number with the decimal separator of the current language."""
    text = fmt.format(value)
    return text.replace(".", ",") if LANG == "fr" else text


def save(fig, name):
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.svg"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


def style(ax, xlabel="x", ylabel="y"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ----------------------------------------------------------------------
def fig_hero():
    """The headline picture: trend, nested bands, and the truth on top."""
    data = make_trumpet(n=450, seed=0)
    fit = fit_polyband(data.x, data.y, order_mean=2, order_width=1)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    art = plot_polyband(fit, data.x, data.y, ax=ax, nsigma=(1, 2),
                        color=ACCENT, point_color=MUTED,
                        point_kw=dict(s=22, alpha=0.45))

    # Relabel through the returned artists rather than re-plotting.
    art.points.set_label(T("data"))
    art.trend.set_label(T("trend"))
    for band, k in zip(art.bands, (2, 1)):
        band.set_label(T("band_1s").replace("1$", f"{k}$"))

    grid = fit.grid()
    for sign in (-1, 1):
        ax.plot(grid, data.true_mean(grid) + sign * data.true_sigma(grid),
                color=TRUTH, linestyle="--", linewidth=1.4,
                label=T("true_width") if sign > 0 else "_nolegend_")

    style(ax)
    ax.legend(handles=art.legend_handles + [ax.lines[-1]], fontsize=9,
              loc="upper left")
    save(fig, "hero")
    return fit


# ----------------------------------------------------------------------
def fig_band_vs_error():
    """The distinction the whole package is built around.

    Left: both quantities drawn on one sample. Right: how each behaves as the
    sample grows, which is the cleanest way to show they are not the same
    thing. The trend error follows the 1/sqrt(N) reference line; the band
    settles on the true width and stays there.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    data = make_trumpet(n=400, seed=4)
    fit = fit_polyband(data.x, data.y, 2, 1)
    grid = fit.grid()
    ax = axes[0]
    ax.scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0,
               label=T("data_n", n=400))
    lo, hi = fit.envelope(grid)
    ax.fill_between(grid, lo, hi, color=ACCENT, alpha=0.18, linewidth=0,
                    label=T("bve_band"))
    tlo, thi = fit.trend_band(grid)
    ax.fill_between(grid, tlo, thi, color=WARN, alpha=0.75, linewidth=0,
                    label=T("bve_trend_err"))
    ax.plot(grid, fit.predict(grid), color=ACCENT, linewidth=2.2,
            label=T("trend"))
    ax.set_title(T("bve_left_title"), fontsize=10.5)
    style(ax)
    ax.legend(fontsize=8.5, loc="upper left")

    # Right panel: behaviour with sample size, at mid-range x.
    x0 = 5.0
    sizes = np.unique(np.round(np.logspace(np.log10(30), np.log10(20000), 14)
                               ).astype(int))
    widths, errors = [], []
    for n in sizes:
        d = make_trumpet(n=int(n), seed=100 + int(n))
        f = fit_polyband(d.x, d.y, 2, 1)
        widths.append(float(f.scatter(x0)))
        errors.append(float(f.mean_error(x0)))

    ax = axes[1]
    truth = float(data.true_sigma(np.array([x0]))[0])
    ax.plot(sizes, widths, "o-", color=ACCENT, linewidth=2.0, markersize=5,
            label=T("bve_halfwidth"))
    ax.axhline(truth, color=TRUTH, linestyle="--", linewidth=1.4,
               label=T("bve_true_at", x=num(x0, "{:g}")))
    ax.plot(sizes, errors, "s-", color=WARN, linewidth=2.0, markersize=4.5,
            label=T("bve_error"))
    ref = np.asarray(errors, dtype=float)[0] * np.sqrt(sizes[0] / sizes)
    ax.plot(sizes, ref, color=WARN, linestyle=":", linewidth=1.3,
            label=T("bve_ref"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(T("bve_right_title"), fontsize=10.5)
    style(ax, xlabel=T("bve_xlabel"),
          ylabel=T("bve_ylabel", x=num(x0, "{:g}")))
    ax.legend(fontsize=8.5, loc="lower left")
    save(fig, "band_vs_error")


# ----------------------------------------------------------------------
def fig_orders():
    """What each degree controls, swept from too rigid to too flexible.

    One degree is varied per row while the other is held at the value the data
    were generated with, so each row is a clean one-dimensional sweep. Both
    rows run past the correct answer on purpose: the interesting failure is
    not only the band that cannot bend, it is also the band that bends to
    follow noise, and BIC turns back up in both directions.
    """
    data = make_trumpet(n=350, seed=0)
    # (which degree is swept, the value the data were generated with, combos)
    rows = [
        ("mean", 2, [(0, 1), (1, 1), (2, 1), (9, 1)]),
        ("width", 1, [(2, 0), (2, 1), (2, 2), (2, 8)]),
    ]
    best = fit_polyband(data.x, data.y, 2, 1).bic

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.0), sharex=True, sharey=True)
    for r, (row_key, true_degree, combos) in enumerate(rows):
        for c, (om, ow) in enumerate(combos):
            ax = axes[r, c]
            fit = fit_polyband(data.x, data.y, order_mean=om, order_width=ow)
            grid = fit.grid()
            ax.scatter(data.x, data.y, s=12, alpha=0.35, color=MUTED, linewidths=0)
            lo, hi = fit.envelope(grid)

            varied = om if r == 0 else ow
            if varied < true_degree:
                status, colour = T("orders_under"), WARN
            elif varied == true_degree:
                status, colour = T("orders_right"), ACCENT
            else:
                status, colour = T("orders_over"), VIOLET

            ax.fill_between(grid, lo, hi, color=colour, alpha=0.20, linewidth=0)
            ax.plot(grid, fit.predict(grid), color=colour, linewidth=2.0)
            ax.plot(grid, data.true_mean(grid), color=TRUTH, linestyle="--",
                    linewidth=1.2)
            for sign in (-1, 1):
                ax.plot(grid, data.true_mean(grid)
                        + sign * data.true_sigma(grid),
                        color=TRUTH, linestyle=":", linewidth=1.0)

            delta = fit.bic - best
            tag = (T("orders_best") if abs(delta) < 0.05
                   else f"{delta:+.0f}".replace("+", "+"))
            # Parameter names are code, so they stay in English in both versions.
            ax.set_title(f"order_mean={om}, order_width={ow}\n"
                         f"{status}   BIC={fit.bic:.0f}  ({tag})",
                         fontsize=9.0, color=colour)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    axes[0, 0].set_ylim(-2.0, 14.0)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    axes[0, 0].set_ylabel(T("orders_ylab_mean"), fontsize=9.5)
    axes[1, 0].set_ylabel(T("orders_ylab_width"), fontsize=9.5)
    save(fig, "orders")


# ----------------------------------------------------------------------
def fig_robust():
    """Gaussian against Student-t on a contaminated sample."""
    data = make_heavy_tails(n=600, seed=1, contamination=0.09)
    gauss = fit_polyband(data.x, data.y, 2, 0)
    robust = fit_polyband(data.x, data.y, 2, 0, nu=4)
    truth = float(data.true_sigma(np.array([5.0]))[0])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, fit, name in ((axes[0], gauss, T("robust_gauss")),
                          (axes[1], robust, T("robust_student"))):
        grid = fit.grid()
        ax.scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0)
        lo, hi = fit.envelope(grid)
        ax.fill_between(grid, lo, hi, color=ACCENT, alpha=0.20, linewidth=0,
                        label=T("robust_band"))
        ax.plot(grid, fit.predict(grid), color=ACCENT, linewidth=2.1)
        for sign in (-1, 1):
            ax.plot(grid, data.true_mean(grid) + sign * data.true_sigma(grid),
                    color=TRUTH, linestyle="--", linewidth=1.4,
                    label=T("robust_true") if sign > 0 else "_nolegend_")
        width = T("robust_width", w=num(float(fit.scatter(5.0))),
                  t=num(truth))
        ax.set_title(f"{name}\n{width}", fontsize=10)
        style(ax, ylabel="y" if fit is gauss else "")
    axes[0].legend(fontsize=8.5, loc="upper left")
    save(fig, "robust")


# ----------------------------------------------------------------------
def fig_logspace():
    """Why a multiplicative quantity has to be fitted in log space."""
    data = make_decades(n=400, seed=2)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    linear = fit_polyband(data.x, data.y, 2, 1)
    grid = linear.grid()
    axes[0].scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0)
    lo, hi = linear.envelope(grid)
    axes[0].fill_between(grid, lo, hi, color=WARN, alpha=0.20, linewidth=0)
    axes[0].plot(grid, linear.predict(grid), color=WARN, linewidth=2.1)
    axes[0].axhline(0, color=LINE, linewidth=1.0)
    axes[0].set_title(T("log_linear"), fontsize=10)
    style(axes[0])

    logfit = fit_polyband(data.x, data.y, 2, 1, log_y=True)
    grid = logfit.grid()
    axes[1].scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0)
    lo, hi = logfit.envelope(grid)
    axes[1].fill_between(grid, lo, hi, color=ACCENT, alpha=0.20, linewidth=0)
    axes[1].plot(grid, logfit.predict(grid), color=ACCENT, linewidth=2.1)
    axes[1].set_yscale("log")
    axes[1].set_title(T("log_log"), fontsize=10)
    style(axes[1])
    save(fig, "logspace")


# ----------------------------------------------------------------------
def fig_yerr():
    """Separating intrinsic scatter from measurement noise."""
    data = make_with_errors(n=400, seed=3)
    naive = fit_polyband(data.x, data.y, 1, 0)
    aware = fit_polyband(data.x, data.y, 1, 0, yerr=data.yerr)
    truth = float(data.true_sigma(np.array([5.0]))[0])

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    grid = naive.grid()
    ax.errorbar(data.x, data.y, yerr=data.yerr, fmt="o", markersize=3,
                color=MUTED, alpha=0.45, elinewidth=0.8, capsize=0,
                linestyle="none", label=T("yerr_data"))

    lo, hi = naive.envelope(grid)
    ax.fill_between(grid, lo, hi, color=WARN, alpha=0.16, linewidth=0,
                    label=T("yerr_ignoring", w=num(float(naive.scatter(5.0)))))
    lo, hi = aware.envelope(grid)
    ax.fill_between(grid, lo, hi, color=ACCENT, alpha=0.26, linewidth=0,
                    label=T("yerr_intrinsic", w=num(float(aware.scatter(5.0)))))
    ax.plot(grid, aware.predict(grid), color=ACCENT, linewidth=2.1,
            label=T("trend"))
    for sign in (-1, 1):
        ax.plot(grid, data.true_mean(grid) + sign * data.true_sigma(grid),
                color=TRUTH, linestyle="--", linewidth=1.4,
                label=T("yerr_true", t=num(truth)) if sign > 0 else "_nolegend_")
    style(ax)
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "yerr")


# ----------------------------------------------------------------------
def fig_bins():
    """The binned alternative, and why a continuous fit beats it."""
    data = make_trumpet(n=220, seed=8)
    edges = np.arange(0, 11, 1.0)
    centres, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (data.x >= lo) & (data.x < hi)
        if sel.sum() > 1:
            centres.append(0.5 * (lo + hi))
            means.append(float(np.mean(data.y[sel])))
            stds.append(float(np.std(data.y[sel], ddof=1)))
    centres = np.array(centres)
    means = np.array(means)
    stds = np.array(stds)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)

    axes[0].scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0)
    axes[0].fill_between(centres, means - stds, means + stds, color=WARN,
                         alpha=0.20, linewidth=0, step="mid")
    axes[0].plot(centres, means, "o-", color=WARN, linewidth=1.8, markersize=5)
    axes[0].set_title(T("bins_left"), fontsize=10)
    style(axes[0])

    fit = fit_polyband(data.x, data.y, 2, 1)
    grid = fit.grid()
    axes[1].scatter(data.x, data.y, s=16, alpha=0.4, color=MUTED, linewidths=0)
    lo, hi = fit.envelope(grid)
    axes[1].fill_between(grid, lo, hi, color=ACCENT, alpha=0.20, linewidth=0)
    axes[1].plot(grid, fit.predict(grid), color=ACCENT, linewidth=2.1)
    for sign in (-1, 1):
        axes[1].plot(grid, data.true_mean(grid) + sign * data.true_sigma(grid),
                     color=TRUTH, linestyle="--", linewidth=1.3)
        axes[0].plot(grid, data.true_mean(grid) + sign * data.true_sigma(grid),
                     color=TRUTH, linestyle="--", linewidth=1.3)
    axes[1].set_title(T("bins_right"), fontsize=10)
    style(axes[1], ylabel="")
    save(fig, "bins_vs_polyband")


# ----------------------------------------------------------------------
def fig_diagnostics():
    """The four-panel calibration check.

    plot_diagnostics labels its panels in English, the way the rest of the
    library does. For the French figure the labels are rewritten afterwards
    rather than adding translations to the package itself.
    """
    data = make_trumpet(n=800, seed=5)
    fit = fit_polyband(data.x, data.y, 2, 1)
    fig = plot_diagnostics(fit, data.x, data.y, color=ACCENT)

    ax1, ax2, ax3, ax4 = fig.axes[:4]
    ax1.set_title(T("diag_resid_title"), fontsize=10)
    ax1.set_ylabel(T("diag_resid_y"))
    ax2.set_title(T("diag_dist_title"), fontsize=10)
    ax2.set_xlabel(T("diag_resid_y"))
    ax2.set_ylabel(T("diag_density"))
    for line in ax2.get_lines():
        if line.get_label() and not line.get_label().startswith("_"):
            line.set_label(T("diag_gaussian"))
    ax2.legend(fontsize=8)
    ax3.set_title(T("diag_qq_title"), fontsize=10)
    ax3.set_xlabel(T("diag_qq_x"))
    ax3.set_ylabel(T("diag_qq_y"))
    ax4.set_title(T("diag_cov_title"), fontsize=10)
    ax4.set_xlabel(T("diag_cov_x"))
    ax4.set_ylabel(T("diag_cov_y"))

    for ax in fig.axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    save(fig, "diagnostics")


# ----------------------------------------------------------------------
def fig_underfit_width():
    """What a too-rigid width polynomial does to the residuals."""
    data = make_trumpet(n=900, seed=6)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)

    for ax, ow, title in ((axes[0], 0, T("uf_rigid")),
                          (axes[1], 1, T("uf_correct"))):
        fit = fit_polyband(data.x, data.y, 2, ow)
        z = fit.zscore(data.x, data.y)
        ax.scatter(data.x, z, s=14, alpha=0.35, color=MUTED, linewidths=0)
        for k, ls in ((1, "--"), (2, ":")):
            for sign in (-1, 1):
                ax.axhline(sign * k, color=ACCENT, linestyle=ls, linewidth=1.2)
        ax.axhline(0, color=ACCENT, linewidth=1.5)
        inside = float(np.mean(np.abs(z) < 1)) * 100
        ax.set_title(f"{title}\n{T('uf_inside', p=f'{inside:.0f}')}",
                     fontsize=10)
        style(ax, ylabel=T("diag_resid_y") if ow == 0 else "")
    save(fig, "underfit_width")


# ----------------------------------------------------------------------
VIOLET = "#c77dff"


def _tilt_profile(n, seed, tilts):
    """Profile likelihood along a forced tilt of the width polynomial.

    The tilt is added to the coefficient of ``t`` in ``ln sigma``, which makes
    the band too narrow at one end of the x range and too wide at the other.
    Every other parameter, the whole trend and the constant term of the width,
    is then re-optimised: the fit is given every chance to buy the tilt back.
    The excess is what it still cannot recover.
    """
    data = make_trumpet(n=n, seed=seed)
    fit = fit_polyband(data.x, data.y, 2, 1, dof_correction=False)
    t = (data.x - fit.x_offset) / fit.x_scale
    design_mean, design_width = np.vander(t, 3), np.vander(t, 2)
    p_opt = np.concatenate([fit.coeff_mean, fit.coeff_width])
    args = (design_mean, design_width, data.y, None, None)
    nll0 = _neg_log_like(p_opt, *args)
    slope = fit.coeff_width[0]

    excess = []
    for a in tilts:
        def objective(free, a=a):
            return _neg_log_like(
                np.array([free[0], free[1], free[2], slope + a, free[3]]), *args
            )
        res = minimize(objective, p_opt[[0, 1, 2, 4]], method="Nelder-Mead",
                       options={"maxiter": 20000, "maxfev": 20000,
                                "xatol": 1e-10, "fatol": 1e-10})
        excess.append(float(res.fun) - nll0)
    return np.array(excess)


def fig_optimisation():
    """Why an under-estimate here cannot be paid for by an over-estimate there.

    Left: the per-point cost of a wrong width, split into its two competing
    terms. Middle: what a compensating error would look like. Right: the
    profile likelihood along that compensation, which has a single minimum at
    zero and rises on both sides.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    # --- (a) the tug-of-war between the two terms of the likelihood -----
    ax = axes[0]
    f = np.logspace(np.log10(0.45), np.log10(2.2), 400)
    chi2 = 0.5 / f ** 2          # expected value of z^2 / 2 when sigma is f x too big
    logs = np.log(f)
    ax.plot(f, chi2 - 0.5, color=WARN, linestyle="--", linewidth=1.6,
            label=T("opt_a_chi2"))
    ax.plot(f, logs, color=VIOLET, linestyle="--", linewidth=1.6,
            label=T("opt_a_logs"))
    ax.plot(f, chi2 - 0.5 + logs, color=ACCENT, linewidth=2.4,
            label=T("opt_a_total"))
    ax.axvline(1.0, color=TRUTH, linestyle=":", linewidth=1.3)
    ax.axhline(0.0, color=LINE, linewidth=1.0)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0])
    ax.set_xticklabels([num(v, "{:g}") for v in (0.5, 0.7, 1.0, 1.5, 2.0)])
    ax.set_ylim(-1.3, 1.15)
    ax.set_title(T("opt_a_title"), fontsize=10.5)
    style(ax, xlabel=T("opt_a_x"), ylabel=T("opt_a_y"))
    ax.legend(fontsize=8, loc="lower center")

    # --- (b) what a compensating error looks like ----------------------
    ax = axes[1]
    data = make_trumpet(n=5000, seed=6)
    fit = fit_polyband(data.x, data.y, 2, 1)
    grid = fit.grid()
    t = (grid - fit.x_offset) / fit.x_scale
    for a, colour in ((-0.2, WARN), (0.2, VIOLET)):
        ax.plot(grid, np.exp(a * t), color=colour, linewidth=2.0,
                label=T("opt_b_tilt", a=num(a, "{:+.1f}")))
    ax.plot(grid, fit.scatter(grid) / data.true_sigma(grid), color=ACCENT,
            linewidth=2.2, label=T("opt_b_fit"))
    ax.axhline(1.0, color=TRUTH, linestyle="--", linewidth=1.4)
    ax.set_ylim(0.72, 1.38)
    ax.set_title(T("opt_b_title"), fontsize=10.5)
    style(ax, ylabel=T("opt_b_y"))
    ax.legend(fontsize=8.5, loc="upper left")

    # --- (c) the price of that compensation ----------------------------
    ax = axes[2]
    tilts = np.linspace(-0.35, 0.35, 29)
    for n, seed, colour, marker in ((450, 0, ACCENT, "o"), (1800, 11, MUTED, "s")):
        excess = _tilt_profile(n, seed, tilts)
        ax.plot(tilts, excess, color=colour, linewidth=2.0,
                label=T("opt_c_n", n=n))
        for a, spot in ((-0.2, WARN), (0.2, VIOLET)):
            if n == 450:
                value = float(np.interp(a, tilts, excess))
                ax.plot([a], [value], marker=marker, color=spot, markersize=8,
                        zorder=5)
    ax.axvline(0.0, color=TRUTH, linestyle=":", linewidth=1.3)
    ax.set_ylim(bottom=0)
    ax.set_title(T("opt_c_title"), fontsize=10.5)
    style(ax, xlabel=T("opt_c_x"), ylabel=T("opt_c_y"))
    ax.legend(fontsize=8.5, loc="upper center")

    save(fig, "optimisation")


# ----------------------------------------------------------------------
# Robustness: the two axes along which outliers can hurt.
# ----------------------------------------------------------------------
OUT_N = 500          # points per realisation in the scans
OUT_TRIALS = 25      # realisations per grid node
OUT_SIGMA = 0.6      # true width of the clean component
_SCAN_CACHE = {}


def _contaminated(seed, frac, scale, n=OUT_N):
    """Clean quadratic trend of constant width, with a fraction of outliers."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    mean = 2.0 + 0.9 * x - 0.06 * x ** 2
    bad = rng.random(n) < frac
    y = mean + rng.normal(0.0, OUT_SIGMA * np.where(bad, scale, 1.0))
    return x, y, bad


def _one_scan_fit(job):
    seed, frac, scale, nu = job
    x, y, _ = _contaminated(seed, frac, scale)
    try:
        fit = fit_polyband(x, y, 2, 0, nu=nu)
        return float(fit.scatter(5.0)) / OUT_SIGMA
    except Exception:                     # pragma: no cover - defensive
        return np.nan


def _scan(fracs, scales, nus):
    """Median fitted width over realisations, for every (frac, scale, nu).

    Cached, because the figures are built once per site language and the
    numbers do not depend on the language.
    """
    key = (tuple(fracs), tuple(scales), tuple(str(n) for n in nus))
    if key in _SCAN_CACHE:
        return _SCAN_CACHE[key]

    jobs, nodes = [], []
    for frac in fracs:
        for scale in scales:
            for nu in nus:
                nodes.append((frac, scale, nu))
                jobs += [(7000 + i, frac, scale, nu) for i in range(OUT_TRIALS)]
    with Pool() as pool:
        vals = pool.map(_one_scan_fit, jobs)
    vals = np.asarray(vals).reshape(len(nodes), OUT_TRIALS)
    out = {node: float(np.nanmedian(v)) for node, v in zip(nodes, vals)}
    _SCAN_CACHE[key] = out
    return out


def _one_clean_fit(job):
    """Fitted width and the band multiple giving 68.3% coverage, clean data."""
    seed, nu = job
    x, y, _ = _contaminated(seed, 0.0, 1.0)
    fit = fit_polyband(x, y, 2, 0, nu=nu)
    z = np.abs(fit.zscore(x, y))
    return float(fit.scatter(5.0)) / OUT_SIGMA, float(np.quantile(z, 0.683))


def _clean_calibration(nus):
    key = ("clean", tuple(str(n) for n in nus))
    if key in _SCAN_CACHE:
        return _SCAN_CACHE[key]
    jobs = [(9000 + i, nu) for nu in nus for i in range(OUT_TRIALS)]
    with Pool() as pool:
        vals = pool.map(_one_clean_fit, jobs)
    vals = np.asarray(vals).reshape(len(nus), OUT_TRIALS, 2)
    out = {nu: (float(np.median(v[:, 0])), float(np.median(v[:, 1])))
           for nu, v in zip(nus, vals)}
    _SCAN_CACHE[key] = out
    return out


def fig_outlier_gallery():
    """The two axes of contamination, shown on the data themselves.

    Top row: more and more outliers, each 6 times wider than the core. Bottom
    row: always 10% of the sample, but each one further out. The Gaussian band
    tracks the contamination in both directions; the Student-t band stops
    noticing along the bottom row entirely.
    """
    cases_frac = [(0.0, 6.0), (0.05, 6.0), (0.15, 6.0), (0.30, 6.0)]
    cases_scale = [(0.10, 3.0), (0.10, 10.0), (0.10, 100.0), (0.10, 1000.0)]

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6), sharex=True, sharey=True)
    for row, cases in enumerate((cases_frac, cases_scale)):
        for col, (frac, scale) in enumerate(cases):
            ax = axes[row, col]
            x, y, bad = _contaminated(100 + row * 10 + col, frac, scale)
            gauss = fit_polyband(x, y, 2, 0)
            robust = fit_polyband(x, y, 2, 0, nu=4.0)
            grid = gauss.grid()

            ax.scatter(x[~bad], y[~bad], s=11, alpha=0.40, color=MUTED,
                       linewidths=0, label=T("data"))
            ax.scatter(x[bad], y[bad], s=16, alpha=0.75, color=WARN,
                       linewidths=0, label=T("og_outliers"))
            for fit, colour, key in ((gauss, WARN, "og_gauss_band"),
                                     (robust, ACCENT, "og_student_band")):
                lo, hi = fit.envelope(grid)
                ax.fill_between(grid, lo, hi, color=colour, alpha=0.18,
                                linewidth=0)
                ax.plot(grid, lo, color=colour, linewidth=1.4, label=T(key))
                ax.plot(grid, hi, color=colour, linewidth=1.4)
            truth = 2.0 + 0.9 * grid - 0.06 * grid ** 2
            for sign in (-1, 1):
                ax.plot(grid, truth + sign * OUT_SIGMA, color=TRUTH,
                        linestyle="--", linewidth=1.2,
                        label=T("og_true", w=num(OUT_SIGMA)) if sign > 0
                        else "_nolegend_")

            if row == 0:
                head = T("og_frac", f=f"{frac:.0%}".replace("%", NB + "%")
                         if LANG == "fr" else f"{frac:.0%}",
                         s=f"{scale:.0f}")
            else:
                head = T("og_scale", s=f"{scale:.0f}")
            ax.set_title(
                f"{head}\n"
                f"{T('og_gauss', w=num(float(gauss.scatter(5.0))))}   "
                f"{T('og_student', w=num(float(robust.scatter(5.0))))}",
                fontsize=8.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    axes[0, 0].legend(fontsize=7.5, loc="lower right", framealpha=0.92)

    # A fixed y window on the clean component, so the panels are comparable
    # and the extreme outliers are honestly reported as being off the frame.
    axes[0, 0].set_ylim(-2.5, 9.0)
    for row in range(2):
        for col in range(4):
            ax = axes[row, col]
            x, y, _ = _contaminated(100 + row * 10 + col,
                                    (cases_frac if row == 0 else cases_scale)[col][0],
                                    (cases_frac if row == 0 else cases_scale)[col][1])
            off = int(np.sum((y < -2.5) | (y > 9.0)))
            if off:
                ax.text(0.5, 0.02, T("og_offscale", n=off), transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=8, color=WARN)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    save(fig, "outliers_gallery")


def fig_outlier_scan():
    """The same two axes, measured rather than eyeballed."""
    nus = [None, 3.0, 4.0, 7.0]
    colours = {None: WARN, 3.0: TRUTH, 4.0: ACCENT, 7.0: VIOLET}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # --- (a) deviance axis, fraction held at 10% ------------------------
    scales = [1.0, 2.0, 3.0, 6.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    table = _scan([0.10], scales, nus)
    ax = axes[0]
    for nu in nus:
        ax.plot(scales, [table[(0.10, s, nu)] for s in scales], "o-",
                color=colours[nu], linewidth=2.0, markersize=4,
                label=T("os_gauss") if nu is None else T("os_nu", v=f"{nu:g}"))
    ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=1.3,
               label=T("os_ok"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_yticks([1, 3, 10, 30, 100, 300])
    ax.set_yticklabels(["1", "3", "10", "30", "100", "300"])
    ax.set_title(T("os_a_title"), fontsize=10.5)
    style(ax, xlabel=T("os_a_x"), ylabel=T("os_y"))
    ax.legend(fontsize=8, loc="upper left")

    # --- (b) fraction axis, deviance held at 10x ------------------------
    fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    table = _scan(fracs, [10.0], nus)
    ax = axes[1]
    for nu in nus:
        ax.plot(fracs, [table[(f, 10.0, nu)] for f in fracs], "o-",
                color=colours[nu], linewidth=2.0, markersize=4,
                label=T("os_gauss") if nu is None else T("os_nu", v=f"{nu:g}"))
    ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=1.3)
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_yticks([0.8, 1, 1.5, 2, 3, 5, 7])
    ax.set_yticklabels([num(v, "{:g}") for v in (0.8, 1, 1.5, 2, 3, 5, 7)])
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels([f"{v:.0%}" for v in (0, 0.1, 0.2, 0.3, 0.4)])
    ax.set_title(T("os_b_title"), fontsize=10.5)
    style(ax, xlabel=T("os_b_x"), ylabel="")
    ax.legend(fontsize=8, loc="upper left")

    # --- (c) the bill, on data with no outliers at all ------------------
    cal_nus = [2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0]
    cal = _clean_calibration(cal_nus)
    ax = axes[2]
    ax.plot(cal_nus, [cal[n][0] for n in cal_nus], "o-", color=ACCENT,
            linewidth=2.0, markersize=5, label=T("os_c_scale"))
    ax.plot(cal_nus, [cal[n][1] for n in cal_nus], "s-", color=WARN,
            linewidth=2.0, markersize=4.5, label=T("os_c_k"))
    ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=1.3)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(cal_nus)
    ax.set_xticklabels([f"{v:g}" for v in cal_nus])
    ax.set_title(T("os_c_title"), fontsize=10.5)
    style(ax, xlabel=T("os_c_x"), ylabel="")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "outliers_scan")


def build(lang):
    global LANG, OUT
    LANG = lang
    OUT = DOCS / "figures" / lang
    print(f"\n[{lang}]")
    fit = fig_hero()
    fig_band_vs_error()
    fig_orders()
    fig_robust()
    fig_logspace()
    fig_yerr()
    fig_bins()
    fig_diagnostics()
    fig_underfit_width()
    fig_optimisation()
    fig_outlier_gallery()
    fig_outlier_scan()
    return fit


if __name__ == "__main__":
    print("Generating polyband documentation figures")
    fit = build("en")
    build("fr")

    print("\nHero fit summary, quoted on the page:")
    print(fit.summary())
    data = make_trumpet(n=450, seed=0)
    best, table = select_orders(data.x, data.y, 4, 2)
    print("\nBIC ranking, top 5:")
    for om, ow, val in table[:5]:
        print(f"  order_mean={om} order_width={ow}  BIC={val:.1f}")
    print("\nCoverage of the hero fit:")
    for k, obs, exp in fit.coverage(data.x, data.y):
        print(f"  {k:.0f} sigma: {obs:.3f} observed vs {exp:.3f} expected")
