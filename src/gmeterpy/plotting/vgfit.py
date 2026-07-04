import itertools

import matplotlib.pylab as plt
import numpy as np

from gmeterpy.corrections.vgrad import polynomial_vgg_correction_uncertainty


def plot_ties(df, gp, errorbar=False, xerr=None, ax=None):
    if ax is None:
        _fig, ax = plt.subplots()

    markers = itertools.cycle(("*", "s", "^", "p", "o", "D"))
    colors = itertools.cycle(("g", "r", "m", "c", "y"))

    # plot ties
    for _eqn, group in df.groupby("ci"):
        ga = gp(group.h) + group.resid
        color = next(colors)
        ax.plot(
            ga,
            group.h,
            "--",
            marker=next(markers),
            color=color,
            markersize=10,
            alpha=0.8,
            linewidth=1.0,
        )
        if errorbar:
            ax.errorbar(ga, group.h, xerr=xerr, color=color)

    return ax


def _plot_common(ax=None):
    if ax is None:
        fig = plt.figure(figsize=(6, 7))
        ax = fig.gca()

    ax.set_ylabel("Height ($m$)", fontsize=12)
    ax.set_xlabel(r"Gravity ($\mu Gal$)", fontsize=12)
    ax.set_xlim((-10, 10))
    ax.set_ylim((0, 1.4))

    return fig


def plot_fit(df, gp, cov_params, station, h_ref=0.710):

    h_min = min(df.h)
    al = (gp(h_min) - gp(1.0)) / (h_min - 1.0)
    gpr = lambda x: gp(x) - al * x

    # common
    fig = _plot_common()
    ax = fig.gca()

    # plot source data
    plot_ties(df, gpr, ax=ax)

    # plot curve
    h = np.linspace(0.0001, 1.4, 100)
    ax.plot(gpr(h), h, "b-", linewidth=2.0)

    # 1-sigma diff with h_ref height
    u = polynomial_vgg_correction_uncertainty(h, np.ones_like(h) * h_ref, cov_params)
    ci_l = gpr(h) - u
    ci_u = gpr(h) + u
    ax.plot(ci_l, h, "b", ci_u, h, "b", linestyle="dashed")
    ax.fill_betweenx(h, ci_l, ci_u, alpha=0.05, color="0.05")

    title = f"{station}"
    subtitle = rf"degree={gp.order}, ${al:.1f}\,\mu$Gal / m substructed, $u_{{k = 1}}$"
    fig.suptitle(title, fontsize=14, y=0.95)
    fig.gca().set_title(subtitle, fontsize=12)

    return fig
