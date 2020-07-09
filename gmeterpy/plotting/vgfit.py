
import itertools
import numpy as np
import matplotlib.pylab as plt
from gmeterpy.corrections.vgrad import poly_uncertainty_eval

def plot_ties(df, gp, errorbar=False, xerr=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    markers = itertools.cycle(('*', 's', '^', 'p', 'o', 'D'))
    colors = itertools.cycle(('g', 'r', 'm', 'c', 'y'))

    # plot ties
    for eqn, group in df.groupby('ci'):
        ga = gp(group.h) + group.resid
        color = next(colors)
        ax.plot(ga, group.h, '--', marker=next(markers),
                color=color, markersize=10, alpha=0.8, linewidth=1.0)
        if errorbar:
            ax.errorbar(ga, group.h, xerr=xerr, color=color)

    return ax

def _plot_common(ax=None):
    if ax is None:
        fig = plt.figure(figsize=(6, 7))
        ax = fig.gca()

    ax.set_ylabel('Height ($m$)', fontsize=12)
    ax.set_xlabel('Gravity ($\mu Gal$)', fontsize=12)
    ax.set_xlim((-10, 10))
    ax.set_ylim((0, 1.4))

    return fig


def plot_fit(df, gp, gu, station, h_ref=0.710):
    # common
    fig = _plot_common()
    ax = fig.gca()

    # plot source data
    plot_ties(df, gp, ax=ax)

    # plot curve
    h = np.linspace(0.0001, 1.4, 100)
    ax.plot(gp(h), h, 'b-', linewidth=2.0)

    # 1-sigma diff with h_ref height
    h_ref = h_ref
    u = poly_uncertainty_eval(h, np.ones_like(h) * h_ref, *gu)
    ci_l = gp(h) - u
    ci_u = gp(h) + u
    ax.plot(ci_l, h, 'b', ci_u, h, 'b', linestyle='dashed')
    ax.fill_betweenx(h, ci_l, ci_u, alpha=0.05, color='0.05')

    return fig

