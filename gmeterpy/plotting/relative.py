#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import matplotlib.dates as dates
import matplotlib.ticker as ticker

from gmeterpy.stats import rms


def autocorrelation_plot(series, ax=None, **kwds):
    """Autocorrelation plot for time series.
    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    kwds : keywords
        Options to pass to matplotlib plotting method
    Returns:
    -----------
    ax: Matplotlib axis object
    """
    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca(xlim=(1, n), ylim=(-1.0, 1.0))
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
    x = series.index.to_pydatetime()
    y = [r(xi) for xi in np.arange(n) + 1]
    z95 = 1.959963984540054
    z99 = 2.5758293035489004

    repeat = lambda k: np.repeat(k, len(x))

    y95 = repeat(z99 / np.sqrt(n))
    y99 = repeat(z95 / np.sqrt(n))

    ax.plot(x, y95, color='grey')
    ax.plot(x, y99, linestyle='--', color='grey')
    ax.axhline(y=0.0, color='black')
    ax.plot(x, -y99, linestyle='--', color='grey')
    ax.plot(x, -y95, color='grey')
    ax.set_ylim([-1, 1])
    ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    return ax


def plot_error(series, ax=None, **kwds):
    if ax is None:
        fig, ax = plt.subplots()

    ax.vlines(series.index.to_pydatetime(), [0], series.values, linestyle='solid')
    ax.yaxis.set_major_locator(ticker.LinearLocator(5))
    ax.set_ylabel('Error ($\mu$Gal)')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.grid()
    ax.set_ylim([0, series.max()])

    return ax


def plot_temperature(in_temp, out_temp=None, ax=None, **kwads):
    if ax is None:
        fig, ax = plt.subplots()

    lns1 = ax.plot_date(in_temp.index.to_pydatetime(), in_temp.values, 'k.', label='in_temp')
    ax.yaxis.set_major_locator(ticker.LinearLocator(3))
    ax.set_ylabel('Inside temp. (mK)')

    if out_temp is not None:
        ax2 = ax.twinx()
        lns2 = ax2.plot_date(out_temp.index.to_pydatetime(), out_temp.values, 'kx', label='out_temp')
        ax2.yaxis.set_major_locator(ticker.LinearLocator(3))
        ax2.set_ylabel('Outside temp. (C$^o$)')

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        legend = ax2.legend(lns, labs, loc=2, ncol=2, fontsize=11)
        legend.get_frame().set_facecolor('w')
    else:
        legend = ax.legend(loc=2, ncol=1, fontsize=11)

    ax.grid()
    legend.get_frame().set_facecolor('w')

    return ax


def plot_tilts(tilt_y, tilt_x, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    lns1 = ax.plot_date(tilt_x.index.to_pydatetime(), tilt_x.values, 'k.',
                        label='tilt_x')
    lns2 = ax.plot_date(tilt_y.index.to_pydatetime(), tilt_y.values, 'kx',
                        label='tilt_y')

    max_tilt = max([abs(tilt_x).max(), abs(tilt_y).max()])

    if max_tilt <= 20:
        ax.axhline(y=10, color='k', linestyle='--')
        ax.axhline(y=-10, color='k', linestyle='--')
        ax.set_ylim([-20, 20])
    else:
        ax.axhline(y=20, linewidth=2, color='r')
        ax.axhline(y=-20, linewidth=2, color='r')

    ax.yaxis.set_major_locator(ticker.FixedLocator(range(-25, 25, 5),
                                                   nbins=5))

    ax.set_ylabel('Tilt [arcsec]')

    ax.grid()
    legend = ax.legend(loc=2, ncol=2, fontsize=11)
    legend.get_frame().set_facecolor('w')

    return ax


def plot_setup(data, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    c = pd.Categorical(data.name)

    for i, group in data.groupby('setup'):
        xm = group.index.mean()
        yi = c.categories.get_indexer(group.name.unique()) + 1
        xb, xe = group.index[[0, -1]].to_pydatetime()

        ax.plot([xb, xe], [yi, yi], 'k', [xb, xe], [yi, yi], 'k|')
        ax.plot([xb, xb], [0, yi], 'k-', [xe, xe], [0, yi], 'k-')
        ax.text(xm, yi, int(i), bbox=dict(facecolor='white', edgecolor='black',
                                          linestyle='solid'), horizontalalignment='center')

    ax.yaxis.set_major_locator(ticker.FixedLocator(range(1, 7, 1)))

    ax.set_ylim([0.5, len(c.categories) + 0.5])
    ax.set_yticklabels(c.categories)

    return ax


def plot_truncate(series, axes):
    for ax in axes:
        for i, group in series.groupby(series):
            xmin = group.index.min().to_pydatetime()
            xmax = group.index.max().to_pydatetime()
            ax.axvline(xmin, ls='--', color='k')
            ax.axvline(xmax, ls='--', color='k')
            ax.axvspan(xmin, xmax, alpha=0.15, color='0.5')

    return axes


def plot_drift(drift, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    par = dict(drift.res.params)

    #start, end = line.data.sort_index().index[[0, -1]]
    start, end = drift.readings.data.sort_index().index[[0, -1]]
    x = pd.date_range(start=start, end=end, freq='T')
    y = -drift.drift(x.to_julian_date() - drift.t0)

    lns1 = ax.plot(x, y, 'k', label='')
    ax.yaxis.set_major_locator(ticker.LinearLocator(5))
    ax.set_ylim(np.around([y.min(), y.max()], 3))
    ax.set_ylabel('Gravity [mGal]')

    ax2 = ax.twinx()
    drift.readings.data['resid'] = drift.res.resid
    for i, group in drift.readings.data.groupby('setup'):
        ax2.plot(group.index.to_pydatetime(), group.resid * 1000, 'k-')

    ax2.set_ylim([-10, 10])
    ax2.yaxis.set_major_locator(ticker.LinearLocator(5))
    ax2.set_ylabel('Residuals [$\mu$Gal]')

    ax.grid()

    #lns = lns1
    #labs = [l.get_label() for l in lns]
    #legend=ax2.legend(lns, labs, loc=2, fontsize=11)
    # legend.get_frame().set_facecolor('w')

    # residual statistics
    stats = drift.res.resid.describe()[['mean', 'min', 'max']]
    stats['rms'] = rms(drift.res.resid)
    title = 'Resid. stats.: '
    for n, x in zip(stats.index, stats.values):
        title += '{} = {:.1f} $\mu$Gal   '.format(n, x * 1000)

    ax.set_title(title)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))

    return ax


def plot_loop_processing(loop1, loop2):
    #data1 = loop1.data.set_index('time')
    #data2 = loop2.data.set_index('time')
    data1 = loop1.data
    #data2 = loop2.data
    data2 = loop2.readings.data

    fig = plt.figure(figsize=(10, 10))

    fig.add_subplot(6, 1, 1)
    date = data1.index.to_pydatetime()[0].date()
    fig.gca().text(0.0, 1.005, 'Date: ' + str(date),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=fig.gca().transAxes,
                   fontsize=11)

    fig.gca().text(1.0, 1.005, 'Gravimeter: CG5 #' +
                   str(loop1.data.meter_sn.unique()[0]),
                   horizontalalignment='right',
                   verticalalignment='bottom',
                   transform=fig.gca().transAxes,
                   fontsize=11)

    # setup
    plot_setup(data1[['setup', 'name']], ax=fig.gca())

    # error
    fig.add_subplot(6, 1, 2)
    data1['err'] = data1.stdev / np.sqrt(data1.dur - data1.rej)
    plot_error(data1.err * 1000, ax=fig.gca())

    # temperature
    fig.add_subplot(6, 1, 3)
    plot_temperature(data1.in_temp, data1.out_temp,
                     ax=fig.gca())

    # tilt
    fig.add_subplot(6, 1, 4)
    plot_tilts(data1.tilt_x, data1.tilt_y, ax=fig.gca())

    # drift
    fig.add_subplot(6, 1, 5)
    #plot_drift(loop2.drift, ax=fig.gca())
    plot_drift(loop2, ax=fig.gca())

    fig.add_subplot(6, 1, 6)
    for i, group in data2.groupby('setup'):
        autocorrelation_plot(group.g_result, ax=fig.gca(), color='k')

    # truncate
    plot_truncate(data2.setup, fig.axes)

    for ax in fig.axes:
        ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
        ax.set_xlim([dates.date2num(data1.index.min()), dates.date2num(data1.index.max())])

    fig.tight_layout()

    return fig
