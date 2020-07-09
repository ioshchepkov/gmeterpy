
import numpy as np
import matplotlib.pylab as plt


def plot_drops(x, rejected=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    if rejected is not None:
        pass
    ax.plot(x, '.')
    ax.set_xlabel('Drop number')
    ax.set_xlim(1, len(x))
    ax.set_ylabel('$\mu Gal$')
    return ax


def plot_residuals(residuals, time=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    if time is None:
        time = np.arange(len(residuals))
        ax.set_xlim(0, len(residuals))
    else:
        ax.set_xlabel('Time [s]')

    ax.plot(time, residuals)
    ax.set_ylabel('Residuals [nm]')
    ax.grid()

    return ax


def plot_sets():
    pass
