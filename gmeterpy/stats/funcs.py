#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import t


def grubbs_outlier_test(y_i, alpha=0.95):
    """
    Perform Grubbs' outlier test.

    ARGUMENTS
    y_i (list or numpy array) - dataset
    alpha (float) - significance cutoff for test

    RETURNS
    G_i (list) - Grubbs G statistic for each member of the dataset
    Gtest (float) - rejection cutoff; hypothesis that no outliers exist if G_i.max() > Gtest
    no_outliers (bool) - boolean indicating whether there are no outliers at specified significance level
    index (int) - integer index of outlier with maximum G_i
    """
    s = y_i.std()
    G_i = np.abs(y_i - y_i.mean()) / s
    N = y_i.size
    # Upper critical value of the t-distribution with N − 2 degrees of freedom and a significance level of α/(2N)
    tcrit = t.isf(1 - alpha/(2*N), N-2)
    Gtest = (N-1)/np.sqrt(N) * np.sqrt(t**2 / (N-2+tcrit**2))
    G = G_i.max()
    index = G_i.argmax()
    no_outliers = (G > Gtest)
    return [G_i, Gtest, no_outliers, index]


def tau_outlier_test(x, scale=1.0, alpha=0.05):
    dof = x.size
    t_half_alpha = t.ppf(1 - alpha/2, dof - 1)
    tau_half_alpha = t_half_alpha * np.sqrt(dof)
    tau_half_alpha /= np.sqrt(dof - 1 + t_half_alpha**2)
    return np.abs(x)/scale <= tau_half_alpha


def rms(x):
    """Return root mean square"""
    return np.sqrt(np.mean(x**2))
