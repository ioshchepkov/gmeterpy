#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from  scipy.stats import sigmaclip, t

def wstdev(data, weights=None, ddof=1):
    """Return weighted standart deviation"""
    if weights is None:
        weights = np.ones_like(data)
    abs_err = np.absolute(data - data.mean())
    n = len(data)
    return np.sqrt(np.sum(weights * abs_err**2) / ((n - ddof) * np.sum(weights)))

def mad(arr, c=0.6745):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med) / c)

def sigma_clip(data, sigma=3.0, mask=True):
    _, lower, upper = sigmaclip(data, low=sigma, high=sigma)
    idx = np.where((data >= lower) & (data <= upper))[0]
    if mask:
        return idx
    else:
        return data[idx]

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
    tcrit = t.isf(1 - alpha/(2*N), N-2) # Upper critical value of the t-distribution with N − 2 degrees of freedom and a significance level of α/(2N)
    Gtest = (N-1)/np.sqrt(N) * np.sqrt(t**2 / (N-2+tcrit**2))
    G = G_i.max()
    index = G_i.argmax()
    no_outliers = (G > Gtest)
    return [G_i, Gtest, no_outliers, index]

def tau_outlier_test(x, scale=1.0, alpha=0.05):
    dof = x.size
    t_half_alpha = t.ppf(1 - alpha/2, dof - 1)
    tau_half_alpha =  t_half_alpha * np.sqrt(dof)
    tau_half_alpha /= np.sqrt(dof - 1 + t_half_alpha**2)
    return np.abs(x)/scale <= tau_half_alpha

def rms(x):
    """Return root mean square"""
    return np.sqrt(np.mean(x**2))

def lstsqadj(A, L, W=None):

    if W is None:
        W = np.asmatrix(np.identity(len(l)))

    #normal matrix
    N = A.T * W * A
    b = A.T * W * L 

    #least squares solution
    x, rs, rank, sv = np.linalg.lstsq(N, b)

    #cofactor matrix
    Qxx = np.linalg.inv(N)

    #residuals
    v = A*x - L

    #std of unit weigth
    n, k = A.shape
    s0 = float(np.sqrt(v.T*W*v/(n-k)))

    #covariance matrix
    Cxx = s0**2 * Qxx 

    return x, rs, rank, v, s0, Cxx

def lstsqadj_free_datum(A, L, S, W=None):

    if W is None:
        W = np.asmatrix(np.identity(len(l)))

    #normal matrix
    SS = S*S.T
    N = A.T * W * A
    b = A.T * W * L 

    #least squares solution
    x, rs, rank, sv = np.linalg.lstsq(N + SS, b)

    #residuals
    v = A*x - L

    #std of unit weigth
    n, k = A.shape
    s0 = float(np.sqrt(v.T*W*v/(n+1-k)))

    #covariance matrix
    Cxx = s0**2 * np.linalg.pinv(N) 

    return x, rs, rank, v, s0, Cxx

def interpolate(ts, datetime_index):
    x = pd.concat([ts, pd.Series(index=datetime_index)])
    return x.groupby(x.index).first().sort_index().interpolate(method="linear")[datetime_index]

"""
def chi2_test(self, alpha=0.95):
    lower, upper = np.asarray(chi2.interval(alpha, self.df_resid))
    chisq = self.s0**2/self.scale**2 * self.df_resid

    x = [self.scale, self.s0**2, self.s0, chisq, lower, upper]
    n = ['astdev', 'var0', 'stdev0', 'chisq', 'lower', 'upper']
    out = pd.Series(x, n)

    if lower < chisq < upper:
        cond = True
    else:
        cond = False

    return out, cond
"""
