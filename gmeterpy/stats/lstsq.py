#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def lstsqadj(A, L, W=None):

    if W is None:
        W = np.asmatrix(np.identity(len(l)))

    # normal matrix
    N = A.T * W * A
    b = A.T * W * L

    # least squares solution
    x, rs, rank, sv = np.linalg.lstsq(N, b)

    # cofactor matrix
    Qxx = np.linalg.inv(N)

    # residuals
    v = A*x - L

    # std of unit weigth
    n, k = A.shape
    s0 = float(np.sqrt(v.T*W*v/(n-k)))

    # covariance matrix
    Cxx = s0**2 * Qxx

    return x, rs, rank, v, s0, Cxx


def lstsqadj_free_datum(A, L, S, W=None):

    if W is None:
        W = np.asmatrix(np.identity(len(l)))

    # normal matrix
    SS = S*S.T
    N = A.T * W * A
    b = A.T * W * L

    # least squares solution
    x, rs, rank, sv = np.linalg.lstsq(N + SS, b)

    # residuals
    v = A*x - L

    # std of unit weigth
    n, k = A.shape
    s0 = float(np.sqrt(v.T*W*v/(n+1-k)))

    # covariance matrix
    Cxx = s0**2 * np.linalg.pinv(N)

    return x, rs, rank, v, s0, Cxx
