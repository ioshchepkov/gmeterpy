#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame, concat
from patsy import dmatrix
from statsmodels.tools.tools import categorical
from scipy.linalg import block_diag


def _common_output(dm, names, return_type='dataframe'):
    if return_type == 'matrix':
        return dm, names
    elif return_type == 'dataframe':
        return DataFrame(dm, columns=names.values())
    else:
        raise ValueError('return_type must be "dataframe" or "matrix"')


def dmatrix_dummy(data, return_type='dataframe'):
    if len(data) == 1:
        dm, names = [1], {0: data[0]}
    else:
        dm, names = categorical(data, drop=True, dictnames=True)

    return _common_output(dm, names, return_type=return_type)


def dmatrix_calibration(meters, z, return_type='matrix'):
    dm, names = categorical(meters, drop=True, dictnames=True)
    dm = dm*np.tile(z.reshape((len(z), 1)), len(names))
    return _common_output(dm, names, return_type=return_type)


def dmatrix_drift_poly(data, order=1, return_type='dataframe'):
    dm = DataFrame()
    data['date'] = data.index.date
    for n, line in data.groupby(['line']):
        line = line.copy()
        line['dt0'] = line.jd - line.jd.min()
        dml = np.vander(line.dt0, order+1)
        names = [str(n) + '_t' + str(k) for k in range(order+1)][::-1]
        dml = DataFrame(dml, columns=names)
        dm = dm.append(dml).fillna(0)

    names = {key: value for key, value in enumerate(dm.columns)}
    dm = np.asmatrix(dm)

    return _common_output(dm, names, return_type=return_type)


def dmatrix_relative_gravity_readings(readings, fix_meters='all', drift=False, **kwargs):
    dm = dmatrix_dummy(readings.name.values)
    if fix_meters != 'all':
        dmrg = dmatrix_calibration(readings.meter_sn.values,
                                   readings.g_result.values, return_type='dataframe')
        dmrg = dmrg.drop(fix_meters, axis=1)
        dm = concat([dm, dmrg], axis=1)
    if drift:
        dmd = dmatrix_drift_poly(readings, **kwargs)
        dm = concat([dm, dmd], axis=1)
    else:
        dml = dmatrix_dummy(readings.line.values)
        dm = concat([dm, dml], axis=1)

    return dm


def dmatrix_ties(ties, fix_meters='all'):
    stations = np.append(ties.data['from'].values, ties.data['to'].values)
    dm, names = dmatrix_dummy(stations, return_type='matrix')
    dm1, dm2 = np.split(dm, 2)
    dm = DataFrame(-dm1 + dm2, columns=names.values())
    if fix_meters != 'all':
        dmrg = dmatrix_calibration(ties.data.meter_sn.values,
                                   ties.data.delta_g.values, return_type='dataframe')
        dmrg = dmrg.drop(fix_meters, axis=1)
        dm = concat([dm, dmrg], axis=1)

    return dm


def dmatrix_gravity_values(data, ag_shift=False, fix_meters='all', **kwargs):
    dm = dmatrix_dummy(data.name.values)
    if ag_shift and fix_meters != 'all':
        dms = dmatrix_dummy(data.meter_sn.values)
        dms = dms.drop(fix_meters, axis=1)
        dm = concat([dm, dms], axis=1)

    return dm


def dmvstack(dm1, wm1, l1, dm2, wm2, l2):
    dm = dm1.append(dm2, sort=True).fillna(0)
    wm = block_diag(wm1, wm2)
    l = np.vstack((l1, l2))

    return dm, wm, l
