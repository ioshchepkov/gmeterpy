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


def dmatrix_gravity(data, by=None,
        column='name', return_type='dataframe'):

    if by is not None:
        dm_data = (by + '_' + data[by].astype(str) + '_' +
                data[column].astype(str))
    else:
        dm_data = data[column]

    dm = dmatrix_dummy(dm_data,
            return_type='dataframe')

    names = {key:value for key, value in enumerate(dm.columns)}
    dm = np.asmatrix(dm)

    return _common_output(dm, names, return_type=return_type)


def dmatrix_calibration(data, by=None,
        meter_column='meter_sn',
        fix_meters=None,
        calibration_column='g_result',
        return_type='matrix'):

    if by is not None:
        dm_data = (by + '_' + data[by].astype(str) + '_' +
                data[meter_column].astype(str))
    else:
        dm_data = data[meter_column].copy()

    if fix_meters is not None:
        mask = data[meter_column].isin(fix_meters)
        dm_data[~mask] = np.nan

    dm = categorical(dm_data, drop=True)
    dm = (dm*np.tile(data[calibration_column].values.reshape(
        (dm.shape[0], 1)), dm.shape[1]))

    names = {key:value for key, value in enumerate(dm.columns)}
    dm = np.asmatrix(dm)

    return _common_output(dm, names, return_type=return_type)


def dmatrix_polynomial(data, drift_order=1, time_column='jd',
        reference_time='first', return_type='dataframe'):

    # NOTE: If time_column is None, try to use index

    # TODO: move this to an option
    if reference_time is not None:
        if reference_time == 'first':
            t0 = data[time_column].min()
        else:
            t0 = reference_time

        dt0 = data[time_column] - t0
    else:
        dt0 = data[time_column]

    dm = np.polynomial.polynomial.polyvander(dt0, deg=drift_order)
    names = ['drift_poly_t' + str(k) for k in range(dm.shape[1])]

    dm = DataFrame(dm, columns=names)
    names = {key:value for key, value in enumerate(dm.columns)}
    dm = np.asmatrix(dm)

    return _common_output(dm, names, return_type=return_type)


def dmatrix_drift(data, by=None, drift_model='polynomial',
        drift_args={'drift_order':1},
        time_column='jd', reference_time='first',
        return_type='dataframe'):

    if drift_model == 'polynomial':
        dmatrix_drift_model = dmatrix_polynomial
    else:
        raise ValueError('Only `polynomial` drift is supported.')

    if by is not None:
        dm = []
        for n, group in data.groupby(by=by):
            dmd = dmatrix_drift_model(group,
                    time_column=time_column,
                    reference_time=reference_time,
                    return_type='dataframe',
                    **drift_args)
            colnames = [str(by) + '_' + str(n) + '_' +
                    name for name in dmd.columns]

            dmd.columns = colnames
            dm.append(dmd)
        dm = concat(dm, axis=1).fillna(0)
    else:
        dm = dmatrix_drift_model(data,
                time_column=time_column,
                reference_time=reference_time,
                return_type='dataframe',
                **drift_args)

    names = {key:value for key, value in enumerate(dm.columns)}
    dm = np.asmatrix(dm)

    return _common_output(dm, names, return_type=return_type)


def dmatrix_relative_gravity_readings(readings,
        gravity=True, gravity_column='name', gravity_by=None,
        drift=True, drift_model='polynomial', drift_args={'drift_order':1},
        time_column='jd', reference_time='first', drift_by=None,
        calibration=False, calibration_column='g_result',
        meter_column='meter_sn', calibration_by=None, fix_meters=None):

    dm = []

    # gravity (stations)
    if gravity:
        dm.append(dmatrix_gravity(readings,
                by=gravity_by,
                column=gravity_column,
                return_type='dataframe')
                )

    if drift:
        dm.append(dmatrix_drift(readings,
                by=drift_by,
                drift_model=drift_model,
                drift_args=drift_args,
                time_column=time_column,
                reference_time=reference_time,
                return_type='dataframe')
                )

    if calibration:
        dm.append(dmatrix_calibration(readings,
                by=calibration_by,
                meter_column=meter_column,
                fix_meters=fix_meters,
                calibration_column=calibration_column,
                return_type='dataframe')
                )

    dm = concat(dm, axis=1).fillna(0)

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
