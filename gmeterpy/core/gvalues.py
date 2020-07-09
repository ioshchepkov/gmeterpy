#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd

import statsmodels.api as sm
from gmeterpy.utils.dmatrices import dmatrix_gravity_values

class GravityValues:
    def __init__(self, data=None, meta=None):
        self._data = data
        self._meta = meta

    @property
    def data(self):
        return self._data

    @property
    def meta(self):
        return self._meta

    @classmethod
    def from_file(cls, fname, **kwargs):
        df = pd.read_csv(fname, **kwargs)
        return GravityValues(df)

    def to_file(self, fname, **kwargs):
        pd.options.display.float_format = '{:.4f}'.format
        with open(fname, 'w') as f:
            f.write(self.__str__(**kwargs) + '\n')

    def __str__(self, **kwargs):
        return self.data.to_string(**kwargs)

    def stations(self):
        return self.data.name.unique()

    def agmeters(self):
        if hasattr(self.data, 'meter_sn'):
            return self.data.meter_sn.unique()
        else:
            return np.array([])

    def dmatrices(self, w_col=None, **kwargs):
        dm = dmatrix_gravity_values(self.data, **kwargs)
        if w_col is not None:
            wm = np.diag(self.data[w_col])
        else:
            wm = np.eye(dm.shape[0], dm.shape[1])

        y = np.asmatrix(self.data.g).T

        return dm, wm, y
