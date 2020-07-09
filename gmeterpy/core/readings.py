 #!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from six import string_types

import pandas as pd
import numpy as np

import gmeterpy.units as u

__all__ = ["Readings"]

class Readings:

    def __init__(self, data=None, meta=None, units=None, corrections=None, **kwargs):

        if data is not None:
            self._data = data
        else:
            self._data = pd.DataFrame()

        if meta is None:
            self._meta = {}
        else:
            self._meta = meta

        self._corrections = kwargs.pop('corrections', None)

        if corrections is None:
            self._corrections = {key: (key, {}) for key in self._data.columns
                    if 'c_' in key}
        else:
            self._corrections = corrections

        self._proc = kwargs.pop('proc', {})

        if units is None:
            self.units = {}
        else:
            self.units = units

        if not self._data.empty:
            self._update_g_result()
            self._data['jd'] = self._data.index.to_julian_date()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d

    @property
    def meta(self):
        return self._meta

    @property
    def corrections(self):
        return self._corrections

    @property
    def columns(self):
        """Return all column labels in the data.

        """
        return list(self._data.columns.values)

    @property
    def index(self):
        """Return index of the data.

        """
        return self._data.index

    def copy(self):
        return copy.deepcopy(self)

    def quantity(self, column, **kwargs):
        """ Return a `~astropy.units.Quantity` for the given column.

        Parameters
        ----------
        column : str
            The column name to return as Quantity.

        """
        # TODO: multiple columns
        # TODO: add error handling (no units)
        values = self._data[column].values
        unit = self.units[column]
        return u.Quantity(values, unit)

    def mask(self, column, minv, maxv):

        cc = self._data[column]
        c = ((cc <= minv) & (cc >= maxv))
        self._data = self._data.mask(c)

        return self

    def filter(self, minv, maxv, column='g_result', groupby=None):
        cc = self._data[column]
        c = ((cc >= minv) & (cc <= maxv))
        self._data = self._data[c]

        return self

    # start corrections
    def set_correction(self, name, value=0.0, **kwargs):
        self._corrections[name] = (value, kwargs)
        self._update_g_result()

    def _update_correction(self, name):
        value, kwargs = copy.deepcopy(self._corrections[name])
        if hasattr(value, '__call__'):
            for key in kwargs:
                if isinstance(kwargs[key],
                              string_types) and hasattr(self.data, kwargs[key]):
                    kwargs[key] = getattr(self.data, kwargs[key]).copy()
            value = value(**kwargs)

        if isinstance(value, (int, float, list, np.ndarray)):
            self._data[name] = value
        elif isinstance(value, pd.Series):
            if isinstance(value.index, pd.DatetimeIndex):
                self._data[name] = self.interpolate_from_ts(value).values
            else:
                self._data[name] = value.values
        else:
            self._data[name] = getattr(self._data, value)

    def del_correction(self, name, drop=True):
        if drop:
            del self._data[name]
        del self._corrections[name]
        self._update_g_result()

    def _update_g_result(self):
        self._data['g_result'] = self._data.g
        for key in self._corrections:
            self._update_correction(key)
            self._data.g_result += self.data[key]

    def interpolate_from_ts(self, ts):
        idx = pd.Series(index=self._data.index)
        x = pd.concat([ts, idx])
        val = x.groupby([x.index]).first().sort_index().interpolate(
            method='time')[idx.index]
        return pd.Series(val)

    def merge(self, to_merge):
        data = self.data.reset_index(drop=False)
        data = data.merge(to_merge).set_index('time').sort_index()
        self._data = data
        return self

    def to_file(self, fname, before=[], after=[], **kwargs):
        # TODO: New method __str__
        columns = before
        columns.extend(['date', 'time', 'g'])
        columns.extend(self.corrections.keys())
        columns.extend(['g_result'])
        columns.extend(after)
        data = self.data.copy()
        data['date'] = data.index.date
        data['time'] = data.index.time
        data = data.sort_index().reset_index(drop=True)
        data = data.drop('jd', axis=1)
        columns.extend(data.drop(columns, axis=1).columns)
        #pd.options.display.float_format = '{:.3f}'.format
        #data['g_result'] = data.g_result.map('{:.4f}'.format)
        #data['line'] = data.line.astype(str)
        data = data.to_string(index=False, columns=columns)
        with open(fname, 'wt') as f:
            f.write(data + '\n')
