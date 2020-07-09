#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np


class Drift:

    def __init__(self, readings, w_col=None, scale=1.0, order=1):
        self.readings = readings.copy()
        self.order = order
        self.scale = scale

        self.t0 = self.readings.data.jd.min()
        self.readings._data['dt0'] = self.readings.data.jd - self.t0

        formula = 'g_result ~ dt0 '
        if self.order > 1:
            for c in range(2, self.order + 1):
                formula += '+ I(dt0**' + str(c) + ')'
        formula += ' + C(name) - 1'

        if w_col is not None:
            weights = scale**2 / self.readings.data[w_col]**2
            res = smf.wls(formula, data=self.readings.data, weights=weights).fit()
        else:
            res = smf.rlm(formula, data=self.readings.data,
                    M=sm.robust.norms.HuberT()).fit()

        names, index = np.unique(self.readings.data.name.values, return_index=True)
        names = self.readings.data.name[index]
        res.model.exog_names[:len(names)] = names
        self.res = res

        coefs = np.append(self.res.params[-self.order:][::-1], 0)
        poly = np.poly1d(coefs, variable='t')
        self._drift = lambda x: poly(x)

        self.readings._data['c_drift'] = np.around(self.drift(self.readings.data.dt0), 4)

        self.readings._data['resid'] = self.res.resid
        self.readings._data['weights'] = self.res.weights

    def drift(self, *args, **kwargs):
        return -self._drift(*args, **kwargs)

    def plot(self, **kwargs):
        return plot_drift(self, **kwargs)

    def report(self):
        out = ''
        meter = self.readings.rgmeters()[0]
        out += 'Meter: '
        out += str(meter) + '\n'
        out += '== Parameters ==\n'
        out += 'Truncate@start: '
        out += str(self.readings._proc['truncate_before'])
        out += '\nTruncate@end: '
        out += str(self.readings._proc['truncate_after']) + '\n'

        out += self.res.summary2().tables[0].to_string(index=False,
                                                       header=False)

        out += '\n== Results ==\n'
        out += self.res.summary2().tables[1].iloc[:, :2].to_string()
        out += '\n== Covariance matrix ==\n'
        pd.options.display.float_format = '{:.4E}'.format
        out += self.res.cov_params().to_string()

        return out
