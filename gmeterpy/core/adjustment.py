#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm
from scipy.stats import chi2

from gmeterpy.core.relative import RelativeReadings
from gmeterpy.core.relative import Ties
from gmeterpy.core.gvalues import GravityValues

from gmeterpy.utils.dmatrices import dmvstack, dmatrix_dummy
from gmeterpy.utils.stats import rms, tau_outlier_test


class Adjustment:
    def __init__(self):
        self.stations = np.array([])
        self.fix_stations = np.array([])

        self.rgmeters = np.array([])
        self.fix_rgmeters = np.array([])
        self.det_rgmeters = np.array([])

        self.agmeters = np.array([])
        self.fix_agmeters = np.array([])
        self.det_agmeters = np.array([])

        self.dm = None
        self.wm = None
        self.l = None

        self._endog_position = {}

    def copy(self):
        return deepcopy(self)

    def _add_to_position(self, value):
        if value in self._endog_position.values():
            raise ValueError('There is a same data type already!')
        if self._endog_position:
            key = max(self._endog_position) + 1
        else:
            key = 0

        self._endog_position.update({key:value})

    def add_data(self, data, **kwargs):
        dm, wm, l = data.dmatrices(**kwargs)

        if self.dm is not None:
            self.dm, self.wm, self.l = dmvstack(self.dm, self.wm, self.l, dm, wm, l)
        else:
            self.dm, self.wm, self.l = dm, wm, l

        if hasattr(data, 'stations'):
            self.stations = np.unique(np.append(self.stations, data.stations()))
        if hasattr(data, 'rgmeters'):
            self.rgmeters = np.unique(np.append(self.rgmeters, data.rgmeters()))
        if hasattr(data, 'agmeters'):
            self.agmeters = np.unique(np.append(self.agmeters, data.agmeters()))

        if isinstance(data, (RelativeReadings, Ties)):
            if isinstance(data, RelativeReadings):
                self.readings = data
                self._add_to_position('readings')
                self._drift_params = dm.drop(data.stations(), axis=1).columns
            elif isinstance(data, Ties):
                self.ties = data
                self._add_to_position('ties')

            fix_rgmeters = kwargs.get('fix_meters', 'all')
            if fix_rgmeters != 'all':
                self.fix_rgmeters = np.append(self.fix_rgmeters, fix_rgmeters)
                det_rgmeters = [x for x in self.rgmeters if x not in
                        self.fix_rgmeters]
                self.det_rgmeters = np.append(self.det_rgmeters, det_rgmeters)

        elif isinstance(data, GravityValues):
            self.fix_stations = data
            self._add_to_position('gvalues')
            fix_agmeters = kwargs.get('fix_meters', 'all')
            if fix_agmeters != 'all':
                self.fix_agmeters = np.append(self.fix_agmeters, fix_agmeters)
                det_agmeters = np.setdiff1d(self.agmeters, fix_agmeters)
                self.det_agmeters = np.append(self.det_agmeters, det_agmeters)
                if len(fix_agmeters) == 0:
                    offset = dmatrix_dummy(np.sort(self.det_agmeters))
                    w_offset = kwargs.get('weighted_offset', True)
                    if w_offset:
                        weights = self.fix_stations.data[['meter_sn',
                            'weight']].groupby('meter_sn').mean().sort_index()
                        weights = weights.weight.values
                    else:
                        weights = np.ones(len(offset))
                    self.dm, self.wm, self.l = dmvstack(self.dm, self.wm,
                            self.l, offset, np.diag(weights),
                            np.asmatrix(np.zeros(len(offset))).T)

        self.det_rgmeters = np.unique(self.det_rgmeters)

        return self

    def det_stations(self):
        out = np.array([x for x in self.stations if x not in
            self.fix_stations.stations()])
        return out

    def adjust(self, method='WLS', **kwargs):
        if method == 'OLS':
            model = sm.OLS(self.l, self.dm, **kwargs)
        elif method == 'WLS':
            model = sm.WLS(self.l, self.dm, weights=np.diag(self.wm), **kwargs)
        elif method == 'RLM':
            model = sm.RLM(self.l, self.dm, **kwargs)

        self._method = method

        return AdjustmentResults(model.fit(), self)

class AdjustmentResults:
    def __init__(self, results, model):

        self.res = results
        self.model = model

        names = self.model.dm.columns

        self._params = self.res.params
        self._sigma = pd.Series(np.sqrt(np.diag(self.res.cov_params())), names)
        self._s0 = np.sqrt(self.res.scale)

    @property
    def params(self):
        return self._params

    @property
    def sigma(self):
        return self._sigma

    @property
    def s0(self):
        return self._s0

    def ties(self):
        data = self.model.ties.data.copy()
        data['weight'] = self.res.model.weights[:len(data)]
        data['resid'] = self.res.resid.iloc[:len(data)].values

        if self.model._method in ('OLS', 'WLS'):
            tau_passed = tau_outlier_test(self.res.wresid.values, scale=self.s0)
            data['stresid'] = self.res.wresid.iloc[:len(data)].values
            #data['tau_passed'] = tau_outlier_test(data.stresid, scale=self.s0)
            data['tau_passed'] = tau_passed[:len(data)]

        u_adj = self.adjusted_ties()[['from', 'to', 'stdev']]
        u_adj.columns = ['from', 'to', 'u_adj']
        data = data.reset_index().merge(u_adj, on = ['from', 'to']).set_index('index')
        data.index.name = ''

        data['En'] = (data.resid.abs() / np.sqrt(data.stdev**2 +
                data.u_adj**2)).round(1)

        return data

    def fixed_stations(self):
        columns = ['name', 'meter_sn', 'g', 'stdev', 'weight']
        fix_data = self.model.fix_stations.data.copy()[columns].set_index('name')
        if len(self.model.det_agmeters) != 0 and len(self.model.fix_agmeters) == 0:
            idx_l = -(len(fix_data) + len(self.model.det_agmeters))
            idx_r = - len(self.model.det_agmeters)
        else:
            idx_l = -len(fix_data)
            idx_r = None

        #fix_data['resid'] = self.res.resid.iloc[-len(fix_data):].values
        fix_data['resid'] = self.res.resid.iloc[idx_l:idx_r].values
        if self.model in ('OLS', 'WLS'):

            tau_passed = tau_outlier_test(self.res.wresid.values, scale=self.s0)
            fix_data['stresid'] = self.res.wresid.iloc[idx_l:idx_r].values
            #fix_data['tau_passed'] = tau_outlier_test(fix_data.stresid.values, scale=self.s0)
            fix_data['tau_passed'] = tau_passed[idx_l:idx_r]

        #fix_data['resid'] = fix_data.g - fix_data.adj
        #fix_data['weight'] = self.res.model.weights[-len(fix_data):]
        fix_data['weight'] = self.res.model.weights[idx_l:idx_r]
        fix_data['u_adj'] = self.sigma.loc[self.model.fix_stations.stations()]
        fix_data['En'] = (fix_data.resid.abs() / np.sqrt(fix_data.stdev**2 +
                fix_data.u_adj**2)).round(1)
        return fix_data.reset_index().set_index(self.model.fix_stations.data.index)

    def fixed_stations_stats(self, by):
        agrouped = self.fixed_stations().groupby(by)
        data=pd.DataFrame({'mean' : agrouped.resid.mean(),
            'n' : agrouped.resid.count(),
            'rms': agrouped.resid.apply(rms)})
        pd.options.display.float_format = '{:.4f}'.format
        return data.reset_index()

    def gravity(self, stations='all'):
        if stations == 'all':
            idx = self.model.stations
        elif stations == 'unknown':
            idx = self.model.det_stations()

        df = pd.DataFrame({
            'g' : self.params.loc[idx],
            'stdev' : self.sigma.loc[idx]})
        pd.options.display.float_format = '{:.4f}'.format
        return df

    def calibration(self):
        df = pd.DataFrame({
            'coef' : 1 - self.params.loc[self.model.det_rgmeters],
            'stdev' : self.sigma.loc[self.model.det_rgmeters]})
        df.index.name = 'meter_sn'
        df = df.reset_index()
        df['meter_sn'] = df.meter_sn.astype(int).astype(str)
        pd.options.display.float_format = '{:.7f}'.format
        return df

    def offset(self):
        df = pd.DataFrame({
            'offset' : self.params.loc[self.model.det_agmeters],
            'stdev' : self.sigma.loc[self.model.det_agmeters]})
        df.index.name = 'meter_sn'
        df = df.reset_index()
        pd.options.display.float_format = '{:.4f}'.format
        return df

    def adjusted_ties(self):
        st = self.model.stations

        adj_ties= pd.DataFrame()
        for i, s1 in enumerate(st):
            for s2 in st[i+1:]:
                delta_g = self.params[s2] - self.params[s1]
                var1 = self.res.cov_params()[s1][s1]
                var2 = self.res.cov_params()[s2][s2]
                cov = self.res.cov_params()[s1][s2]
                u =  np.sqrt(var1 + var2 - 2*cov)
                adj_tie = pd.Series({'from': s1, 'to': s2,
                    'delta_g':delta_g, 'stdev':u})
                adj_ties = adj_ties.append(adj_tie, ignore_index=True,
                        sort=True)
        return adj_ties

    def drift_params(self):
        idx = self.model._drift_params
        if hasattr(self.model, '_drift_params'):
            lines = [l.split('_')[0] for l in idx]
            parnames = [l.split('_')[1] for l in idx]
            unique_lines = np.unique(lines)
            n = len(unique_lines)
            params = self.params[idx].values
            coef = np.reshape(params, (n, len(parnames)))

            df = pd.DataFrame(coef, columns=parnames, index=unique_lines)
            df.index.names=['line']

            return df.reset_index()
        else:
            'There is no drift parameters in this adjustment!'

    def resid_stats(self, by=['meter_sn']):
        grouped  = self.ties().groupby(by=by)
        df = pd.DataFrame({
            'mean' : grouped.resid.mean(),
            'n': grouped.resid.count(),
            'rms' :grouped.resid.apply(rms)})
        return df

    def chi2(self, scale=1.0):
        r = self.res.df_resid
        lower, upper = np.asarray(chi2.interval(0.95, r))
        chisq = self.s0**2/scale**2 * r
        return chisq, lower, upper

    def qqplot(self, ax=None, **kwargs):
        """Q-Q plot of the residuals"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        sm.graphics.qqplot(self.res.resid, fit=True, line='s', ax=ax, **kwargs)
        ax.set_title('Normal Q-Q plot')
        return ax

    def report(self, scale=1.0):
        if hasattr(self.model, 'ties'):
            date = self.model.ties.data.date
            t=(date.min() + (date.max() - date.min()) / 2)
            epoch = round((t.year + t.month/12 + t.day/t.daysinmonth / 12), 3)
        else:
            epoch = None

        #write files
        with open('gravity.txt', 'w') as f:
            f.write(self.gravity().to_string() + '\n')
        with open('calibration.txt', 'w') as f:
            pd.options.display.float_format = '{:.7f}'.format
            f.write(self.calibration().to_string() + '\n')
        with open('adjreport.txt', 'w') as f:
            f.write('Mean epoch: ' + str(epoch))
            n = self.res.nobs
            k = self.res.df_model
            r = self.res.df_resid
            f.write('\nObservations: {:.0f}\nUnknowns: {:.0f}\nDegrees of freedom: {:.0f}\n'.format(n, k, r))
            """
            f.write('\nA prirori st.dev. for ties: {:.4f}\n'.format(astdev))
            f.write('\nExcluded\n')
            f.write(10*'=' + '\n')
            f.write(exclude)
            """

            if self.model._method in ('OLS', 'WLS'):
                f.write('\nChi2 test\n')
                f.write(9*'=' + '\n')
                chisq, lower, upper = self.chi2(scale=scale)
                f.write('A priori st.dev. u. weight: {}\n'.format(scale))
                f.write('Ref. var.: {}\n'.format(self.s0**2))
                f.write('Ref.st. dev.: {}\n'.format(self.s0))
                f.write('X^2 = {}\n'.format(chisq))
                f.write('lower = {}\n'.format(lower))
                f.write('upper = {}\n'.format(upper))
                if lower < chisq < upper:
                    f.write('Test PASSED!\n')
                else:
                    f.write('Test FAILED!\n')

            pd.options.display.float_format = '{:.4f}'.format

            if self.model.fix_stations:
                f.write('\nFixed data and residuals\n')
                f.write(25*'=' + '\n')
                pd.options.display.float_format = '{:.4f}'.format
                f.write(self.fixed_stations().to_string() + '\n')

                rms_resid = rms(self.fixed_stations().resid)
                n_resid = len(self.fixed_stations())
                f.write('Total: rms = {:.4f}, n = {:.0f}\n'.format(rms_resid,
                    n_resid))

                f.write('\nResiduals statistics by absolute meters\n')
                f.write(self.fixed_stations_stats('meter_sn').to_string() + '\n')

                if not self.offset().empty:
                    f.write('\nOffsets of the absolute meters\n')
                    f.write(self.offset().to_string() + '\n')

                f.write('\nResiduals statistics by fixed stations\n')
                f.write(self.fixed_stations_stats('name').to_string() + '\n')

            if hasattr(self.model, 'ties'):
                f.write('\nTies and residuals\n')
                f.write(18*'=' + '\n')
                pd.options.display.float_format = '{:.4f}'.format
                f.write(self.ties().to_string() + '\n')

                f.write('\nResiduals statistics by relative meters\n')
                f.write(39*'=' + '\n')
                f.write(self.resid_stats().sort_values('n',
                    ascending=True).reset_index().to_string())
                f.write('\nTotal: rms = {:.4f}, n = {:.0f}\n'.format(rms(self.ties().resid),
                    len(self.ties())))
                f.write('\n\nResiduals statistics by ties\n')
                f.write(28*'=' + '\n')
                f.write(self.resid_stats(['from', 'to']).sort_values('n',
                    ascending=False).reset_index().to_string())
                f.write('\n\nResiduals statistics by operators\n')
                f.write(33*'=' + '\n')
                f.write(self.resid_stats(['operator']).sort_values('n',
                    ascending=False).reset_index().to_string())
                f.write('\n\nRepeated statistics\n')
                f.write(19*'=' + '\n')
                grouped = self.ties().groupby(['from', 'to', 'meter_sn'])[['resid']]
                df = pd.DataFrame()
                for i, group in grouped:
                    if len(group) > 1:
                        group = group.copy()
                        group['from_mean'] = np.round((group.delta_g - group.delta_g.mean()), 4)
                        df = df.append(group, sort=True)
                        f.write(group.to_string(index=False) + '\n')

            with open('ties_adjusted.txt', 'w') as f:
                pd.options.display.float_format = '{:.4f}'.format
                f.write(self.adjusted_ties()[['from', 'to', 'delta_g', 'stdev']].to_string(index = False))

        with open('cov.txt', 'w') as f:
            pd.options.display.float_format = '{:.8E}'.format
            f.write(self.res.cov_params().to_string(index = True))

        with open('residuals.txt', 'w') as f:
            pd.options.display.float_format = '{:.4f}'.format
            f.write(self.res.resid.to_string(index = False))
