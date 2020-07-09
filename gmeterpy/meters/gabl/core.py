#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import sigmaclip, ttest_ind_from_stats, t, f
from tqdm import tqdm

from gmeterpy.core.readings import Readings
from gmeterpy.meters.freefall.fsol import doppler_shift_corr
from gmeterpy.corrections import atmosphere_pressure_corr
from gmeterpy.corrections import polar_motion_corr
from gmeterpy.meters.gabl.parser import load_from_path
from gmeterpy.plotting.absolute import plot_residuals

_formatters = {
    'doy': '{:.0f}'.format,
    'year': '{:.0f}'.format,
    'time': '{:%H:%M:%S}'.format,
    'temp': '{:.1f}'.format,
    'h_eff': '{:.3f}'.format,
    'v0': '{:.3f}'.format,
    'xp': '{:.4f}'.format,
    'yp': '{:.4f}'.format,
    'drops': '{:.0f}'.format,
    'series': '{:.0f}'.format,
    'set': '{:.0f}'.format,
    'accepted': '{:.0f}'.format,
    'rejected': '{:.0f}'.format,
    'date': '{:%d.%m.%Y}'.format}


_corrections = {'c_polar': (polar_motion_corr, {'xp': 'xp',
                                                'yp': 'yp', 'lat': 'lat', 'lon': 'lon'}),
                'c_vac': (lambda x, y: x * y, {'x': 'aero_coeff', 'y': 'vac'}),
                'c_atm': (atmosphere_pressure_corr, {'height': 'height', 'p_0':
                                                     'pres', 'baro_factor': 'baro_factor'}),
                'c_dop': (doppler_shift_corr, {'g0': 'g', 'v0': 'v0', 't':
                                               'drop_duration'}),
                }

def t_statistics(x, stderr, n, alpha=0.05):
    stderr = np.sqrt(stderr**2)
    std = stderr*np.sqrt(n)
    t_exp, pval = ttest_ind_from_stats(x[0], std[0], n[0], x[1], std[1], n[1],
            equal_var = False)
    df = np.sum(stderr**2) / np.sum(stderr**2 /(n-2))
    t_crit = t.ppf(1 - alpha/2, df)
    passed = False
    if pval > alpha:
        passed = True
    return (t_exp, pval, t_crit, df, passed)

def mean_time(data):
    return data.time.min() + (data.time - data.time.min()).mean()

def group_wmean(group, weighted=True):
    data = group.g_result.values
    if len(data) > 1:
        if weighted:
            weights = 1 / group.err.values**2
            g_result = np.average(data, weights=weights)
            err = np.sqrt(
                    np.sum(weights*(data - g_result)**2) /
                    ((data.size - 1) * np.sum(weights))
                    )
            stdev = err * np.sqrt(data.size)
        else:
            g_result = np.mean(data)
            err = np.sqrt(np.sum(group.err.values**2)) / data.size
            stdev = np.std(data, ddof=1)
    else:
        g_result = group.g_result.values[0]
        err = group.err.values[0]
        stdev = group.stdev.values[0]

    mtime = mean_time(group)
    results = pd.Series({
        'g_result': g_result,
        'date': mtime,
        'time': mtime,
        'err': err,
        'stdev': stdev,
        'xp': group.xp.mean(),
        'yp': group.yp.mean(),
        'pres': group.pres.mean(),
        'temp': group.temp.mean(),
        'h_eff': group.h_eff.mean(),
        'v0': group.v0.mean(),
        'accepted': group.accepted.sum(),
        'rejected': group.rejected.sum(),
    })
    return results


class GABLProject(Readings):

    def __init__(self, drops=[], *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._drops = drops
        self._set = None
        self._corrections = kwargs.pop('corrections', _corrections)

        self._path = None

    @classmethod
    def load(cls, path, **kwargs):
        return GABLProject(load_from_path(path, **kwargs))

    def get_drops(self, azimuth=None, seance=None, series=None, accepted=True):
        data = self._drops.copy()
        if azimuth is not None:
            data = list(
                filter(lambda x: x._model._meta['azimuth'] == azimuth, data))
        if seance is not None:
            data = list(
                filter(lambda x: x._model._meta['seance'] == seance, data))
        if series is not None:
            data = list(
                filter(lambda x: x._model._meta['series'] == series, data))
        if accepted:
            data = list(filter(lambda x: x._model._meta['accepted'], data))
        return data

    def proc_drops(self, before=0, after=0, **kwargs):
        data = pd.DataFrame()
        for nn, drop in tqdm(enumerate(self._drops), total=len(self._drops)):
            drop.truncate(before, after)
            drop = drop.fit(**kwargs)
            data = data.append(pd.Series({**drop.results, **drop._model._meta}),
                               ignore_index=True)
            self._drops[nn] = drop

        data['accepted'] = data['accepted'].astype(bool)
        if np.any([data.lat.unique().size, data.lon.unique().size,
                   data.height.unique().size]) > 1:
            raise ValueError('Station coordinates are not the same for drops')
        data['jd'] = data.set_index('time').index.to_julian_date()
        data.rename(columns={'g0': 'g'}, inplace=True)
        self._data = data.sort_values(['time']).set_index('time')
        self._update_g_result()
        self.h_eff()

        self._data['time'] = self._data.index
        self._data['doy'] = self._data.index.dayofyear
        self._data['year'] = self._data.index.year
        self._data['series'] = self._data['series'].astype(int)
        self._data['nn'] = self._data['nn'].astype(int)

    def filter(self, by=['azimuth', 'seance', 'series'], sigma=3.0):
        # filter drops in series by sigma-rule
        _data = pd.DataFrame()
        data = self._data.copy()
        data['time'] = data.index
        for _, group in data.reset_index(drop=True).groupby(by):
            group = group.copy()
            _, lower, upper = sigmaclip(
                group.g_result.values, low=sigma, high=sigma)
            condition = (group.g_result >= lower) & (group.g_result <= upper)
            group.loc[~condition, 'accepted'] = False
            _data = _data.append(group, ignore_index=True)
        self._data = _data.set_index('time', drop=False).sort_index()

        accepted = self._data['accepted'].values
        rejected_idx = np.where(~accepted)[0]
        for idx, drop in enumerate(self._drops):
            if idx in rejected_idx:
                drop._model._meta['accepted'] = False
            else:
                drop._model._meta['accepted'] = True

        return self

    def mean(self, by=['seance', 'azimuth', 'series'], corrections=True):

        if corrections:
            corrs = sorted(self.corrections.keys())
        else:
            corrs = []

        cols = ['seance', 'azimuth', 'series', 'time', 'doy', 'year',
                'g_result'] + corrs + ['xp', 'yp', 'pres', 'temp', 'h_eff',
                                       'v0', 'accepted']

        series_rej = self._data.loc[~self._data.accepted,
                                    cols].groupby(by)
        series = self._data.loc[self._data.accepted, cols].groupby(by)

        srs_mean = series.mean()
        srs_mean['time'] = series.apply(lambda x: x.index.min() +
                                        (x.index - x.index.min()).to_series().mean())
        srs_mean['stdev'] = series.g_result.std()
        srs_mean['err'] = series.g_result.sem()
        srs_mean['pres'] = series.pres.median()
        srs_mean['temp'] = series.temp.median()
        srs_mean['h_eff'] = series.h_eff.median()
        srs_mean['v0'] = series.v0.median()
        srs_mean['accepted'] = series.size()
        srs_mean['rejected'] = series_rej.size().fillna(0)
        cols = ['time', 'doy', 'year',
                'g_result', 'stdev', 'err'] + corrs + [
            'xp', 'yp', 'pres', 'temp', 'h_eff', 'v0', 'accepted',
                        'rejected']

        srs_mean = srs_mean[cols].sort_values('time')

        self._set = GABLSet(srs_mean)

        return self

    def get_residuals(self, by=['azimuth', 'seance', 'series']):
        out = []
        for idx, group in self._data.groupby(by):
            drops = self.get_drops(*idx)
            residuals = []
            for drop in drops:
                residuals.append(drop.stats_results.resid.tolist())
            out.append((idx, residuals))
        return out

    def plot_residuals(self, by=['azimuth', 'seance', 'series']):
        plots = []
        for name, residuals in self.get_residuals(by=by):
            resid_mean = np.mean(np.array(residuals).T, axis=1) * 10e9
            fig = plot_residuals(resid_mean).figure
            fig_name = '_'.join(str(x) for x in name)
            plots.append((fig_name, fig))
        return plots

    def h_eff(self, by=['seance', 'series']):
        data = pd.DataFrame()
        for _, group in self._data.reset_index(drop=False).groupby(by):
            group = group.copy()
            h0 = 0.5 * group.v0.mean()**2 / (group.g.mean() * 10**-8)
            h_eff = group.meter_height.mean() - h0 - group.h1.mean()
            group['h0'] = h0
            group['h_eff'] = h_eff
            data = data.append(group, ignore_index=True)
        self._data = data.set_index('time')

    def report(self):
        data = self._data.copy().reset_index(drop=True)
        corrs = sorted(self.corrections.keys())

        cols = ['seance', 'azimuth', 'series', 'nn', 'time',
                'doy', 'year', 'g', 'err'] + corrs + ['g_result', 'xp', 'yp',
                                                      'pres', 'temp', 'h_eff', 'v0',
                                                      'accepted']

        pd.options.display.float_format = '{:.2f}'.format
        with open('drops.txt', 'w') as f:
            f.write(data[cols].sort_values('time').set_index([
                'azimuth', 'seance', 'series', 'nn']).to_string(formatters=_formatters,
                                                                sparsify=False))

        with open('set.txt', 'w') as f:
            f.write(self._set._data.to_string(formatters=_formatters))

        if self._set._seances is not None:
            cols = ['date', 'time',
                    'g_result', 'err', 'stdev', 'xp', 'yp', 'pres', 'temp',
                    'h_eff', 'v0', 'set', 'accepted', 'rejected']

            with open('seances.txt', 'w') as f:
                f.write(self._set._seances[cols].sort_values('time').to_string(
                    formatters=_formatters))

        cols = ['azimuth', 'date', 'time',
                'g_result', 'err', 'stdev', 'xp', 'yp', 'pres',
                'temp', 'h_eff', 'v0', 'set', 'accepted', 'rejected']

        with open('azimuths.txt', 'w') as f:
            f.write(self._set._azimuths[cols].sort_values('time').set_index('azimuth').to_string(
                formatters=_formatters))
            f.write('\n\n')
            cols.remove('stdev')
            f.write(self._set._directions[cols].sort_values('time').set_index('azimuth').to_string(
                formatters=_formatters))
            f.write('\n\n')
            if len(self._set._directions) > 1:
                dirs = self._set._directions.azimuth
                t_exp, pval, t_crit, dof, passed = self._set._t_test
                f.write('Welch\'s t-test\n')
                f.write('t = {:.3f}, p-value = {:.3f}\n'.format(
                    t_exp, pval))
                f.write('t_critical = {:.3f}, alpha = {}\n'.format(t_crit, 0.05))
                if not passed:
                    f.write('NOT Passed! (p-value < alpha and |t| > t_critical)\n')
                    f.write('The mean values are significantly different!')
                else:
                    f.write('Passed! (p-value > alpha and |t| < t_critical)')
                f.write('\n\n')
            f.write(self._set._final[cols[1:]].to_string(
                formatters=_formatters))

        with open('final.txt', 'w') as f:
            f.write(self._set._final[cols[1:]].to_string(
                formatters=_formatters))


class GABLSet:

    def __init__(self, data):
        #self._date = datetime.datetime.utcnow()
        self._data = data
        self._seances = None
        self._azimuths = None
        self._directions = None
        self._t_test = None
        self._final = None

    def proc(self):
        # Seances calculation
        if 'seance' in self._data.index.names:
            seances = self._data.reset_index(
                drop=False).groupby(['azimuth', 'seance'])
            results = seances.apply(lambda x: group_wmean(x, weighted=True))
            results['set'] = seances.size()
            self._seances = results

        results = self._data

        # Combine azimuths
        az_grouped = results.reset_index().groupby('azimuth')
        az_result = az_grouped.apply(lambda x: group_wmean(x, weighted=True))
        az_result['set'] = az_grouped[self._data.index.names[-1]].size()

        az_result = az_result.reset_index(drop=False)
        self._azimuths = az_result

        def get_azimuth_line(az):
            if az in ('n', 's'):
                return 'ns'
            elif az in ('w', 'e'):
                return 'we'

        dir_grouped = az_result.set_index('azimuth').groupby(get_azimuth_line)
        dir_result = dir_grouped.apply(
            lambda x: group_wmean(x, weighted=False))
        dir_result['set'] = dir_grouped.set.sum()
        dir_result['azimuths'] = 2
        dir_result.index.name = 'azimuth'
        dir_result = dir_result.reset_index(drop=False)

        self._directions = dir_result

        # Final results
        fin_result = pd.DataFrame(group_wmean(dir_result, weighted=False).to_dict(),
                                  index=[0])

        fin_result['set'] = dir_result.set.sum()
        fin_result['azimuths'] = dir_result.azimuths.sum()

        if len(self._directions) > 1:
            # t-statistics
            self._t_test = t_statistics(dir_result.g_result.values,
                    dir_result.err.values, dir_result.accepted.values)

            # if not passed
            if not self._t_test[-1]:
                d = dir_result.g_result.values[0] - dir_result.g_result.values[1]
                err = 0.5*np.sqrt(np.sum(dir_result.err.values**2) + d**2)
                fin_result['err'] = err

        self._final = fin_result
