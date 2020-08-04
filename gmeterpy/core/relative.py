#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import networkx as nx

import statsmodels.formula.api as smf
import statsmodels.api as sm

from scipy.cluster.vq import kmeans, whiten, vq

from gmeterpy.core.readings import Readings
from gmeterpy.core.adjustment import AdjustmentResults
from gmeterpy.core.dmatrices import (dmatrix_ties,
        dmatrix_relative_gravity_readings)


def _closures(df, root=None):
    """Closures analysis in the network.

    """

    network = nx.from_pandas_edgelist(df, 'from', 'to',
            edge_attr='delta_g',
            create_using=nx.DiGraph())

    basis = nx.cycle_basis(network.to_undirected(), root=root)
    out = []
    for closure in basis:
        closure_sum = 0
        for node1, node2 in zip(closure, closure[1:] + closure[:1]):
            if network.has_edge(node1, node2):
                dg = network[node1][node2]['delta_g']
            else:
                dg = -network[node2][node1]['delta_g']
            closure_sum += dg
        out.append((closure, round(closure_sum, 4)))

    return out


class RelativeReadings(Readings):

    def __init__(self, *args, **kwargs):

        auto_sid = kwargs.pop('auto_sid', False)
        self.auto_setup_id = kwargs.pop('auto_setup_id', False)
        nos = kwargs.pop('number_of_stations', None)

        super().__init__(*args, **kwargs)

        if auto_sid and nos is not None:
            self.auto_station_id(nos)

        self.setup_id()

        # TODO: auto_loop
        if 'loop' not in self._data.columns:
            self._data['loop'] = 1

    def stations(self):
        return self.data.name.unique()

    def rgmeters(self):
        return self.data.meter_sn.unique()

    def auto_sid(self, number_of_stations):
        whitened = whiten(np.asarray(self.data['g_result']))
        codebook, _ = kmeans(whitened, number_of_stations, iter=100)
        code, _ = vq(whitened, np.sort(codebook[::-1]))
        self._data['sid'] = code
        self.setup_id()

        return self

    def setup_id(self):
        #TODO: by loop
        idx = np.concatenate(([0], np.where(self.data['sid'][:-1].values !=
            self.data['sid'][1:].values)[0] + 1,
            [len(self.data)]))

        rng = [(a, b) for a, b in zip(idx, idx[1:])]
        setup = []
        for i in range(len(rng)):
            l, r = rng[i]
            app = np.ones(r - l) * i
            setup = np.append(setup, app)

        self._data['setup'] = setup.astype('int') + 1

        return self

    def auto_loop(self):
        raise NotImplementedError

    @classmethod
    def from_file(self, fname, **kwargs):
        def parser(x): return datetime.datetime.strptime(
                x, '%Y-%m-%d %H:%M:%S')

        df = pd.read_csv(fname, delim_whitespace=True, parse_dates=[
            ['date', 'time']], index_col=0, date_parser=parser)
        df.index.name = 'time'

        return RelativeReadings(data=df)

    def to_file(self, *args, **kwargs):
        kwargs['before'] = ['sid', 'meter_sn']
        kwargs['after'] = ['stdev']
        super().to_file(*args, **kwargs)

    def get_repeated_mask(self):
        #TODO: return not only mask, but RelativeReadings
        #TODO: by loop
        data = self._data.copy()
        rep = data.groupby('name').setup.unique().apply(len) > 1
        rep = rep.reset_index()
        rep.columns = ['name', 'in_repeated']
        data = data.reset_index().merge(rep).set_index('time').sort_index()
        mask = data.in_repeated.values
        return mask

    def dmatrices(self, w_col=None, **kwargs):
        dm = dmatrix_relative_gravity_readings(self.data.copy(), **kwargs)

        if w_col is not None:
            wm = np.diag(self.data[w_col])
        else:
            wm = np.identity(len(dm))

        y = np.asmatrix(self.data.g_result.copy()).T

        return dm, wm, y

    def adjust(self, gravity=True, drift_args={'drift_order':1},
            sm_model=sm.RLM, sm_model_args={'M':sm.robust.norms.HuberT()},
            **kwargs):
        """Least squares adjustment of the relative readings.

        """

        # t0 = readings.data.jd.min()
        # readings._data['dt0'] = readings.data.jd - t0

        # design matrix
        dm, _ , y = self.dmatrices(
                gravity=gravity,
                drift_args=drift_args,
                **kwargs)

        res = sm_model(y, dm, **sm_model_args).fit()

        #readings.meta['proc']['t0'] = t0
        #readings._meta.update({'proc': {
        #    'drift_args' : drift_args}})

        return RelativeReadingsResults(self, res)


class RelativeReadingsResults(AdjustmentResults):

    def __init__(self, readings, results):

        super().__init__(readings, results)

        self.readings = self.model

        #self.order = self.readings._meta['proc']['drift_order']
        #self.scale = scale

        #self.t0 = self.readings.data.jd.min()
        #self.readings._data['dt0'] = self.readings.data.jd - self.t0

        #self.readings._data['c_drift'] = np.around(
        #self.drift(self.readings.data.dt0), 4)
        #self.readings._data['resid'] = self.res.resid.values
        #self.readings._data['weights'] = self.res.weights.values

    def drift(self):
        drift_params = self.res.params[
                self.res.params.index.str.startswith('drift')]

        coefs = np.append(self.res.params[-self.order:][::-1], 0)
        return -np.poly1d(coefs, variable='t')

    def has_ties(self):
        if len(self.readings.stations()) < 2:
            return False
        else:
            return True

    def ties(self, ref=None, sort=False):
        stations = self.readings.stations()

        if not self.has_ties():
            print('Warning: You have only one station. Nothing to tie with')
            return Ties()

        adjg = pd.DataFrame({
            'g': self.res.params[stations],
            'stdev': self.res.bse[stations]
            })
        if sort:
            if isinstance(sort, bool):
                adjg = adjg.sort_index()
            elif isinstance(sort, list):
                adjg = adjg.loc[sort]

        if ref is None:
            from_st = adjg.index.values[:-1]
            to_st = adjg.index.values[1:]
            delta_g = (adjg.g.shift(-1) - adjg.g).values[:-1]
        elif isinstance(ref, str):
            if ref not in stations:
                raise Exception('Station {} does not exist.'.format(ref))
            else:
                from_st = ref
                to_st = adjg[adjg.index != ref].index.values
                delta_g = (adjg.loc[to_st].g - adjg.loc[from_st].g).values
        elif isinstance(ref, list):
            from_st, to_st = [p for p in zip(*ref)]
            delta_g = [adjg.loc[p2].g - adjg.loc[p1].g for p1,
                    p2 in zip(from_st, to_st)]

        ties = pd.DataFrame({
            'from': from_st,
            'to': to_st,
            'delta_g': delta_g,
            })

        ties['date'] = self.readings.data.index.date[0].strftime('%Y-%m-%d')
        ties['meter_sn'] = self.readings.data.meter_sn.unique()[0]
        ties['operator'] = self.readings.data.operator.unique()[0]

        count = self.readings.data.groupby('name').setup.unique()

        for index, row in ties.iterrows():
            name1 = row['from']
            name2 = row['to']

            var1 = self.res.bse[name1]**2
            var2 = self.res.bse[name2]**2
            covar = self.res.cov_params()[name1][name2]
            stdev = np.sqrt(var1 + var2 - 2 * covar)

            ties.loc[index, 'stdev'] = stdev
            ties.loc[index, 'n'] = min(len(count[name2]), len(count[name1]))

        return Ties(ties)

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


class Ties:

    def __init__(self, df=None):

        self.print_cols = ['from', 'to', 'date',
                'meter_sn', 'operator', 'delta_g', 'stdev']

        if df is not None:
            self._data = df
        else:
            self._data = pd.DataFrame(columns=self.print_cols)
            #df['meter_sn'] = df.meter_sn.astype(str)

        # sort from and to
        from_to = self._data[['from', 'to']].values
        data = self._data[(from_to != np.sort(from_to))[:, 0]]

        self._data.drop(data.index, inplace=True)
        data = data.rename(index=str, columns={'from': 'to', 'to': 'from'})
        data['delta_g'] = -data.delta_g
        self._data = self._data.append(data, sort=True)[
                self.print_cols].sort_values(['from', 'to'])

    def copy(self):
        return deepcopy(self)

    @property
    def data(self):
        return self._data

    @classmethod
    def from_file(self, fname):
        df = pd.read_csv(fname, delim_whitespace=True, parse_dates=[2])
        return Ties(df=df)

    def to_file(self, fname='ties.txt'):
        pd.options.display.float_format = '{:.4f}'.format
        with open(fname, 'w') as f:
            f.write(self.__str__() + '\n')

    @classmethod
    def load_from_path(self, path, pattern='ties*txt'):
        import os
        import fnmatch
        df = pd.DataFrame()
        for root, _, files in os.walk(path):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    tie = Ties.from_file(os.path.join(root, f))
                    df = df.append(tie.data, ignore_index=True)

        return Ties(df)

    def __str__(self):
        pd.options.display.float_format = '{:.4f}'.format
        #self._data['n'] = self.data.n.map('{:.0f}'.format)
        return self._data.reset_index()[self.print_cols].to_string(index=False)

    def stations(self):
        return np.unique(np.append(self.data['from'].values,
            self.data['to'].values))

    def rgmeters(self):
        return self.data.meter_sn.unique()

    def dmatrices(self, w_col=None, **kwargs):
        dm = dmatrix_ties(self, **kwargs)
        if w_col is not None:
            wm = np.diag(self.data[w_col])
        else:
            wm = np.identity(len(dm))

        y = np.asmatrix(self.data.delta_g).T

        return dm, wm, y

    def closures(self, by=None, root=None):
        """Closures analysis in the network"""

        if by is None:
            df = self.data.groupby(['from', 'to'])
            out = _closures(df.delta_g.mean().reset_index(
                drop=False), root=root)
        else:
            out = {}
            for i, group in self.data.groupby(by):
                df = group.groupby(['from', 'to'])
                cl = _closures(df.delta_g.mean().reset_index(
                    drop=False), root=root)
                if cl:
                    out[str(i)] = cl

        return out
