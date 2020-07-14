#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import networkx as nx

from scipy.cluster.vq import kmeans, whiten, vq

from gmeterpy.core.readings import Readings
from gmeterpy.core.drift import Drift
from gmeterpy.core.dmatrices import (dmatrix_ties,
        dmatrix_relative_gravity_readings)


def closures(df, root=None):
    """Closures analysis in the network"""

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
        idx = np.concatenate(([0], np.where(self.data['sid'][:-1].values !=
                                            self.data['sid'][1:].values)[0] + 1, [len(self.data)]))

        rng = [(a, b) for a, b in zip(idx, idx[1:])]
        setup = []
        for i in range(len(rng)):
            l, r = rng[i]
            app = np.ones(r - l) * i
            setup = np.append(setup, app)

        self._data['setup'] = setup.astype('int') + 1

        return self

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
        data = self._data.copy()
        rep = data.groupby('name').setup.unique().apply(len) > 1
        rep = rep.reset_index()
        rep.columns = ['name', 'in_repeated']
        data = data.reset_index().merge(rep).set_index('time').sort_index()
        mask = data.in_repeated.values
        return mask

    def truncate(self, by=None, before=0, after=0):
        # if before is None:
        #    before = self._proc['truncate_before']
        # if after is None:
        #    after = self._proc['truncate_after']

        # TODO: add default by

        data = self._data.reset_index()
        data = data.groupby(by).apply(lambda x: x.iloc[before:(len(x) -
                                                               after)]).reset_index(drop=True)
        self._data = data.set_index('time')

        self._proc['truncate_before'] = before
        self._proc['truncate_after'] = after

        return self

    def split(self, by=None):
        splitted = []
        for n, group in self._data.groupby(by):
            l = RelativeReadings(group.copy(), meta=self.meta)
            splitted.append(l)
        return splitted

    def dmatrices(self, w_col=None, **kwargs):
        dm = dmatrix_relative_gravity_readings(self.data.copy(), **kwargs)

        if w_col is not None:
            wm = np.diag(self.data[w_col])
        else:
            wm = np.identity(len(dm))

        y = np.asmatrix(self.data.g_result.copy()).T

        return dm, wm, y

    def fit(self, *args, **kwargs):

        mask = kwargs.pop('mask', False)
        set_corr = kwargs.pop('set_corr', True)

        readings = self.copy()

        if mask:
            readings._data = readings.data[self.get_repeated_mask()].copy()

        self.drift = Drift(readings, *args, **kwargs)
        self.res = self.drift.res
        self.data['dt0'] = self.data.jd - self.drift.t0

        #self.meta['proc']['drift_order'] = self.drift.drift_order

        self._data['c_drift'] = np.around(self.drift.drift(self.data.dt0), 4)

        if set_corr:
            self.set_correction('c_drift', 'c_drift')

        return self

    def has_ties(self):
        if len(self.stations()) < 2:
            return False
        else:
            return True

    def ties(self, ref=None, sort=False):
        stations = self.stations()

        if not self.has_ties():
            print('Warning: You have only one station. Nothing to tie with')
            return Ties()

        if not hasattr(self, 'res'):
            raise Exception('You need to fit your measurements first!')

        adjg = pd.DataFrame({'g': self.res.params[stations], 'stdev':
                             self.res.bse[stations]})
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

        ties['date'] = self.data.index.date[0].strftime('%Y-%m-%d')
        ties['meter_sn'] = self.data.meter_sn.unique()[0]
        ties['operator'] = self.data.operator.unique()[0]

        count = self.data.groupby('name').setup.unique()

        for index, row in ties.iterrows():
            name1 = row['from']
            name2 = row['to']

            var1 = self.drift.res.bse[name1]**2
            var2 = self.drift.res.bse[name2]**2
            covar = self.drift.res.cov_params()[name1][name2]
            stdev = np.sqrt(var1 + var2 - 2 * covar)

            ties.loc[index, 'stdev'] = stdev
            ties.loc[index, 'n'] = min(len(count[name2]), len(count[name1]))

        return Ties(ties)


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
            out = closures(df.delta_g.mean().reset_index(
                drop=False), root=root)
        else:
            out = {}
            for i, group in self.data.groupby(by):
                df = group.groupby(['from', 'to'])
                cl = closures(df.delta_g.mean().reset_index(
                    drop=False), root=root)
                if cl:
                    out[str(i)] = cl

        return out
