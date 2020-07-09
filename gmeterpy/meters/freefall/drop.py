#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import statsmodels.api as sm

import gmeterpy.units as u
from gmeterpy.meters.freefall.effective_height import free_fall_effective_measurement_height


class FreeFallDrop:
    def __init__(self, time, distance, **kwargs):
        self._time = time
        self._distance = distance
        self._meta = kwargs.pop('meta', None)

    @classmethod
    def from_random(cls):
        pass

    @property
    def time(self):
        """Return time intervals.

        """
        return self._time

    @property
    def distance(self):
        """Return distance intervals.

        """
        return self._distance

    def truncate(self, before=0, after=0):
        after = -after if after != 0 else len(self.time)
        t = self.time[before:after]
        dt = t[before] - t[0]

        self._time = t - dt
        self._distance = self.distance[before:after]

        return self

    def fit(self, model='OLS', gradient=0.0, **kwargs):

        time = self.time
        time_sq = time**2

        # derivatives
        z0 = 1 + 0.5 * gradient * time_sq
        v0 = time * (1 + 1 / 6 * gradient * time_sq)
        g0 = 0.5 * time_sq * (1 + 1 / 12 * gradient * time_sq)

        # design matrix
        X = np.column_stack((v0, g0, z0))

        if model == 'OLS':
            res = sm.OLS(self.distance, X, **kwargs).fit()
        elif model == 'RLM':
            M = kwargs.pop('M', sm.robust.norms.AndrewWave())
            res = sm.RLM(self.distance, X, M=M).fit()

        return FreeFallDropResults(self, res)


class FreeFallDropResults:
    def __init__(self, model, stats_results):

        self._model = model
        self._stats_results = stats_results

        self._results = {}

        self._results['g0'] = stats_results.params[1] * 1e8
        self._results['err'] = stats_results.bse[1] * 1e8
        self._results['v0'] = stats_results.params[0]
        self._results['z0'] = stats_results.params[2]
        self._results['drop_duration'] = self._model.time[-1]

        h_eff = free_fall_effective_measurement_height(
            self.results['v0'] * u.m / u.s,
            self.results['drop_duration'] * u.s)
        # effective height - legacy
        self._results['h1'] = h_eff.value

    @property
    def results(self):
        return self._results

    @property
    def model(self):
        return self._model

    @property
    def stats_results(self):
        return self._stats_results

    def plot_residuals(self):
        pass
