#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm
from scipy.stats import chi2


class AdjustmentResults:
    def __init__(self, model, results):

        self.model = model
        self.res = results

        self._s0 = np.sqrt(self.res.scale)

    @property
    def number_of_observations(self):
        return self.res.nobs

    @property
    def number_of_unknowns(self):
        return self.res.df_model

    @property
    def degrees_of_freedom(self):
        return self.res.df_resid

    @property
    def params(self):
        return self.res.params

    def _params_contains(self, *args, **kwargs):
        return self.params.index.str.contains(*args, **kwargs)

    def _params_startswith(self, *args, **kwargs):
        return self.params.index.str.startswith(*args, **kwargs)

    def _params_endswith(self, *args, **kwargs):
        return self.params.index.str.endswith(*args, **kwargs)

    def has_params(self, name):
        return np.any(self._params_contains(name))

    @property
    def sigma(self):
        return self.res.bse

    @property
    def s0(self):
        return self._s0

    def chi2(self, alpha=0.95, scale=1.0):
        dof = self.degrees_of_freedom
        lower, upper = np.asarray(chi2.interval(alpha=alpha, df=dof))
        chisq = self.s0**2/scale**2 * dof
        return chisq, lower, upper

    def qqplot(self, ax=None, **kwargs):
        """Q-Q plot of the residuals.

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))

        sm.graphics.qqplot(self.res.resid, fit=True, line='s', ax=ax, **kwargs)
        ax.set_title('Normal Q-Q plot')
        return ax
