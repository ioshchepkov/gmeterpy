#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from string import ascii_lowercase
from statsmodels.api import WLS
from statsmodels.tools.tools import categorical


def vert_grad_corr(a, b, h1, h2):
    return (a + b * (h1 + h2)) * (h2 - h1)


def poly_uncertainty_eval(h1, h2, sa, sb, covab):
    u = abs(h2 - h1) * np.sqrt(sa**2 + (h2 + h1)
                               ** 2 * sb**2 + 2 * (h2 + h1) * covab)
    return u


def fit_floating_gravity(data, deg=2, **kwargs):
    """Fit floating gravity model to the observations.

    """

    # transform data
    df = pd.DataFrame({'g':
                       np.concatenate(
                           (np.zeros_like(data.delta_g), data.delta_g)),
                       'h': np.concatenate((data['level_1'], data['level_2'])) / 1000,
                       'ci': np.tile(data.runn, 2)})

    df = df.drop_duplicates(['ci', 'h', 'g'])
    df = df.sort_values(['ci', 'g'], ascending=[True, False])
    df = df.reset_index(drop=True)

    # observations
    endog = np.asarray(df.g)

    # design matrix
    exog_1 = np.vander(df.h.values, N=deg+1, increasing=True)[:, 1:]
    exog_2 = categorical(df.ci.values, drop=True)
    exog = np.hstack((exog_2, exog_1))

    # rename unknowns
    h_level_1 = data.drop_duplicates(['level_1', 'runn']).level_1
    h0 = ['h({:,.3f})'.format(hi / 1000) for hi in np.asarray(h_level_1)]
    poly_cnames = [x for x in ascii_lowercase[:deg]]
    cnames = np.append(h0, poly_cnames)
    exog = pd.DataFrame(exog, columns=cnames)

    # fit
    results = WLS(endog, exog, **kwargs).fit()

    return df, results


def fit_gravity_differences(data, deg=2, **kwargs):
    """Polynomial fit of the gravity differences.

    """

    # endog
    endog = np.asarray(data.delta_g)

    # design_matrix
    exog_1 = np.vander(data.level_1.values, N=deg+1, increasing=True)[:, 1:]
    exog_2 = np.vander(data.level_2.values, N=deg+1, increasing=True)[:, 1:]
    exog = exog2 - exog1

    # rename unknowns
    poly_cnames = [x for x in ascii_lowercase[:deg]]
    exog = pd.DataFrame(exog, columns=poly_cnames)

    # fit
    results = WLS(endog, exog, **kwargs).fit()

    return results


def fit_gravity(data, deg=2, **kwargs):
    """Polynomial fit of the gravity values.

    """

    # endog
    endog = np.asarray(data.g)

    # design_matrix
    exog = np.vander(data.level.values, N=deg+1, increasing=True)[:, 1:]

    # rename unknowns
    poly_cnames = [x for x in ascii_lowercase[:deg]]
    exog = pd.DataFrame(exog, columns=poly_cnames)

    # fit
    results = WLS(endog, exog, **kwargs).fit()

    return results


def generate_report(report_file, data, res, gp, gu, station):
    """Generate text report of the fit.

    """

    se_a, se_b, covab = gu
    h_control = np.array([0.000, 0.05, 0.130, 0.270,
                          0.700, 0.720, 0.900, 1.000, 1.200, 1.300, 1.4])
    with open(report_file, 'w') as f:
        print('Results of the second-order polynomial fit of the vertical gravity gradients at '
              + station + '\n', file=f)
        f.write('Data:\n')
        f.write(5 * '*' + '\n')
        f.write('{}\n\n'.format(data.to_string(index=False, na_rep='---')))
        f.write('Fit summary:\n' + 12 * '*' + '\n')
        f.write('{0:2.0f} observations -\n'.format(res.nobs))
        f.write('{0:2.0f} parameters =\n'.format(len(res.params)))
        f.write('{0:2.0f} degrees of freedom\n\n'.format(res.df_resid))

        f.write('Sum of squared residuals: {}\n\n'.format(res.ssr))

        pd.options.display.float_format = '{:.2f}'.format
        f.write(res.summary2(alpha=0.05).tables[1].to_string() + '\n\n')

        f.write('Covariance matrix\n')
        f.write(res.cov_params().to_string(
            float_format='{:.6f}'.format) + '\n\n')

        f.write('Prediction table:\n')
        f.write(18 * '*' + '\n')
        f.write('{:17}\t{:^}\t\t{:>}\n'.format(
            'Heights', 'Gravity diffs', 'Gradients'))
        for hc in h_control:
            ha = h_control[np.where(h_control > hc)]
            for h1, h2 in zip(np.repeat(hc, len(ha)), ha):
                dg = gp(h2) - gp(h1)
                se_dg = poly_uncertainty_eval(h1, h2, se_a, se_b, covab)
                grad = dg / (h2 - h1)
                se_grad = se_dg / abs(h2 - h1)
                f.write(('{:.3f} --> {:.3f} m,\t{:6.1f} +/- {:.1f} uGal,' +
                         '\t{:6.1f} +/- {:.1f} uGal/m\n').format(h1, h2, dg, se_dg, grad, se_grad))
            f.write('\n')
