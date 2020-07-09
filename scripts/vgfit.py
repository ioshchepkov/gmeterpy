#!/usr/bin/env python
# encoding: utf-8

import argparse
import numpy as np
from pandas import read_csv

from gmeterpy.corrections.vgrad import fit_floating_gravity
from gmeterpy.plotting.vgfit import plot_fit
from gmeterpy.corrections.vgrad import generate_report

parser = argparse.ArgumentParser(description='Second-order polynomial fit of the vertical gravity gradients')
parser.add_argument('-i', metavar='in-file', type=argparse.FileType('rt'), required=True)
parser.add_argument('-n', '--name', default='')

#parser.add_argument('--plot', dest='plot', action='store_true')
#parser.add_argument('--no-plot', dest='plot', action='store_false')
#parser.set_defaults(plot=True)

opt = parser.parse_args()

data = read_csv(opt.i)

df, res = fit_floating_gravity(data, deg=2)
a, b = res.params[-2], res.params[-1]

se_a = res.bse[-2]
se_b = res.bse[-1]
covab = res.cov_params()['a']['b']
df['resid'] = res.resid

gp = np.poly1d([b, a, 0])

make_plot = True
make_report = True

station = opt.name

if make_plot:
    h_min = min(df.h)
    al = (gp(h_min) - gp(1.0)) / (h_min - 1.0)
    fig = plot_fit(df, lambda x: gp(x) - al*x, (se_a, se_b, covab), station)
    title = '{} ({:.1f} $\mu Gal/m$  substructed)'.format(station, al)
    fig.gca().set_title(title, fontsize=14)
    plot_file = station + '_all.png'
    fig.savefig(plot_file)

if make_report:
    report_file = 'vg_' + station + '_fit_all.txt'
    generate_report(report_file, data, res, gp, (se_a, se_b, covab), station)

