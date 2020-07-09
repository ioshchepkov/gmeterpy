#!/usr/bin/env python

import os
import argparse
import configparser
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from gmeterpy.core.relative import RelativeReadings
from gmeterpy.plotting.relative import plot_loop_processing
from gmeterpy.utils.stats import interpolate

from gmeterpy.corrections.tides.tamura import tide
from gmeterpy.corrections.tides.prolet import prolet
from gmeterpy.corrections.atmosphere import atmosphere_pressure_corr

conf_parser = argparse.ArgumentParser(description='Process relative measurements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)

conf_parser.add_argument("-c", "--conf_file",
                        help="specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()

defaults = {
        "tide_model": 'longman',
        "drift_order": 1,
        "tb": 0,
        "ta": 1,
        }

if args.conf_file:
    config = configparser.ConfigParser()
    config.read([args.conf_file])
    defaults.update(dict(config.items("ProjectParams")))

parser = argparse.ArgumentParser(
        parents=[conf_parser])
parser.set_defaults(**defaults)

parser.add_argument('file', help='input data file')

#parser.add_argument('--merge', action='append', default=[],
#        help='files to merge (stations, level, etc.)')

# Merge files
parser.add_argument('--stations', type=open,
        help='specify file with station inforamtion')

# Processing options
parser.add_argument('--tb', type=int, metavar='N',
        help='truncate N readings in the beginning of each setup')
parser.add_argument('--ta', type=int, metavar='N',
        help='truncate N readings in the end of each setup')
parser.add_argument('--drift-order', type=int,
        help='polynomial order for a drift function')

# Corrections
parser.add_argument('--tide-model', type=str,
        help='tide model to calculate tide correction')
parser.add_argument('--tide', type=open,
        help='file with tides')
parser.add_argument('--calibration', type=open,
        help='file with calibration cofficients')
parser.add_argument('--pressure', type=open,
        help='file with pressure measurements')

# Filter
#parser.add_argument('--filter', nargs=3,
#        help='filter data by values in columns')

# Plots
parser.add_argument('--plot-tide-diff', action='store_true',
        help='plot the difference between the old and the new tide models')

args = parser.parse_args(remaining_argv)

proc_dir = os.path.dirname(os.path.abspath(args.file))
os.chdir(proc_dir)

readings = RelativeReadings.from_file(args.file, use_drift=True)
suffix = '_line_' + str(int(readings.data.line.unique()[0]))

#if args.auto_sid:
#   readings = readings.auto_sid(2)

#merge_files = args.merge
#for f in merge_files:
#    to_merge = pd.read_csv(f)
#    readings = readings.merge(to_merge)

if args.stations is not None:
    to_merge = pd.read_csv(args.stations)
    readings = readings.merge(to_merge)
else:
    sid = readings.data.sid.values
    name = ['S' + str(x) for x in sid]
    readings = readings.merge(
            pd.DataFrame({'sid' : sid, 'name': name}))

# apply gravimeter calibration parameter
if args.calibration is not None:
    cparams = pd.read_csv(args.calibration, delim_whitespace=True).drop('stdev', axis=1)
    readings = readings.merge(cparams)
    readings.data['c_calibr'] = readings.data.g*(readings.data.coef - 1)
    readings.set_correction('c_calibr', 'c_calibr')

#apply tide model
tide_model = args.tide_model
#needed only if tide_model is 'atlantida' or 'ts'
if tide_model in ('atlantida', 'ts'):
    if args.tide is not None:
        tide_file = os.path.abspath(args.tide.name)
    else:
        raise Exception('You need to specify filename with tides')

#tide correction
if tide_model in ('tamura', 'atlantida', 'custom'):
    old = readings.data.c_tide.copy()
    if tide_model == 'tamura':
        d,t = readings.data, readings.data.index
        c_tide = list(map(tide, t.year, t.month, t.day, t.hour, t.minute, t.second, d.lon, d.lat, d.height))
        readings._data['c_tide'] = np.asarray(c_tide)/1000
    elif tide_model == 'atlantida':
        c_tide = prolet(tide_file)
        readings._data['c_tide'] = readings.interpolate_from_ts(c_tide)/1000
    elif tide_model == 'custom':
        tides = pd.read_csv(tide_file, index_col='time', parse_dates=True)
        for name, group in readings.data.groupby('name'):
            c_tide = interpolate(tides[tides.name == name]['tide'], group.index)/1000
            readings.data.loc[readings.data.name == name, 'c_tide'] = c_tide
    if args.plot_tide_diff:
        diff = readings.data.c_tide - old
        (diff - diff.mean()).plot()
        plt.savefig('tide_diff' + suffix + '.png', dpi=600, format='png')

readings.set_correction('c_tide', 'c_tide')

# pressure correction
if args.pressure is not None:
    pres = pd.read_csv(args.pressure, index_col='time', parse_dates=True)
    readings.data['pres'] = interpolate(pres['pres'], readings.data.index)
    readings.data['c_atm'] = (atmosphere_pressure_corr(readings.data.height,
        readings.data['pres'])/1000).round(4)
    readings.set_correction('c_atm', 'c_atm')
    readings.data.c_atm.plot()

loop = readings.copy().truncate(by='setup', before=args.tb, after=args.ta)

# filter
#loop = loop.filter('tilt_y', -20, 20)
#loop = loop.filter('tilt_x', -20, 20)

dur_med = readings.data.dur.median()
loop = loop.filter(dur_med, dur_med + 20, column='dur')

#loop = loop.filter('setup', 8, 17)
#loop = loop.filter('stdev', -1, 0.1)

adj = loop.fit(mask=False, order=args.drift_order)

adj.to_file('readings' + suffix + '.redu')

report = adj.drift.report()
with open('report' + suffix + '.txt', 'w') as f:
    f.write(report)

ties = adj.ties()
ties.to_file(fname='ties' + suffix + '.txt')

plot_loop_processing(readings, adj)
plt.savefig('plot' + suffix + '.png', dpi=600, format='png')
