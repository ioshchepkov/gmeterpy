#!/usr/bin/env python

import os
import argparse
import configparser

from gmeterpy.meters.gabl.core import GABLProject
from gmeterpy.plotting.absolute import plot_residuals, plot_drops
from gmeterpy.corrections.tides.prolet import prolet
from gmeterpy.corrections import get_polar_motion

conf_parser = argparse.ArgumentParser(description='Process GABL data',
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      add_help=False)

conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()

defaults = {
    "station": '',
    "station_id": '0000',
    "lat": None,
    "lon": None,
    "elev": None,
    "timedelta": 0,
    "baro_factor": 0.3,
    "meter_type": 'GBL-M',
    "meter_sn": '001',
    "tdb": 0,
    "tda": 0,
    "sigma": 3.0,
    "reject_by": 'series',
    "mean_by": 'series',
    "drop_model": 'OLS',
}

if args.conf_file:
    config = configparser.ConfigParser()
    config.read([args.conf_file])
    defaults.update(dict(config.items("ProjectParams")))

parser = argparse.ArgumentParser(
    # Inherit options from config_parser
    parents=[conf_parser]
)

parser.set_defaults(**defaults)

parser.add_argument('path', help='path to data', action='store')

# Station
parser.add_argument('--station', type=str,
                    help='human-readable station name')
parser.add_argument('--station-id', type=str,
                    help='computer-readable station id')
parser.add_argument('--lat', type=float,
                    help='latitude')
parser.add_argument('--lon', type=float,
                    help='longitude')
parser.add_argument('--elev', type=float,
                    help='height above sea level')
parser.add_argument('--timezone', type=float,
                    help='timedelta from UTC in hours')
parser.add_argument('--baro-factor', type=float,
                    help='barometric admittance factor, default is 0.3')

# Instrument
parser.add_argument('--meter-type', type=str,
                    help='type of the gravimeter')
parser.add_argument('--meter-sn', type=str,
                    help='s/n of the gravimeter')

# Corrections
parser.add_argument('--tide', type=argparse.FileType('r'), required=True)

# Processing
parser.add_argument('--tdb', type=int,
                    help='truncate drop in the beginning')
parser.add_argument('--tda', type=int,
                    help='truncate drop in the end')
parser.add_argument('--sigma', type=float,
                    help='rejection level, default is 3.0')
parser.add_argument('--reject-by', type=str,
                    help='reject drops in series, seances or azimuths, default is in series')
parser.add_argument('--mean-by', type=str,
                    help='find mean value in series, seances or azimuths, default is in series')
parser.add_argument('--drop-model', type=str,
                    help='regression model for drop fitting')

# Plots
parser.add_argument('--plot-residuals', action='store_true')


args = parser.parse_args(remaining_argv)

path = args.path
ftide = args.tide.name

before = args.tdb
after = args.tda

config = {
    'station': args.station,
    'sid': args.station_id,
    'baro_factor': args.baro_factor}

if args.lat is not None:
    config.update({'lat': float(args.lat)})
if args.lon is not None:
    config.update({'lon': float(args.lon)})
if args.elev is not None:
    config.update({'height': float(args.elev)})

proj = GABLProject.load(path, add_to_meta=config, timezone=args.timezone)

proj.proc_drops(before, after, model=args.drop_model)

os.chdir(path)

# corrections
proj.set_correction('c_tide', prolet, fname=ftide, which='solid')
proj.set_correction('c_ocean', prolet, fname=ftide, which='ocean')

# update polar motion coordinates
xp, yp = get_polar_motion(proj._data.jd.values)
proj.add_quantity_column('xp', xp)
proj.add_quantity_column('yp', yp)
proj._update_g_result()

if args.reject_by.lower() == 'series':
    by = ['azimuth', 'seance', 'series']
elif args.reject_by.lower() == 'seance':
    by = ['azimuth', 'seance']
elif args.reject_by.lower() == 'azimuth':
    by = ['azimuth']

proj = proj.filter(by=by, sigma=args.sigma)

if args.mean_by.lower() == 'series':
    by = ['azimuth', 'seance', 'series']
elif args.mean_by.lower() == 'seance':
    by = ['azimuth', 'seance']
elif args.mean_by.lower() == 'azimuth':
    by = ['azimuth']

proj = proj.mean(by=by)
proj._set.proc()

proj.report()

'''
fig2 = plot_drops(group.g_result.values)
fig_name2 = '_'.join(str(x) for x in idx)
fig2.figure.savefig('plot/drops_' + fig_name + '.png')
'''
if args.plot_residuals:
    if not os.path.exists('plot'):
        os.makedirs('plot')
    for fig_name, fig in proj.plot_residuals(by=by):
        fig.savefig('plot/' + fig_name + '.png')
