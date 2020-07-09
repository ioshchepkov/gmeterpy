#!/usr/bin/env python
# coding: utf-8

"""Convert Scintrex CG-x txt file to table txt"""

import argparse

import os
from gmeterpy.meters.scintrex import ScintrexCG5, ScintrexCG6

parser = argparse.ArgumentParser(description='Convert Scintrex CG-x file')
parser.add_argument('infile', metavar='FILE', type=argparse.FileType('rt'),
        help='Input Scintrex CG-x data file')
parser.add_argument('--meter-type',
        default='cg5', choices=['cg5', 'cg6'],
        help='type of the gravimeter')
parser.add_argument('--remove-drift', action='store_false',
        help='Remove linear drift defined as Drift value in SETUP PARAMETERS')
args = parser.parse_args()

proc_dir = os.path.dirname(os.path.abspath(args.infile.name))
os.chdir(proc_dir)

if args.meter_type.lower() == 'cg5':
    survey = ScintrexCG5(args.infile.name, use_drift=args.remove_drift)
elif args.meter_type.lower() == 'cg6':
    survey = ScintrexCG6(args.infile.name, use_drift=args.remove_drift)

#show lines info
for line, data in survey.data.groupby('line'):
    print('line:', line)
    print('\tsid:\t', data.sid.unique())
    print('\tsetup:\t', data.setup.unique())

# write lines in separate files
#lines = survey.lines().copy()
#for i, line in enumerate(lines):
#    suffix = '_line_' + str(int(line.data.line.unique()[0]))
#    line.to_file('readings' + suffix + '.txt')

survey.to_file('readings.txt')
