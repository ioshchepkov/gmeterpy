#!/usr/bin/env python
# encoding: utf-8

"""Module containing parsers for PROLET program for earth tides predictions"""

import re
import io
from pandas import Series, read_csv

def prolet(fname, which = 'both'):

    data = ''
    with open(fname, 'r') as f:
        for line in f:
            if re.search('^\d+\s\d+', line.strip()):
                data += line

    col_names = ['ymd', 'hms', 'body_ocean', 'body', 'ocean']

    df = read_csv(io.StringIO(data),
            delim_whitespace = True,
            names = col_names,
            parse_dates = {'time':['ymd', 'hms']},
            index_col = 'time')
    #to uGal
    df = df / 10

    if which == 'both':
        return -Series(df['body_ocean'])
    elif which == 'solid':
        return -Series(df['body'])
    elif which == 'ocean':
        return -Series(df['ocean'])
