#!/usr/bin/env python
# encoding: utf-8

import os
import fnmatch
import struct
import configparser
from datetime import timedelta, datetime
import numpy as np
import pandas as pd

from gmeterpy.meters.freefall.drop import FreeFallDrop


def read_binary_drop(fname, dist_ivals=False, full=False, timezone=0):
    with open(fname, 'rb') as f:
        # skip first four symbols
        f.seek(4)
        # parse options and metadata
        opt = struct.unpack('diidfff', f.read(36))

        ff = f.read()
        # parse data
        data = np.frombuffer(ff, dtype=np.dtype('u1'))
        data2 = np.frombuffer(ff, dtype=np.dtype('u4'))

    xtra, high, mid, low = np.asarray(data.reshape(int(len(data) / 4), 4).T,
            dtype=int)

    opt = {'time': datetime(1899, 12, 30) + timedelta(days=opt[0]) -
            timedelta(hours=timezone),
            'npoints': opt[1],
            'divider': opt[2],
            'wavelength': opt[3] * 10**-6,
            'vac': opt[4],
            'pres': opt[5],
            'temp': opt[6]}

    if not 600 < opt['pres'] < 800:
        del opt['pres']
        del opt['temp']

    # norming
    xtra_norm = np.asarray(255 / (xtra.max() - xtra.min()) *
            (xtra - xtra.min()), dtype=int)

    # take into account that counter is 24 bit size
    shift_idx = np.where(np.ediff1d(low) < 0)[0] + 1
    lenlow = len(low)
    shift_idx = np.concatenate(([0], shift_idx, [lenlow]))

    shift = np.zeros(lenlow, dtype=int)
    k = 16777216  # 2**24
    shift2 = [int(hex(x * k)[2:] + '00', 16) for x in range(len(shift_idx) - 1)]
    for i in range(len(shift_idx) - 1):
        shift[shift_idx[i]:shift_idx[i + 1]] = shift2[i]

    counter_xtra = data2 - xtra + shift - xtra_norm

    # calculate time and distance intervals in sec and meters
    intervals = counter_xtra - counter_xtra[0]
    time_intervals = intervals / (384e8)  # 150MHz*256

    if dist_ivals or full:
        index = np.arange(intervals.size)
        distance_intervals = opt['divider'] * index * \
                opt['wavelength'] / 2

    if full:
        df = pd.DataFrame({
            'low': low,
            'mid': mid,
            'high': high,
            'xtra': xtra,
            'xtra_norm': xtra_norm,
            'counter_xtra': counter_xtra})

        df['first_diff_counter_xtra'] = df.counter_xtra.diff()
        df['second_diff_counter_xtra'] = df.first_diff_counter_xtra.diff()
        df['time_intervals'] = time_intervals
        df['distance_intervals'] = distance_intervals
        return df
    elif dist_ivals:
        return time_intervals, distance_intervals, opt
    else:
        return time_intervals, opt


def read_seanceini(root):
    config = configparser.ConfigParser()
    config.read(os.path.join(root, 'seance.ini'))
    params = config['SeanceParams']

    df = {
            'seance_name': params['SeanceName'],
            'station': params['LocateName'],
            'meter_height': float(params['GravimeterHeight']),
            'aero_coeff': float(params['CorrectK']),
            'xp': float(params['X_Polus']),
            'yp': float(params['Y_Polus']),
            'lat': float(params['Latitude']),
            'lon': float(params['Longitude']),
            'height': float(params['Altitude']),
            'pres': float(params['ActualPressure']),
            'temp': float(params.get('Temperature', np.nan)),
            'baro_factor': 0.4,
            'accepted': bool(True)}

    return df


def load_from_path(path, add_to_meta=None, timezone=0):
    snc_paths = []
    for root, _, files in os.walk(path):
        if any(f for f in files if fnmatch.fnmatch(f, 'seance.ini')):
            snc_paths.append(root)

    drops = []
    for snc_n, snc_path in enumerate(snc_paths):
        meta = read_seanceini(snc_path)
        # TODO: add units
        if add_to_meta is not None:
            meta.update(add_to_meta)
        name = seance_name_handler(meta['seance_name'])
        #meta['seance_number'] = int(snc_n)
        meta['seance'] = os.path.basename(snc_path).replace(' ', '_')
        meta['azimuth'] = name

        for root, _, files in os.walk(snc_path):
            for f in files:
                if fnmatch.fnmatch(f, 'SingleDrop*'):
                    nn = int(f.lstrip('SingleDrop'))
                    srs_n = int(os.path.basename(root).lstrip('Series'))

                    meta = meta.copy()
                    meta['nn'] = nn
                    meta['series'] = srs_n

                    time, distance, drop_meta = read_binary_drop(
                            os.path.join(root, f), dist_ivals=True,
                            timezone=timezone)
                    meta.update(drop_meta)

                    drop = FreeFallDrop(time, distance, meta=meta)
                    if len(drop.time) == (drop._meta['npoints']):
                        drops.append(drop)

    drops_sorted = sorted(drops, key=lambda x: x._meta['time'])
    return drops_sorted

def seance_name_handler(name):
    s = name.lower()
    if any(map(s.__contains__, ['north', 'север'])):
        return 'n'
    elif any(map(s.__contains__, ['south', 'юг'])):
        return 's'
    elif any(map(s.__contains__, ['east', 'восток'])):
        return 'e'
    elif any(map(s.__contains__, ['west', 'запад'])):
        return 'w'
