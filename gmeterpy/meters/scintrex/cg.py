#!/usr/bin/env python

import re
import numpy as np
import pandas as pd
from io import StringIO
from gmeterpy.core.relative import RelativeReadings


_CG6_HEADER = {
    'survey': 'Survey Name',
    'meter_sn': 'Instrument Serial Number',
    'operator': 'Operator',
    'drift': 'Drift Rate.*',
    'drift_start': 'Drift Zero Time',
}

_CG6_COLUMNS = [
    'sid', 'date', 'time', 'g_result', 'line',
    'stdev', 'stderr', 'g', 'tilt_x', 'tilt_y',
    'in_temp', 'c_tide', 'c_tilt', 'c_temp',
    'c_drift', 'dur', 'instr_height',
    'lat_user', 'lon_user', 'elev_user',
    'lat_gps', 'lon_gps', 'elev_gps', 'corrections']

_CG6_COLUMNS_DTYPE = {
    'sid': str, 'date': str, 'time': str, 'g_result': float, 'line': str,
    'stdev': float, 'stderr': float, 'g': float, 'tilt_x': float,
    'tilt_y': float,
    'in_temp': float, 'c_tide': float, 'c_tilt': float, 'c_temp': float,
    'c_drift': float, 'dur': float, 'instr_height': float,
    'lat_user': float, 'lon_user': float, 'elev_user': float,
    'lat_gps': float, 'lon_gps': float, 'elev_gps': float, 'corrections': str
}

_CG6_COLUMNS_EXCLUDE = ['corrections', 'g_result']

_CG5_HEADER = {
    'survey': 'Survey name',
    'meter_sn': 'Instrument S/N',
    'operator': 'Operator',
    'drift': 'Drift',
    'drift_start_time': 'DriftTime Start',
    'drift_start_date': 'DriftDate Start',
    'lon': 'LONG',
    'lat': 'LAT',
    'gmt': 'GMT DIFF.',
    'longman': 'Tide Correction',
    'tiltc': 'Cont. Tilt',
    'seism': 'Seismic Filter',
}

_CG5_COLUMNS = [
    'line', 'sid', 'out_temp', 'g',
    'stdev', 'tilt_x', 'tilt_y',
    'in_temp', 'c_tide', 'dur', 'rej', 'time',
    'dectime', 'terr', 'date']

_CG5_COLUMNS_DTYPE = {
    'line': str, 'sid': str, 'out_temp': float, 'g': float,
    'stdev': float, 'tilt_x': float, 'tilt_y': float,
    'in_temp': float, 'c_tide': float, 'dur': int, 'rej': int, 'time': str,
    'dectime': float, 'terr': float, 'date': str
}

_CG5_COLUMNS_EXCLUDE = ['terr', 'dectime']


def _parse_header(line, header_fields):
    for key, value in header_fields.items():
        pattern = re.compile(
            r'({0}):\s+(?P<{1}>.*)'.format(value, key),
            re.IGNORECASE)
        match = pattern.search(line)
        if match:
            return key, match
    return None, None


class _ScintrexCGBase(RelativeReadings):
    def __init__(self, filename, use_drift=True, use_tide=True, **kwargs):

        _header = {}
        data = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or 'Line' in line:
                    continue
                if line.startswith('/'):
                    key, match = _parse_header(line, self._header_fields)
                    self._header_parser(_header, key, match)
                else:
                    line = dict(zip(self._data_columns,
                                    line.replace('--', 'nan').split()))
                    data.append({**line, **_header})

        df = pd.DataFrame(data).astype(self._data_columns_dtype)
        df.drop(self._exclude_columns, axis=1, inplace=True)

        df['datetime'] = pd.to_datetime(df['date'] + " " + df['time'])
        df = df.set_index('datetime')

        if 'corrections' not in kwargs.keys():
            kwargs['corrections'] = self._default_corrections

        # drift correction
        delta_t = (df.index - df.drift_start) / pd.to_timedelta(1, unit='D')
        df['c_drift'] = np.around(
            -delta_t.values * df.drift.values, decimals=4)

        if hasattr(self, '_post_processing'):
            df = self._post_processing(df)

        if use_drift:
            kwargs['corrections']['c_drift'] = ('c_drift', {})
        else:
            df = df.drop(['c_drift'], 1)

        if use_tide:
            kwargs['corrections']['c_tide'] = ('c_tide', {})
        else:
            df = df.drop(['c_tide'], 1)

        df = df.drop(['drift', 'drift_start', 'time', 'date'], 1)

        super().__init__(df, **kwargs)

    def _header_parser(self, header, key, match):
        if key in ('survey', 'operator', 'meter_sn'):
            header[key] = match.group(key).lower()
        elif key in ('drift', 'gmt'):
            header[key] = float(match.group(key))
        elif key in ('lat', 'lon'):
            value, semisph = match.group(key).strip().split()
            header[key] = float(value)
            if semisph in ('W', 'S'):
                header[key] *= -1
        elif key == 'drift_start_time':
            header['drift_start'] = pd.to_timedelta(match.group(key))
        elif key == 'drift_start_date':
            header['drift_start'] += pd.to_datetime(match.group(key))
        elif key == 'drift_start':
            header[key] = pd.to_datetime(match.group(key))
        elif key in ('longman', 'tiltc', 'seism'):
            header[key] = False
            if match.group(key) == 'YES':
                header[key] = True


class ScintrexCG5(_ScintrexCGBase):
    instrument_name = 'Scintrex CG-5'

    _default_corrections = {
        'c_tide': ('c_tide', {})}

    _header_fields = _CG5_HEADER

    _data_columns = _CG5_COLUMNS

    _data_columns_dtype = _CG5_COLUMNS_DTYPE

    _exclude_columns = _CG5_COLUMNS_EXCLUDE

    def _post_processing(self, df):
        # restore drift correction
        df['g'] = df['g'] - df['c_drift']

        # restore tide corection
        c_tide = df.c_tide.copy()
        c_tide[~df['longman'].values] = 0.0
        df['g'] = df['g'] - c_tide
        df = df.drop(['longman'], 1)

        # restore utc
        if 'gmt' in df.columns:
            df.index = df.index + pd.to_timedelta(df['gmt'], unit='hours')
            df = df.drop(['gmt'], 1)

        return df


class ScintrexCG6(_ScintrexCGBase):
    instrument_name = 'Scintrex CG-6'

    _default_corrections = {
        'c_tide': ('c_tide', {}),
        'c_tilt': ('c_tilt', {}),
        'c_temp': ('c_temp', {})}

    _header_fields = _CG6_HEADER

    _data_columns = _CG6_COLUMNS

    _data_columns_dtype = _CG6_COLUMNS_DTYPE

    _exclude_columns = _CG6_COLUMNS_EXCLUDE
