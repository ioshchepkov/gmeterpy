# -*- coding: utf-8 -*-
"""TSoft format reader.

"""

import re
import numpy as np
import pandas as pd

# possible tags in TSoft format
_TAGS = ['TSF-file', 'TIMEFORMAT', 'COUNTINFO', 'INCREMENT', 'CHANNELS',
         'UNITS', 'UNDETVAL', 'COMMENT', 'DATA', 'LABEL',
         'LININTERPOL', 'CUBINTERPOL', 'GAP', 'STEP']


def read_tsf(filename, encoding='utf-8', channels=None):
    """Read TSoft file and return pandas DataFrame.

    """
    blocks = {}
    with open(filename, 'r', encoding=encoding) as file_object:
        for line in file_object:
            _block = re.search('|'.join(_TAGS), line)
            if _block:
                tag = _block[0]
                blocks[tag] = []
                line = line.replace('[' + tag + ']', '')

            line = line.strip()

            if not line:
                continue

            blocks[tag].append(line)

    blocks['UNDETVAL'] = float(blocks['UNDETVAL'][0])
    blocks['TIMEFORMAT'] = str(blocks['TIMEFORMAT'][0])

    datetime_columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
    if blocks['TIMEFORMAT'] == 'DATETIMEFRAC':
        datetime_columns += ['ms']
    elif blocks['TIMEFORMAT'] == 'DATETIME':
        pass
    else:
        raise ValueError

    for idx, channel in enumerate(blocks['CHANNELS']):
        blocks['CHANNELS'][idx] = channel.strip().split(':')

    data_columns = np.asarray(blocks['CHANNELS'])[:, 2]
    columns = datetime_columns + list(data_columns)

    data = np.asarray([line.split() for line in blocks['DATA']])
    df = pd.DataFrame(data, columns=columns, dtype=float).replace(
        blocks['UNDETVAL'], np.NaN)
    time = pd.to_datetime(df[datetime_columns])
    df.drop(datetime_columns, axis='columns', inplace=True)
    df.set_index(time, inplace=True)

    if channels is not None:
        df = df[channels]

    return df
