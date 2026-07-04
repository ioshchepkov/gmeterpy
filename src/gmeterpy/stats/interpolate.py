#!/usr/bin/env python

import pandas as pd


def interpolate(ts, datetime_index):
    x = pd.concat([ts, pd.Series(index=datetime_index)])
    return (
        x.groupby(x.index)
        .first()
        .sort_index()
        .interpolate(method="linear")[datetime_index]
    )
