#!/usr/bin/env python

from typing import ClassVar

from gmeterpy.core.readings import Readings

# from tqdm import tqdm
from gmeterpy.meters.tsoft import read_tsf

_GPHONE_DAT_COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "g",
    "g_result",
    "c_tide",
    "long_level",
    "cross_level",
    "out_temp",
    "in_temp",
    "pres",
    "sensor_pres",
    "beam_position",
    "c_level",
    "c_drift",
    "c_sensor_pres",
    "c_baro",
    "c_polar",
    "c_ocean",
    "velocity",
    "position",
]

_GPHONE_TSF_COLUMNS = [
    "Gravity",
    "Corrected Gravity",
    "Tide",
    "Long Level",
    "Cross Level",
    "Ambient Temperature",
    "Sensor Temperature",
    "Ambient Pressure",
    "Sensor Pressure",
    "Beam Position",
    "Level Correction",
    "Drift Correction",
    "Sensor Pressure Correction",
    "Berometer Compensation",
    "Ambient Temperature Correction",
    "Meter Temperature Correction",
    "Polar Motion Correction",
    "Ocean Load Correction",
]

_GPHONE_TSF_COLUMNS_RENAMED = [
    "g",
    "g_result",
    "c_tide",
    "tilt_x",
    "tilt_y",
    "out_temp",
    "in_temp",
    "pres",
    "sensor_pres",
    "beam_position",
    "c_level",
    "c_drift",
    "c_sensor_pres",
    "c_baro",
    "c_out_temp",
    "c_in_temp",
    "c_polar",
    "c_ocean",
]


class gPhone(Readings):
    _default_corrections: ClassVar[dict] = {
        "c_tide": ("c_tide", {}),
        "c_level": ("c_level", {}),
        "c_sensor_pres": ("c_sensor_pres", {}),
        "c_baro": ("c_baro", {}),
        "c_out_temp": ("c_out_temp", {}),
        "c_in_temp": ("c_in_temp", {}),
        "c_polar": ("c_polar", {}),
        "c_ocean": ("c_ocean", {}),
    }

    def __init__(self, filename, **kwargs):

        if filename.endswith(".tsf"):
            df = read_tsf(filename, encoding="cp1251")
        else:
            raise ValueError("Only *.tsf extension is supported")

        df.rename(
            columns=dict(
                zip(_GPHONE_TSF_COLUMNS, _GPHONE_TSF_COLUMNS_RENAMED, strict=False)
            ),
            inplace=True,
        )

        if "corrections" not in kwargs.keys():
            kwargs["corrections"] = self._default_corrections

        df = df.drop(["c_drift"], axis=1)
        df["meter_name"] = "gPhone"

        super().__init__(df, **kwargs)
