#!/usr/bin/env python

import argparse
import os

from gmeterpy.meters.gphone.gphone import gPhone


def main():
    parser = argparse.ArgumentParser(description="Convert and preprocess gPhone data")
    parser.add_argument(
        "infile",
        metavar="INPUT",
        type=argparse.FileType("rt"),
        help="Input gPhone data file",
    )

    parser.add_argument("--station", type=str, help="human-readable station name")
    parser.add_argument("--station-id", type=str, help="computer-readable station id")
    parser.add_argument("--lat", type=float, help="latitude")
    parser.add_argument("--lon", type=float, help="longitude")

    args = parser.parse_args()

    proc_dir = os.path.dirname(os.path.abspath(args.infile.name))
    os.chdir(proc_dir)

    data = gPhone(args.infile.name)

    data._data["meter_sn"] = "110"
    data._data["sid"] = "CONT"
    data._data["line"] = 0
    # data._data['dur'] = 1
    data._data["stdev"] = 1
    data._data["rej"] = 0

    data._data["lon"] = args.lon
    data._data["lat"] = args.lat
    # data._data['elev'] = args.elev

    data.to_file("readings.txt")
