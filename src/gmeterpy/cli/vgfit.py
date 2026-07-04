#!/usr/bin/env python

import argparse

import numpy as np
from pandas import read_csv

from gmeterpy.corrections.vgrad import fit_floating_gravity, generate_report
from gmeterpy.plotting.vgfit import plot_fit


def main():
    parser = argparse.ArgumentParser(
        description="Polynomial fit of the vertical gravity gradients data"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("rt"),
        required=True,
        help="Input file",
        metavar="FILE",
    )
    parser.add_argument(
        "-n", "--name", default="", type=str, help="Station name", metavar="STATION"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=int,
        default=2,
        help="Degree of the polynomial model (default: 2).",
        metavar="DEGREE",
    )

    args = parser.parse_args()

    data = read_csv(args.input)

    df, res = fit_floating_gravity(data, deg=args.degree)
    params = res.params.iloc[-args.degree :]
    cov_params = res.cov_params().iloc[-args.degree :, -args.degree :]

    df["resid"] = res.resid

    gp = np.poly1d(np.concatenate([params.values[::-1], np.zeros(1)]), variable="h")

    make_plot = True
    make_report = True

    station = args.name

    if make_plot:
        fig = plot_fit(df, gp, cov_params, station)
        plot_file = station + "_all.png"
        fig.savefig(plot_file)

    if make_report:
        report_file = "vg_" + station + "_fit_all.txt"
        generate_report(report_file, data, res, gp, station)
