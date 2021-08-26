import matplotlib
matplotlib.use('Agg')

from hepanatools.shift.hist import *
from hepanatools.shift.pdf import *
from hepanatools.shift.tables import *
from hepanatools.shift.config import parse_cdf_config
import hepanatools.utils.plot as hpl
import argparse
import pandas as pd
import h5py
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

from mpi4py import MPI
import sys

parser = argparse.ArgumentParser()
parser.add_argument("cdf_config", help="JSON file with optimization configuration")
parser.add_argument("--output_root", default="cdf.root")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrites --output_root and --output_h5 if they exists instead of appending",
)
parser.add_argument("--verbose", action="store_true")
parser.add_argument(
    "--report_every",
    help="Controls the number of fit interations after which to report intermediate results",
    default=5,
    type=int,
)
parser.add_argument(
    "--load_from",
    action="store_true",
    help="Load CDF from file specified in the configuration file and make plots",
)
args = parser.parse_args()

# create bin optimizer from configuration file
config, config_parser = parse_cdf_config(["-c", args.cdf_config])
if MPI.COMM_WORLD.Get_rank() == 0:
    config_parser.print_values()
    sys.stdout.flush()


tables = EventMatchedTable.FromH5(config.event_tables)

binopt = CDFBinOptimizer.FromConfig(config)

# wrap bins function to pass through callback
def bins_func(nominal, shifted, xbins, ybins, **kwargs):
    return binopt(*config.ybins_func(nominal, shifted, xbins, ybins), **kwargs)


# progress callback to track fit results throughout the optimization
progress = ProgressTrackerCallback(verbose=args.verbose, report_every=args.report_every)

# get the right data
nominal = tables[config.var_label][config.cut_label][config.nominal_sample].values
shifted = tables[config.var_label][config.cut_label][config.shifted_sample].values
all_idx = np.arange(nominal.shape[0])
train_idx, test_idx = train_test_split(all_idx, train_size=config.train_size)
train_nominal = nominal[train_idx]
train_shifted = shifted[train_idx]

test_nominal = nominal[test_idx]
test_shifted = shifted[test_idx]

if not args.load_from:
    cdf = CDF2D(
        train_nominal,
        train_shifted,
        xbins=config.xbins,
        ybins=config.ybins,
        constraint=config.bounds,
        bins_func=partial(bins_func, callback=progress),
    )
    comm = MPI.COMM_WORLD
    progress_callbacks = comm.gather(progress, root=0)

    if comm.Get_rank() == 0:
        mode = "w" if args.overwrite else "a"
        with h5py.File(config.output, mode) as f:
            cdf.ToH5(f, config.name)
        with h5py.File(config.save_progress, mode) as f:
            for i, cb in enumerate(progress_callbacks):
                cb.ToH5(f, f"{config.name}_{i}")

else:
    cdf = CDF2D.FromH5(config.output, config.name)
    with h5py.File(config.save_progress, "r") as f:
        nprogress = len(f.keys())
        progress_callbacks = []
        for iprog in range(nprogress):
            progress_callbacks.append(
                ProgressTrackerCallback.FromH5(
                    config.save_progress, f"{config.name}_{i}"
                )
            )


if MPI.COMM_WORLD.Get_rank() == 0:

    fig, ax = hpl.split_subplots(nrows=2, figsize=(10, 8))
    binopt.target.Draw(ax[0], histtype="step", hatch="//", color="k", label="Target")
    cdf_sampling = Hist1D(cdf.Sample(train_nominal), config.objective_bins)
    cdf_sampling.Draw(ax=ax[0], histtype="step", color="r", label="CDF Sampling")
    ax[1].axhline(1, ls="--", c="gray")
    ratio = Hist1D.Filled(cdf_sampling.n / binopt.target.n, config.objective_bins)
    ratio.Draw(ax[1], histtype="step", color="r")
    hpl.savefig(os.path.join(config.plots, f"{config.name}_cdf_vs_target.pdf"))
    plt.close()

    for i, cb in enumerate(progress_callbacks):
        fig, ax = plt.subplots(figsize=(10, 5))
        cb.Draw(*config.xlim, ax)
        hpl.savefig(os.path.join(config.plots, f"{config.name}_progress_{i}.pdf"))
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    cdf.Draw(ax)
    ax.set_xlabel(config.var_label)
    ax.set_ylabel(r"$\Delta$ %s" % config.var_label)
    hpl.savefig(os.path.join(config.plots, f"{config.name}_cdf.pdf"))

    fig, ax = hpl.split_subplots(nrows=2, ncols=2, figsize=(14, 8))

    def plot_split(top, bottom, nominal, shifted, bins, cdf):
        hists = {
            "nominal": Hist1D(nominal, bins=bins),
            "file_shifted": Hist1D(shifted, bins=bins),
            "evt_shifted": Hist1D(cdf.Shift(nominal), bins=bins),
        }
        hists["nominal"].Draw(top, histtype="step", color="b", lw=2, label="Nominal")
        hists["evt_shifted"].Draw(
            top, histtype="step", color="r", lw=2, label="CDF Shift"
        )
        hists["file_shifted"].Draw(
            top, histtype="error", color="k", lw=2, label="File Shift", ms=15
        )

        ratio_evt_shift = Hist1D.Filled(
            hists["evt_shifted"].n / hists["nominal"].n, hists["evt_shifted"].xaxis
        )
        ratio_file_shifted = Hist1D.Filled(
            hists["file_shifted"].n / hists["nominal"].n, hists["file_shifted"].xaxis
        )

        bottom.axhline(1, linestyle="--", color="gray")
        ratio_evt_shift.Draw(bottom, histtype="step", color="r", lw=2)
        ratio_file_shifted.Draw(bottom, histtype="step", color="b", lw=2)

        top.set_ylabel("Events")
        bottom.set_ylabel("Shift / Nominal")
        bottom.set_ylim([0.5, 1.5])

        return top, bottom

    ax[0][0], ax[1][0] = plot_split(
        ax[0][0], ax[1][0], train_nominal, train_shifted, config.var_bins, cdf
    )
    ax[1][0].set_xlabel(config.var_label)
    ax[0][0].legend()
    train_frac = train_nominal.shape[0] / nominal.shape[0] * 100
    ax[0][0].set_title("Training (%.1f%%)" % train_frac)

    ax[0][1], ax[1][1] = plot_split(
        ax[0][1], ax[1][1], test_nominal, test_shifted, config.var_bins, cdf
    )
    ax[1][1].set_xlabel(config.var_label)
    test_frac = test_nominal.shape[0] / nominal.shape[0] * 100
    ax[0][1].set_title("Testing (%.1f%%)" % test_frac)
    plt.tight_layout()
    hpl.savefig(os.path.join(config.plots, f"{config.name}_compare_train_test.pdf"))
