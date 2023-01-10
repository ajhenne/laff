"""laff.laff: main module script."""

__version__ = "0.5.0"

import argparse
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np

import laff.runparameters as runpar
import laff.functions as lf
from .models import Continuum, ModelFlare, FlareObject
from .exceptions import ParameterError

print("======================================================================")
print("Lightcurve and Flare Fitter | Version %s" % __version__)
print("Contact: Adam Hennessy (ah724@leicester.ac.uk")
print("======================================================================")

###############################################################
### PARSE ARGUMENTS
###############################################################

parser = argparse.ArgumentParser(description="Lightcurve and Flare Fitter for GRBs", prog="laff")
parser.add_argument("--version", action='version', version=__version__)
parser.add_argument("data", nargs=1, metavar="data_filepath", type=str, help="Path to the input datafile.")
parser.add_argument("-n", "--name", nargs=1, metavar='name', type=str, help="User specific name for the run, usually GRB name/TriggerID.")
parser.add_argument("-o", "--output", nargs=1, metavar="output", type=str, help="Name of output file to write results to.")
parser.add_argument("-s", "--show", action="store_true", help="Show the plotted fitted lightcurve.")
parser.add_argument("-v", "--verbose", action="store_true", help="Produce more detailed text output.")
parser.add_argument("-q", "--quiet", action="store_true", help="Don't produce any terminal output.")
args = parser.parse_args()

###############################################################
### INIT LOGGING MODULE
###############################################################

# Determine level of output.
if args.quiet:
    logging_level = logging.CRITICAL
elif args.verbose:
    logging_level = logging.DEBUG
else:
    logging_level = logging.INFO

logging.basicConfig(level=logging_level, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("matplotlib").setLevel(logging.CRITICAL) # Silence matplotlib.

###############################################################
### INIT ARGUMENTS 
###############################################################

input_path  = args.data[0] # Data filepath.
output_path = args.output[0] if args.output is not None else False # Output filepath.

# Flare function shape.
if runpar.FlareShape == "fred":
    flareFunc = ModelFlare.fred
elif runpar.FlareShape == "gaussian":
    flareFunc = ModelFlare.gaussian
else:
    logger.critical("Illegal flare shape parameter.")
    raise ParameterError("FlareShape", runpar.FlareShape)

logger.info(f"Input data at {input_path}")
logger.debug(f"Output filepath is {output_path}")
logger.debug(f"Using {runpar.FlareShape} flare model")
logger.info("Starting LAFF run")

def main():

    # Import data.
    data = lf.Imports.swift_xrt(input_path)

    # Find and mark possible deviations.
    flare_start = lf.IdentifyDeviations(data)

    # Find peak and ends of deviations.
    flare_peaks = lf.FindPeaks(data, flare_start)
    flare_decay  = lf.FindEnds(data, flare_start, flare_peaks)

    FlareList = []

    # Assign flares to flare class.
    for idx1, idx2, idx3 in zip(flare_start, flare_peaks, flare_decay):
        FlareList.append(FlareObject(idx1, idx2, idx3))

    logger.info(f"Performing flare verification")
    FlareList = lf.VerifyFlares(data, FlareList)
    logger.info(f"{len(FlareList)} flares remaining from {len(flare_start)} initial deviations")

    # =====
    for flare in FlareList:
        col = 'tab:orange' if flare.keep == True else 'r'

        x = [data.iloc[x]['time'] for x in flare.returnPar()]
        y = [data.iloc[y]['flux'] for y in flare.returnPar()]

        plt.plot(x, y, color=col, zorder=10, linewidth=2)

    # ====  

    """

    First check to see if we can merge two flares


    For each flare check if the (flux nerr) value is greater than the start
    and/or (flux perr) value
    If it doesn't clear this value then remove this flare

    If the 
    
    """

    # ====

    # for strt, peak, decy in zip(flare_start, flare_peaks, flare_decay):
    #     x = [data.iloc[x]['time'] for x in (strt, peak, decy)]
    #     y = [data.iloc[x]['flux'] for x in (strt, peak, decy)]

    #     plt.plot(x,y, linewidth=2, zorder=10)


    plt.errorbar(data.time, data.flux, \
        xerr=[-data.time_nerr, data.time_perr], \
        yerr=[-data.flux_nerr, data.flux_perr], \
    marker='', linestyle='None', capsize=0)

    plt.loglog()

    plt.show()

    # ====

    sys.stdout = sys.__stdout__ # Restore stdout function (incase -q flag used).

    return