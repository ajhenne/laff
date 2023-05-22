import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .methods import (
    _find_deviations,
    _find_minima,
    _find_maxima,
    _find_end,
    _remove_Duplicates,
    _check_AverageGradient,
    _check_AverageNoise,
    _check_FluxIncrease )

###
# DATA FORMAT
# time  time_perr   time_nerr   flux    flux_perr   flux_nerr
# (s)                           (erg/s)                      
####

def findFlares(data):
    """
    Find flares within a GRB lightcurve.

    Longer description.
    
    [Parameters]
        data
            A pandas table containing the light curve data. Columns named [time,
            time_perr, time_nerr, flux, flux_perr, flux_nerr].
            
    [Returns]
        flares
            A nested list of flare start, stop, end indices.
    """

    # Cutoff late data.
    LATE_CUTOFF = True
    data = data[data.time < 2000] if LATE_CUTOFF else data

    # Find deviations, or possible flares.
    deviations = _find_deviations(data)

    # Refine deviations by looking for local minima, or flare starts.
    starts = _find_minima(data, deviations)

    # For each flare start, find the corresponding peak.
    peaks = _find_maxima(data, starts)

    # Combine any duplicate start/peaks.
    starts, peaks = _remove_Duplicates(data, starts, peaks)

    # For each flare peak, find the corresponding flare end.
    DECAYPAR = 3
    ends = _find_end(data, starts, peaks, DECAYPAR)

    flare_start, flare_peak, flare_end = [], [], []
    # Perform some checks to ensure the found flares are valid.
    for start, peak, end in zip(starts, peaks, ends):
        if all(_check_AverageGradient(data, start, peak), \
               _check_AverageNoise(data, start, peak, end), \
               _check_FluxIncrease(data, start, peak, end)):
            flare_start.append(start)
            flare_peak.append(peak)
            flare_end.append(end)

    return flare_start, flare_peak, flare_end