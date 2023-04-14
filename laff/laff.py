import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import vstack

from .laff_settings import RUNPARAMETERS

def findFlares(data):
    """
    

    Input:
        data
            A pandas table or list of lists containing the light curve data.
            Assumed to be time, time_err, flux, flux_err.

    Output:
        Flares
            A nested list of flare (start, stop, end indices).
    """

    from .methods import (
        _find_deviation,
        _find_minima,
        _find_maxima,
        _find_decay,
        _remove_Duplicates,
        _check_FluxIncrease,
        _check_AverageNoise,
        _check_AverageGradient
    )

    # Add flare column.
    if not 'flare' in data.columns: data['flare'] = False

    # Late cutoff check.
    lc_data = data[data.time < 2000] if RUNPARAMETERS['late_cutoff'] else data


    # Find deviations - 'possible flares'.
    deviations = []
    for index in lc_data.index[:-10]:
        if _find_deviation(lc_data, index) == True:
            deviations.append(index)

    # Refine deviations by looking for local minima - flare starts.
    flare_starts = deviations
    for index, start in enumerate(flare_starts):
        flare_starts[index] = _find_minima(lc_data, start)
    flare_starts = sorted(set(flare_starts))

    # For each flare start, find a flare peak.
    flare_peaks = [None] * len(flare_starts)
    for index, start in enumerate(flare_starts):
        flare_peaks[index] = _find_maxima(lc_data, start)

    flare_starts, flare_peaks = _remove_Duplicates(lc_data, flare_starts, flare_peaks)

    # For each flare peak, find the flare end.
    flare_ends = [None] * len(flare_peaks)
    for index, peak in enumerate(flare_peaks):
        flare_ends[index] = _find_decay(lc_data, peak, flare_starts, RUNPARAMETERS['decay_par']) # decay finder function needs completing

    Flares = []
    for i_start, i_peak, i_end in zip(flare_starts, flare_peaks, flare_ends):

        check1 = _check_FluxIncrease(lc_data, i_start, i_peak)
        check2 = _check_AverageNoise(lc_data, i_start, i_peak, i_end)
        check3 = _check_AverageGradient(lc_data, i_start, i_peak, i_end)

        if check1 and check2 and check3:
            Flares.append([i_start, i_peak, i_end])
            
            rise_start = data.index >= i_start
            decay_end = data.index < i_end

            data.loc[rise_start & decay_end, 'flare'] = True

    return data, Flares

def fitContinuum(data, Flares):

    data = data[data.flare == False]


    return data

def fitFlares(data, Flares, ContinuumModel):

    return #FlareModel

def fitLightCurve(data):

    findFlares()
    fitContinuum()
    fitFlares()
    # combinedModel()


def plotResults(data, Flares):

    cont_data = data[data.flare == False]

    # Plot lightcurve.
    plt.errorbar(cont_data.time, cont_data.flux,
    xerr=[-cont_data.time_nerr, cont_data.time_perr], \
    yerr=[-cont_data.flux_nerr, cont_data.flux_perr], \
    marker='', linestyle='None', capsize=0)

    # Plot flares.
    for start, peak, end in Flares:
        plt.axvline(x = data.iloc[start].time, color='g')
        plt.axvline(x = data.iloc[peak].time, color ='b')
        plt.axvline(x = data.iloc[end].time, color='r')

    plt.loglog()
    plt.show()

        





# find deviations

# find peaks
# find starts
# find end


# laff.fitcontinuum
# laff.fitflares
# laff.calculatefluence
# opt -> laff.plot