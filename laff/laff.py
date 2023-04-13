import pandas as pd

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
    )

    # Late cutoff check.
    data = data[data.time < 2000] if RUNPARAMETERS['late_cutoff'] else data

    # Find deviations - 'possible flares'.
    deviations = []
    for index in data.index[:-10]:
        if _find_deviation(data, index) == True:
            deviations.append(index)
    
    temp_dev = deviations

    # Refine deviations by looking for local minima - flare starts.
    for index, start in enumerate(deviations):
        deviations[index] = _find_minima(data, start)
    flare_starts = sorted(set(deviations))

    # For each flare start, find a flare peak.
    flare_peaks = [None] * len(flare_starts)
    for index, start in enumerate(flare_starts):
        flare_peaks[index] = _find_maxima(data, start)

    # Check there's no duplicates in lists.
    flare_peaks, flare_starts = _remove_Duplicates(flare_peaks, flare_starts)

    # For each flare peak, find the flare end.
    flare_ends = [None] * len(flare_peaks)
    for index, peak in enumerate(flare_peaks):
        flare_ends[index] = _find_decay(data, peak, flare_starts, RUNPARAMETERS['decay_par']) # decay finder function needs completing

    Flares = []
    for i_start, i_peak, i_end in zip(flare_starts, flare_peaks, flare_ends):

        check1 = _check_FluxIncrease(data, i_start, i_peak)
        check2 = _check_AverageNoise(data, i_start, i_peak, i_end)

        if check1 and check2:
            Flares.append([i_start, i_peak, i_end])
    
    return Flares, temp_dev

def fitContinuum(data, Flares):

    return #ContinuumModel

def fitFlares(data, Flares, ContinuumModel):

    return #FlareModel

def fitLightCurve(data):

    findFlares()
    fitContinuum()
    fitFlares()
    # combinedModel()

        





# find deviations

# find peaks
# find starts
# find end


# laff.fitcontinuum
# laff.fitflares
# laff.calculatefluence
# opt -> laff.plot