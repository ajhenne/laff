import pandas as pd

from .laff_settings import RUNPARAMETERS

def findFlares(data):
    """
    

    Input:
        data
            A pandas table or list of lists containing the light curve data.
            Assumed to be time, time_err, flux, flux_err.
    """

    from .methods import (
        _find_deviation,
        _find_minima,
        _find_maxima,
        _find_decay,
    )

    # Find deviations - 'possible flares'.
    deviations = []
    for index in data.index[:-2]:
        if _find_deviation(data, index, RUNPARAMETERS['rise_par']) == True:
            deviations.append(index)

    # Refine deviations by looking for local minima - flare starts.
    for index, start in enumerate(deviations):
        deviations[index] = _find_minima(data, start)
    flare_starts = sorted(set(deviations))

    # For each flare start, find a flare peak.
    flare_peaks = [None] * len(flare_starts)
    for index, start in enumerate(flare_starts):
        flare_peaks[index] = _find_maxima(data, start)

    # Implement a check here to ensure two starts don't share the same peak?

    # For each flare peak, find the flare end.
    flare_ends = [None] * len(flare_peaks)
    for index, peak in enumerate(flare_peaks):
        flare_ends[index] = _find_decay(data, peak, flare_starts) # decay finder function needs completing

    Flares = []
    for i_start, i_peak, i_end in zip(flare_starts, flare_peaks, flare_ends):
        Flares.append([i_start, i_peak, i_end])
    
    # Return an organised list, or just return 3 lists of start/peak/ends?
    return Flares

def fitContinuum(data, Flares):

    return ContinuumModel

def fitFlares(data, Flares, ContinuumModel):

    return FlareModel

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