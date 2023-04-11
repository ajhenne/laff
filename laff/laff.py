import pandas as pd

from .laff_settings import RUNPARAMETERS

from .methods import (
    _find_deviation,
    _find_minima,
    _find_maxima,
)


# laff.findflares

def findFlares(data):

    # function: if column names aren't [list] then rename to this

    """
    

    Input:
        data
            A pandas table or list of lists containing the light curve data.
            Assumed to be time, time_err, flux, flux_err.
    """

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

    print(flare_starts)
    print(flare_peaks)
        





# find deviations

# find peaks
# find starts
# find end


# laff.fitcontinuum
# laff.fitflares
# laff.calculatefluence
# opt -> laff.plot