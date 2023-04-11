import numpy as np

def _find_deviation(data, index, RISEPAR):

    flux = np.array(data.flux)
    flux_perr = np.array(data.flux_perr)

    if flux[index] < flux[index+1]:
        return False

    if flux[index+2] > flux[index+1]:
        if flux[index+2] > flux[index] + (flux_perr[index] * RISEPAR):
            return True

    return False

def _find_minima(data, deviation_idx):

    highest_data_index = data.idxmax('index').time

    # Check boundaries - don't want to loop search over start/end of light curve.
    if deviation_idx < 30:
        points = data.iloc[:30 - deviation_idx]
    elif (deviation_idx+30) - highest_data_index >= 0 :
        points = data.iloc[deviation_idx-30:]
    else:
        points = data.iloc[deviation_idx-30:deviation_idx+2]
    # Default: check 30 points before and a couple after.

    minima = data[data.flux == min(points.flux)].index.values[0]

    return minima

def _find_maxima(data, start_idx):

    highest_data_index = data.idxmax('index').time
    diff = start_idx - highest_data_index

    if diff >= 0:
        points = data.iloc[start_idx:start_idx+30-diff]
    else:
        points = data.iloc[start_idx:start_idx+30]

    maxima = data[data.flux == max(points.flux)].index.values[0]

    return maxima