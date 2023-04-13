import numpy as np

def _find_deviation(data, index):

    flux = np.array(data.flux)

    if flux[index] < flux[index+1]:
        return False

    if sum([(flux[index+i+1] > flux[index+i]) for i in range(0,5,1)]) >= 3:
        averageBefore = np.average(flux[index-5:index])
        averageAfter = np.average(flux[index+1:index+6])

        pointAndError = flux[index] + data.iloc[index].flux_perr
        print(averageBefore, averageAfter, pointAndError)

        if pointAndError < averageBefore and pointAndError < averageAfter:
            return False

        return True

    return False

def _find_minima(data, deviation_idx):

    highest_data_index = data.idxmax('index').time

    # Check boundaries - don't want to loop search over start/end of light curve.
    if deviation_idx < 25:
        points = data.iloc[:25 - deviation_idx]
    elif (deviation_idx+25) - highest_data_index >= 0 :
        points = data.iloc[deviation_idx-25:]
    else:
        points = data.iloc[deviation_idx-25:deviation_idx+2]
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

def _remove_Duplicates(peaklist, startlist):

    unique_peaks = set()
    duplicate_indices = []

    for i, peak in enumerate(peaklist):
        if peak in unique_peaks:
            duplicate_indices.append(i)
        else:
            unique_peaks.add(peak)

    for index in duplicate_indices:
        del startlist[index]

    return sorted(unique_peaks), startlist

def _find_decay(data, peak_idx, list_of_starts, DECAYPAR):

    # for peakidx starting counting up through data points
    # at each datapoint evaluate three conditions
    # if n has reached the next flare start we end the flare here immediately
    # calculate some gradients and evaluate those conditions
    
    cond_count = 0
    current_idx = peak_idx

    while cond_count < DECAYPAR:

        if current_idx == peak_idx or current_idx + 1 == peak_idx:
            current_idx += 1
            continue

        if current_idx + 1 == data.idxmax('index').time:
            break

        if current_idx + 1 in list_of_starts:
            break

        current_idx += 1

        def _calc_grad(nextpoint, point):
            deltaFlux = data.iloc[nextpoint].flux - data.iloc[point].flux
            deltaTime = data.iloc[nextpoint].time - data.iloc[point].time
            return deltaFlux/deltaTime

        grad_NextAlong = _calc_grad(current_idx+1, current_idx)
        grad_PrevAlong = _calc_grad(current_idx, current_idx-1)
        grad_PeakToNext = _calc_grad(current_idx, peak_idx)
        grad_PeakToPrev = _calc_grad(current_idx-1, peak_idx)

        cond1 = grad_NextAlong > grad_PeakToNext
        cond2 = grad_NextAlong > grad_PrevAlong
        cond3 = grad_PeakToNext > grad_PeakToPrev

        if cond1 and cond2 and cond3:
            cond_count += 1
        elif cond1 and cond3:
            cond_count += 0.5

    flare_end = current_idx

    return flare_end

def _check_FluxIncrease(data, startidx, peakidx):
    check = data.iloc[peakidx].flux > (data.iloc[startidx].flux + (2 * data.iloc[startidx].flux_perr))
    return check

def _check_AverageNoise(data, startidx, peakidx, endidx):
    average_noise = np.average(data.iloc[startidx:endidx].flux_perr) + abs(np.average(data.iloc[startidx:endidx].flux_nerr))
    flux_increase = min(data.iloc[peakidx].flux - data.iloc[startidx].flux, data.iloc[peakidx].flux - data.iloc[endidx].flux)
    check = flux_increase > average_noise * 1.5
    return check