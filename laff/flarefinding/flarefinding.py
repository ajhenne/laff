import numpy as np

def _find_deviations(data):

    deviations = []
    counter = 0

    for i in data.index[:-1]:

        if data.iloc[i+1].flux > data.iloc[i].flux:
            counter += 1
        else:
            counter = 0

        if counter == 2:
            check = data.iloc[i+1].flux + (data.iloc[i+1].flux_perr * 2) > data.iloc[i-1].flux
            if check:
                deviations.append(i-1)
            counter = 0

    return sorted(set(deviations))

# def _find_deviations(data):

#     deviations = []

#     for index in data.index[:-10]:

#         flux = np.array(data.flux)

#         # Just return false if next point is lower.
#         if flux[index] < flux[index+1]:
#             continue

#         # Check it increases 3 out of 5 times.
#         if not sum([(flux[index+i+1] > flux[index+i]) for i in range(0,5,1)]) >= 3:
#             continue

#         # Calculate average of values before and after point.
#         averageBefore = np.average(flux[index-5:index])
#         averageAfter = np.average(flux[index+1:index+6])

#         # Get point plus error.
#         pointAndError = flux[index] + data.iloc[index].flux_perr

#         # Check it satisfies average.
#         if not pointAndError > averageBefore:
#             continue

#         deviations.append(index)

#     return sorted(set(deviations))

def _find_minima(data, deviations):

    minima = []

    for deviation_index in deviations:
        # Check boundaries - don't want to loop search over start/end of light
        # curve.
        if deviation_index < 25:
            points = data.iloc[:25 - deviation_index]
        elif (deviation_index+25) > data.idxmax('index').time:
            points = data.iloc[deviation_index-10:data.idxmax('index').time]
        else:
            points = data.iloc[deviation_index-25:deviation_index+2]
        # Default: check 30 points before and a couple after.
        minima.append(data[data.flux == min(points.flux)].index.values[0])

    return sorted(set(minima))

def _find_maxima(data, starts):

    maxima = []

    for start_index in starts:
        # Check boundaries - don't want to loop search over start/end of light curve.
        if (data.idxmax('index').time - start_index) <= 30:
            points = data.iloc[start_index:data.idxmax('index').time]
        else:
            points = data.iloc[start_index:start_index+30]
        # Check the next 30 points.
        maxima.append(data[data.flux == max(points.flux)].index.values[0])

    return sorted(maxima)

def _remove_Duplicates(data, startlist, peaklist):
    """
    Look for flare starts with the same peak and combine.
    
    Sometimes indices A and B are found as flare starts, and both share the same
    peak C. Hence, both A and B likely should be combined as one start, the lowest
    flux is likely to be the start. Future thought: or should it just be the
    earlier index? Which is the more general case.
    """

    unique_peaks = set()
    duplicate_peaks = []
    duplicate_index = []

    indicesToRemove = []

    for idx, peak in enumerate(peaklist):
        if peak in unique_peaks:
            duplicate_peaks.append(peak)
            duplicate_index.append(idx)
        else:
            unique_peaks.add(peak)

    unique_peaks = sorted(unique_peaks)
    duplicate_peaks = sorted(duplicate_peaks)
    duplicate_index = sorted(duplicate_index)

    for data_index, peaklist_index in zip(duplicate_peaks, duplicate_index):
        pointsToCompare = [i for i, x in enumerate(peaklist) if x == data_index]
        # points is a pair of indices in peaklist
        # each peaklist has a corresponding startlist
        # so for point a and point b, find the flux in startlist at point a and b
        # compare these two
        # whichever is the lowest flux is more likely the start
        # so we keep this index and discord the other index

        comparison = np.argmin([data.iloc[startlist[x]].flux for x in pointsToCompare])

        del pointsToCompare[comparison]

        for point in pointsToCompare:
            indicesToRemove.append(point)
    
    new_startlist = [startlist[i] for i in range(len(startlist)) if i not in indicesToRemove]
    new_peaklist = [peaklist[i] for i in range(len(peaklist)) if i not in indicesToRemove]

    return new_startlist, new_peaklist

def _find_end(data, starts, peaks, DECAYPAR):
    """
    Find the end of a flare as the decay smooths into afterglow.
    
    For each peak, start counting up through data indices. At each datapoint,
    evaluate three conditions, by calculating several gradients If we reach the next
    flare start, we end the flare here immediately.
    """
    ends = []

    for peak_index in peaks:

        cond_count = 0
        current_index = peak_index

        while cond_count < DECAYPAR:

            # Check if we reach next peak.
            if current_index == peak_index or current_index + 1 == peak_index:
                current_index += 1
                continue
            # Check if we reach end of data.
            if any([current_index + i for i in range(2)] == data.idxmax('index').time):
                break
            # Check if we reach next start.
            if current_index + 1 in starts:
                current_index += 1
                continue

            current_index += 1

            grad_NextAlong = _calc_grad(data, current_index, current_index+1)
            grad_PrevAlong = _calc_grad(data, current_index-1, current_index)
            grad_PeakToNext = _calc_grad(data, peak_index, current_index)
            grad_PeakToPrev = _calc_grad(data, peak_index, current_index-1)

            cond1 = grad_NextAlong > grad_PeakToNext
            cond2 = grad_NextAlong > grad_PrevAlong
            cond3 = grad_PeakToNext > grad_PeakToPrev

            if cond1 and cond2 and cond3:
                cond_count += 1
            elif cond1 and cond3:
                cond_count += 0.5

        ends.append(current_index)

    return sorted(ends)

def _check_FluxIncrease(data, startidx, peakidx):
    """Check the flare increase is greater than x2 the start error."""
    check = data.iloc[peakidx].flux > (data.iloc[startidx].flux + (2 * data.iloc[startidx].flux_perr))
    return check

def _check_AverageNoise(data, startidx, peakidx, endidx):
    """Check if flare is greater than x2 the average noise across the flare."""
    average_noise = abs(np.average(data.iloc[startidx:endidx].flux_perr)) + abs(np.average(data.iloc[startidx:endidx].flux_nerr))
    flux_increase = min(data.iloc[peakidx].flux - data.iloc[startidx].flux, data.iloc[peakidx].flux - data.iloc[endidx].flux)
    check = flux_increase > average_noise * 2
    return check

def _check_PulseShape(data, startidx, peakidx, endidx):

    rise_phase = _calc_grad(data, startidx, peakidx, indexIsRange=True)
    rise_condition = sum(x > 0 for x in rise_phase) / len(rise_phase)

    decay_phase = _calc_grad(data, peakidx, endidx, indexIsRange=True)
    decay_condition = sum(x < 0 for x in decay_phase) / len(decay_phase)

    check = rise_condition > 0.6 and decay_condition > 0.6
    return check

def _calc_grad(data, index1, index2, indexIsRange=False):

    if indexIsRange == False:
        deltaFlux = data.iloc[index2].flux - data.iloc[index1].flux
        deltaTime = data.iloc[index2].time - data.iloc[index1].time
        return deltaFlux/deltaTime

    if indexIsRange == True:

        indices = range(index1, index2)
        deltaFlux = []
        deltaTime = []
        for i in indices:
            deltaFlux.append(data.iloc[i+1].flux - data.iloc[i].flux)
            deltaTime.append(data.iloc[i+1].time - data.iloc[i].time)

        return [flx / tim for flx, tim in zip(deltaFlux, deltaTime)]
    
    else:
        raise ValueError("Parameter range should be boolean.")