import numpy as np

def _find_deviations(data):

    data = np.array(data.flux[:-10])
    deviations = []

    for index in data.index[:10]:

        flux = np.array(data.flux)

        # Just return false if next point is lower.
        if flux[index] < flux[index+1]:
            continue

        # Check it increases 3 out of 5 times.
        if not sum([(flux[index+i+1] > flux[index+i]) for i in range(0,5,1)]) >= 3:
            continue

        # Calculate average of values before and after point.
        averageBefore = np.average(flux[index-5:index])
        averageAfter = np.average(flux[index+1:index+6])

        # Get point plus error.
        pointAndError = flux[index] + data.iloc[index].flux_perr

        # Check it satisfies average.
        if pointAndError < averageBefore and pointAndError < averageAfter:
            continue

        deviations.append(index)

    return sorted(set(deviations))

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
        indicesToRemove.append(*pointsToCompare)
    
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
            if current_index + 1 == data.idxmax('index').time:
                break
            # Check if we reach next start.
            if current_index + 1 in starts:
                break

            current_index += 1

            def __calc_grad(nextpoint, point):
                deltaFlux = data.iloc[nextpoint].flux - data.iloc[point].flux
                deltaTime = data.iloc[nextpoint].time - data.iloc[point].time
                return deltaFlux/deltaTime

            grad_NextAlong = __calc_grad(current_index+1, current_index)
            grad_PrevAlong = __calc_grad(current_index, current_index-1)
            grad_PeakToNext = __calc_grad(current_index, peak_index)
            grad_PeakToPrev = __calc_grad(current_index-1, peak_index)

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
    check = data.iloc[peakidx].flux > (data.iloc[startidx].flux + (2 * data.iloc[startidx].flux_perr))
    return check

def _check_AverageNoise(data, startidx, peakidx, endidx):
    average_noise = np.average(data.iloc[startidx:endidx].flux_perr) + abs(np.average(data.iloc[startidx:endidx].flux_nerr))
    flux_increase = min(data.iloc[peakidx].flux - data.iloc[startidx].flux, data.iloc[peakidx].flux - data.iloc[endidx].flux)
    check = flux_increase > average_noise * 1.5
    return check

def _check_AverageGradient(data, startidx, peakidx, endidx):

    check = False

    def __calc_grad(nextpoint, point):
        deltaFlux = data.iloc[nextpoint].flux - data.iloc[point].flux
        deltaTime = data.iloc[nextpoint].time - data.iloc[point].time
        return deltaFlux/deltaTime

    rise_gradient = [__calc_grad(idx+1, idx) for idx in range(startidx, peakidx)]
    decay_gradient = [__calc_grad(idx+1, idx) for idx in range(peakidx, endidx)]

    if sum([(gradient > 0) for gradient in rise_gradient])/len(rise_gradient) > 0.75:
        check = True

    if sum([(gradient < 0) for gradient in decay_gradient])/len(decay_gradient) > 0.75:
        check = True

    return check