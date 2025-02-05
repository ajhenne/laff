import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger('laff')

def apply_filter(data) -> list:
    logger.debug("Starting sequential_findflares()")

    # size = int(len(data.time)/10) if len(data.time) <= 30 else 13
    # size = 4 if size < 4 else size
    size = 7 if len(data.index) > 7 else 4
    data['savgol'] = savgol_filter(data.flux, window_length=size, polyorder=3)

    #######

    final_index = len(data.flux) - 2
    n = 0
    prev_decay = 0

    FLARES = []

    while n < final_index:

        logger.debug(f"Looking at index {n}")

        search_start = n
        search_count = 0
        
        # Run deviation check.
        if data.iloc[n+1].savgol > data.iloc[n].savgol:
            search_count = 1

            # Boundary if we reach end of data.
            if n+search_count+1 >= final_index:
                n = final_index
                continue

            while data.iloc[n+search_count+1].savgol >= data.iloc[n+search_count].savgol:
                search_count += 1

        if search_count >= 2:
            logger.debug(f"Possible deviation from {search_start}->{search_start+search_count} ({data.iloc[search_start].time})")

            start_point = find_start(data, search_start, prev_decay)
            # print(f'{start_point=}')
            peak_point = find_peak(data, start_point)
            # print(f'{peak_point=}')

            peak_point, decay_point = find_decay(data, start_point, peak_point)
            # logger.debug(f'Possible flare rise from {start_point}->{peak_point}')

            # checks = [  check_rise(data, start_point, peak_point),
            #             check_noise(data, start_point, peak_point, decay_point),
            #             check_above(data, start_point, decay_point),
            #             check_decay_shape(data, peak_point, decay_point)    ]
            checks = [check_noise(data, start_point, peak_point)]
            #dev
            # checks = [True for x in checks]
            #dev
            logger.debug(f"Checks: {checks}")

            if all(checks):
                FLARES.append([start_point, peak_point, decay_point])
                logger.debug(f"Confirmed flare::  {start_point, peak_point, decay_point}")
                n = decay_point
                prev_decay = decay_point
                continue
            else:
                # All checks failed.
                logger.debug("Flare failed passing all tests - discarding.")
        else:
            # search_count not greater than 2, move on.
            pass

        n += 1

    return FLARES


def find_start(data: pd.DataFrame, start: int, prev_decay: int) -> int:
    """Return flare start by looking for local minima."""
    if start < 3:
        points = data.iloc[0:3]
    else:
        points = data.iloc[start-3:start+2]
    minimum = data[data.flux == min(points.flux)].index.values[0]
    minimum = prev_decay if (minimum < prev_decay) else minimum
    logger.debug(f"Flare start found at {minimum}")

    return minimum


def find_peak(data, start):
    """
    Return flare peak by looking for local maxima.

    Starting at point {start}, look for the peak of the flare. Since this could
    be one, or many points away a moving average algorithm is used. Work out
    the average of 5 point chunks and see if this is still rising. Until the
    rise stops, continue to search. Once a decay has been found, the peak is the
    datapoint with maximum value.

    :param data: The pandas dataframe containing the lightcurve data.
    :param start: Integer position of the flare start.
    :return: Integer position of the flare peak.
    """

    chunksize = 4
    prev_chunk = data['flux'].iloc[start] # Flare start position is first point.
    next_chunk = np.average(data.iloc[start+1:start+1+chunksize].flux) # Calculate first 'next chunk'.

    # boundary for end of data
    if start + 1 + chunksize >= len(data.index): # has looped around
        points = data.iloc[start+1:len(data.index)]
        maximum = data[data.flux == max(points.flux)].index.values[0]

        logger.debug(f"Flare peak found at {maximum} - using end of data cutoff.")
        return maximum

    i = 1

    while next_chunk > prev_chunk:
        logger.debug(f"Looking at chunk i={i} : {(start+1)+(chunksize*i)}->{(start+1+4)+(chunksize*i)}")
        # Next chunk interation.
        i += 1
        prev_chunk = next_chunk
        next_chunk = np.average(data.iloc[(start+1)+(chunksize*i):(start+1+chunksize)+(chunksize*i)].flux)
    else:
        # Data has now begin to descend so look for peak up to these points.
        # Include next_chunk in case the peak lays in this list, but is just
        # brought down as an average by remaining points.
        points = data.iloc[start:(start+1+chunksize)+(chunksize*i)]
        maximum = data[data.flux == max(points.flux)].index.values[0]

    logger.debug(f"Flare peak found at {maximum}")
    return maximum


def find_decay(data: pd.DataFrame, start: int, peak: int) -> int:
    """
    Find the end of the flare as the decay smoothes into continuum.

    Longer description.

    :param data:
    :param peak:
    :returns:
    """

    decay = peak
    condition = 0
    # decaypar = 2.5

    logger.debug(f"Looking for decay")

    def calc_grad(data: pd.DataFrame, idx1: int, idx2: int, peak: bool = False) -> int:
        """Calculate gradient between first (idx1) and second (idx2) points."""
        deltaFlux = data.iloc[idx2].savgol - data.iloc[idx1].flux if peak else data.iloc[idx2].savgol - data.iloc[idx1].savgol
        deltaTime = data.iloc[idx2].time - data.iloc[idx1].time
        if deltaTime < 0:
            raise ValueError("It appears data is not sorted in time order. Please fix this.")
        return deltaFlux/deltaTime

    while condition < 3:
        decay += 1

        # Boundary condition for end of data.
        if data.idxmax('index').time in [decay + i for i in range(-1,2)]:  # reach end of data
            logger.debug(f"Reached end of data, automatically ending flare at {decay + 1}")
            condition = 3
            decay = data.idxmax('index').time
            continue

        # Condition for large orbital gaps.
        if (data['time'].iloc[decay+1] - data['time'].iloc[decay]) > (data['time'].iloc[decay] - data['time'].iloc[peak]) * 10:
            logger.debug(f"Gap between {decay}->{decay+1} is greater than {peak}->{decay} * 10")
            condition = 3
            decay += 1
            continue

        # Condition for being greater than peak.
        if data['savgol'].iloc[decay+1] > data['savgol'].iloc[peak]:
            logger.debug("Has risen higher than the peak.")
            condition = 3
            continue

        # Calculate gradients.
        NextAlong = calc_grad(data, decay, decay+1)
        PrevAlong = calc_grad(data, decay-1, decay)
        PeakToCurrent = calc_grad(data, peak, decay, peak=True)
        PeakToPrev = calc_grad(data, peak, decay-1, peak=True)
        
        # print(NextAlong, PrevAlong, PeakToCurrent, PeakToPrev)

        cond1 = NextAlong > PeakToCurrent # Next sequence is shallower than from peak to next current.
        cond2 = NextAlong > PrevAlong # Next grad is shallower than previous grad.
        cond3 = PeakToCurrent > PeakToPrev # Peak to next point is shallower than from peak to previous point.

        if cond1 and cond2 and peak == decay - 1: # special case for first test only
            cond3 = True

        # Evaluate conditions - logic in notebook 20th august.
        # if cond1 and cond2 and cond3:
        #     condition += 1
        # else:
        #     condition = condition - 1 if condition > 0 else 0

        ## dev
        # print(f"At {decay} conditions are [{cond1, cond2, cond3}] and condition before eval is {condition}")
        ## dev


        if cond1 and cond2 and cond3:
            if condition == 2:
                condition = 3
            elif condition == 1:
                condition = 3
            elif condition == 0:
                condition = 2
        else:
            if condition == 2:
                condition = 1
            if condition == 1:
                condition = 0

        if (data['savgol'].iloc[decay] > data['flux'].iloc[start]):
            condition = 0


    logger.debug(f"Decay end found at {decay}")

    # Adjust end for local minima.
    decay = data[data.flux == min(data.iloc[decay-1:decay+1].flux)].index.values[0]
    if decay <= peak:
        raise ValueError('decay is before or on the peak')
        decay = peak + 1

    # Check flare peak adjustments.
    adjusted_flare_peak = data[data.flux == max(data.iloc[peak:decay].flux)].index.values[0]

    if peak < adjusted_flare_peak:
        logger.debug(f"Flare peak adjust from {peak} to {adjusted_flare_peak} - likely a noisy/multiple flare.")
        peak = adjusted_flare_peak

    return peak, decay


    # once end is found we will check if the flare is 'good'
    # if flare is good, accept it and continue search -> from end + 1
    # if flare is not good, disregard and continue search from deviation + 1


#####
# checks

def check_noise(data, start_point, peak_point):
    """Flare rise must be greater than x2 the local noise."""

    flare_rise = data.iloc[peak_point].savgol - data.iloc[start_point].savgol
    noise_level = np.average([data.iloc[start_point:peak_point].flux_perr, abs(data.iloc[start_point:peak_point].flux_nerr)])

    if flare_rise < 2 * noise_level:
        return False
    return True
