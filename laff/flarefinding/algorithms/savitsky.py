import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import logging

from .flare_checks import check_noise, check_slopes, check_above, check_variance

logger = logging.getLogger('laff')

def flares_savgol(data, **kwargs) -> list:

    """_summary_

    Args:
        data (pd.DataFrame): _description_
        algorithm (_type_): _description_

        plot_savgol(bool): overlay the filter used on an ongoing plt object.
    Returns:
        _type_: _description_
    """


    logger.debug("Starting sequential_findflares()")

    if len(data.index) > 15:
        size = 15
    else: size = 4
    
    data['savgol'] = savgol_filter(data.flux, window_length=size, polyorder=3)

    if 'plot_savgol' in kwargs:
        plt.plot(data['time'], data['savgol'], color='m', linestyle='--', linewidth=0.5)
    
    final_index = len(data.flux) - 2
    
    n = 0
    prev_decay = 0
    FLARES = []

    while n < final_index:

        search_start = n
        search_count = 0
        
        # Run deviation check.
        if data.iloc[n+1].savgol > data.iloc[n].savgol:
            search_count = 1

            # Boundary if we reach end of data.
            # print('nsearch2', n+search_count+2)
            # print('finalindex', final_index)
            if n+search_count+2 >= final_index:
                n = final_index
                continue

            try:
                while data.iloc[n+search_count+1].savgol >= data.iloc[n+search_count].savgol:
                    search_count += 1
            except IndexError:

                # reach end of data
                search_count = 0
                n = final_index
                logger.debug('Reached end of data, ending flare search.')
                continue

        if search_count >= 2:

            logger.debug("Possible deviation from %s -> %s (t=%s)", search_start, search_start+search_count, data.iloc[search_start].time)

            start_point = find_start(data, search_start, prev_decay)
            peak_point = find_peak(data, start_point)

            ## Quick rise check.
            savgol_rise_start = data['savgol'].iloc[peak_point] > data['savgol'].iloc[start_point] + 2*(data['flux_perr'].iloc[start_point])
            savgol_rise_before = data['savgol'].iloc[peak_point] > np.average(data['savgol'].iloc[max(start_point - 5, 0):start_point] + 2*(data['flux_perr'].iloc[max(start_point - 5, 0):start_point]))

            if not savgol_rise_start and not savgol_rise_before:
                logger.debug("Doesn't meet savgol rise condition")
                n += 1
                continue

            peak_point, decay_point = find_decay(data, start_point, peak_point)

            checks = [check_noise(data, start_point, peak_point),
                      check_slopes(data, start_point, peak_point, decay_point),
                      check_above(data, start_point, decay_point)]
                    #   check_variance(data, peak_point, decay_point)]    
            logger.debug(f"\tChecks: {checks}")

            if all(checks):
                check_variance(data, peak_point, decay_point)
                check_noise(data, start_point, peak_point)
                FLARES.append([start_point, peak_point, decay_point])
                logger.debug(f"\tconfirmed flare::  {start_point, peak_point, decay_point}")
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


    ## dev - test
    # with the savgol smoothing, somestimes the wrong point is selected.
    if data['flux'].iloc[start + 1] < data['flux'].iloc[start]:

        bottom_next = data['flux'].iloc[start + 1] + data['flux_nerr'].iloc[start + 1]
        top_prev    = data['flux'].iloc[start - 1] + data['flux_perr'].iloc[start - 1]

        if bottom_next < top_prev:
            minimum = start + 1
            logger.debug("\tflare start found at %s (t=%s)", minimum, data['time'].iloc[minimum])
            return minimum

    if start < 3:
        points = data.iloc[0:3]
    else:
        points = data.iloc[start-3:start+2]
    minimum = data[data.flux == min(points.flux)].index.values[0]
    minimum = prev_decay if (minimum < prev_decay) else minimum

    logger.debug(f"\tflare start found at {minimum}")
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

        logger.debug(f"\tFlare peak found at {maximum} - using end of data cutoff.")
        return maximum

    i = 1

    while next_chunk > prev_chunk:
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

    logger.debug("\tFlare peak found at %s (t=%s)", maximum, data['time'].iloc[maximum])
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

    logger.debug("\tlooking for decay")

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
        if (data['time'].iloc[decay+1] - data['time'].iloc[decay]) > (data['time'].iloc[decay] - data['time'].iloc[peak]) * 3:
            logger.debug(f"Gap between {decay}->{decay+1} is greater than {peak}->{decay} * 3")
            condition = 3
            continue

        # Prevent early ending - should decay to below start.
        if data['flux'].iloc[decay] > data['flux'].iloc[start]:
            condition = 0 # should this always be the case?
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

    logger.debug("\tdecay found at %s (t=%s)", decay, data['time'].iloc[decay])

    # Adjust end for local minima.
    decay = data[data.flux == min(data.iloc[decay-1:decay+1].flux)].index.values[0]
    if decay <= peak:
        logger.debug("decay == peak, adding one")
        # raise ValueError('decay is before or on the peak')
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
