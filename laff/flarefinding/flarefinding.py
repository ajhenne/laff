import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('laff')

def sequential_findflares(data) -> list:
    logger.debug("Starting sequential_findflares()")

    final_index = len(data.flux) - 2
    n = 0
    prev_start, prev_decay = 0, 0

    FLARES = []

    while n < final_index:

        dev_start = n
        dev_count = 0
        
        # Run deviation check.

        if data.iloc[n+1].flux > data.iloc[n].flux:
            dev_count = 1
            if n+dev_count+1 >= final_index:
                n = final_index
                continue
            while data.iloc[n+dev_count+1].flux >= data.iloc[n+dev_count].flux:
                dev_count += 1

        if dev_count >= 2:
            logger.debug(f"Possible deviation from {dev_start}->{dev_start+dev_count}")

            start_point = find_start(data, dev_start, prev_decay)
            peak_point = find_peak(data, start_point)

            if check_rise(data, start_point, peak_point):
                decay_point = find_decay(data, peak_point)

                checks = [check_noise(data, start_point, peak_point, decay_point),
                          check_above(data, start_point, decay_point)]
                logger.debug(f"Checks: {checks}")

                if all(checks):
                    FLARES.append([start_point, peak_point, decay_point])
                    n = decay_point + 1
                    prev_decay = decay_point
                    continue
            else:
                # Check failed.
                logger.debug(f"Deviation NOT passed check")
        else:
            # dev_count not greater than 2, move on.
            pass

        n += 1

    return FLARES

def find_start(data: pd.DataFrame, start: int, prev_decay: int) -> int:
    """Return flare start by looking for local minima."""

    if start < 3:
        points = data.iloc[0:3]
    else:
        points = data.iloc[start-3:start+1]
    minimum = data[data.flux == min(points.flux)].index.values[0]
    minimum = (minimum + 1) if (minimum <= prev_decay) else minimum
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

def find_decay(data: pd.DataFrame, peak: int) -> int:
    """
    Find the end of the flare as the decay smoothes into continuum.

    Longer description.

    :param data:
    :param peak:
    :returns:
    """
    decay = peak
    condition = 0
    decaypar = 3

    def calc_grad(data: pd.DataFrame, idx1: int, idx2: int) -> int:
        """Calculate gradient between first (idx1) and second (idx2) points."""
        deltaFlux = data.iloc[idx2].flux - data.iloc[idx1].flux
        deltaTime = data.iloc[idx2].time - data.iloc[idx1].time
        return deltaFlux/deltaTime

    while condition < decaypar:

        decay += 1
        logger.debug(f'for flare {peak}') # ?
        if data.idxmax('index').time in [decay + i for i in range(-1,2)]:
            logger.debug(f"Reached end of data, automatically ending flare at {decay +1}")
            return data.idxmax('index').time

        # Calculate gradients.
        NextAlong = calc_grad(data, decay, decay+1)
        PrevAlong = calc_grad(data, decay-1, decay)
        PeakToCurrent = calc_grad(data, peak, decay)
        PeakToPrev = calc_grad(data, peak, decay-1)

        cond1 = NextAlong > PeakToCurrent # Next sequence is shallower than from peak to next current.
        cond2 = NextAlong > PrevAlong # Next grad is shallower than previous grad.
        cond3 = PeakToCurrent > PeakToPrev # Peak to next point is shallower than from peak to previous point.

        # Evaluate conditions.
        if cond1 and cond2 and cond3:
            condition += 1
        elif cond1 and cond3:
            condition += 0.5
        # else:
            # condition -= 0.5 if condition >= 0.5 else 0

    logger.debug(f"Decay end found at {decay}")

    return decay


    # once end is found we will check if the flare is 'good'
    # if flare is good, accept it and continue search -> from end + 1
    # if flare is not good, disregard and continue search from deviation + 1

def check_rise(data: pd.DataFrame, start: int, peak: int) -> bool:
    """Test the rise is significant enough."""
    if data.iloc[peak].flux > data.iloc[start].flux + (2 * data.iloc[start].flux_perr):
        logger.debug("check_rise: true")
        return True
    else:
        logger.debug("check_rise: false")
        return False


def check_noise(data: pd.DataFrame, start: int, peak: int, decay: int) -> bool:
    """Check if flare is greater than x1.75 the average noise across the flare."""
    average_noise = abs(np.average(data.iloc[start:decay].flux_perr)) + abs(np.average(data.iloc[start:decay].flux_nerr))
    flux_increase = data.iloc[peak].flux - data.iloc[start].flux
    logger.debug(f"noise: {average_noise} | delta_flux: {flux_increase}")
    return True if flux_increase > 1.75 * average_noise else False

# def check_shape(data: pd.DataFrame, start: int, peak:int, decay:int) -> bool:
    # """Check the shape of the flare."""

def check_above(data: pd.DataFrame, start: int, decay: int) -> bool:
    """Check the flare is above the (estimated) continuum."""
    start = 1 if start == 0 else start # if start is first point, don't use n-1
    decay = data.idxmax('index').time - 1 if decay == data.idxmax('index').time else decay

    slope = (data['flux'].iloc[decay+1] - data['flux'].iloc[start-1])/(data['time'].iloc[decay+1] - data['time'].iloc[start-1])
    intercept = data['flux'].iloc[start-1] - slope * data['time'].iloc[start-1]

    points_above = sum(flux > (slope*time+intercept) for flux, time in zip(data['flux'].iloc[start:decay], data['time'].iloc[start:decay]))
    num_points = len(data['flux'].iloc[start:decay])

    logger.debug(f"points above/num_points => {points_above}/{num_points} = {points_above/num_points}")
    return True if points_above/num_points >= 0.7 else False
