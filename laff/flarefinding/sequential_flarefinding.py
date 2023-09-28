import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('laff')

def findFlares(data) -> list:
    logger.debug("Starting findFlares()")

    final_index = len(data.flux) - 1
    n = 0

    FLARES = []

    while n < final_index:

        dev_start = n
        dev_count = 0
        # Run deviation check.

        if data.iloc[n+1].flux > data.iloc[n].flux:
            dev_count = 1
            while data.iloc[n+dev_count+1].flux >= data.iloc[n+dev_count].flux:
                dev_count += 1

        if dev_count >= 2:
            logger.debug(f"Possible deviation from {dev_start}->{dev_start+dev_count}")

            start_point = find_start(data, dev_start)
            peak_point = find_peak(data, start_point)

            if check_rise(data, start_point, peak_point):
                # decay_point = find_decay(data, peak_point)

                # if check_X:
                    # FLARES.append([start_point, peak_point, decay_point])
                    # n = decay_point + 1
                    # continue
                FLARES.append([start_point, peak_point])
                n = peak_point + 5
                continue

                # FLARES.append([start_point, peak_point, decay_point])
                # n = decay_point + 1
                # continue

            else:
                # Check failed.
                logger.debug(f"Deviation NOT passed check")

        else:
            # dev_count not greater than 2, move on.
            pass

        n += 1

    return FLARES

def find_start(data: pd.DataFrame, start: int) -> int:
    """Return flare start by looking for local minima."""

    if start < 3:
        points = data.iloc[0:3]
    else:
        points = data.iloc[start-3:start+1]
    minimum = data[data.flux == min(points.flux)].index.values[0]
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

    # for each position calculate gradients
        # nextalong - calculate gradient current to next
        # prev along - calculate gradient from last to current
        # peak to next - calculate gradient from peak to next
        # peak to prev - calculate graident from peak to last
        # cond1 is nextalong > peaktonext
        # cond2 is nextalong > prev along
        # cond3 is peaktonext > peaktoprev
        # if cond1 cond2 cond3 then cond_count +1
        # if cond1 and cond2 then cound_count +0.5
        # once cond_count >= 4 then end
    
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
