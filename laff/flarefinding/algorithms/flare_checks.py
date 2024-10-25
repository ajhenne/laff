import logging
import pandas as pd
import numpy as np

logger = logging.getLogger('laff')

def check_rise(data: pd.DataFrame, start: int, peak:int) -> bool:
    """Check the index of the rise, and that no points drop below flare start during the rise."""

    # Rise must be greater than x2 the start flux + flux error.
    if not data.iloc[peak].flux > data.iloc[start].flux + (2 * data.iloc[start].flux_perr):
        return False
    
    # During rise, no point should be below the start.
    start_flux = data['flux'].iloc[start]
    rise_fluxes = data['flux'].iloc[start+1:peak]
    if any(value < start_flux for value in rise_fluxes):
        return False
    
    return True



# def check_rise(data: pd.DataFrame, start: int, peak: int) -> bool:
#     """Test the rise is significant enough."""

#     x1 = data.iloc[start].time
#     y1 = data.iloc[start].flux

#     x2 = data.iloc[peak].time
#     y2 = data.iloc[peak].flux

#     alpha = np.log10(y1/y2) / np.log10(x2/x1)

#     print('ALPHA =', alpha)


#     if data.iloc[peak].flux > data.iloc[start].flux + (2 * data.iloc[start].flux_perr):
#         return True
#     else:
#         return False

# ##############################################################################################

# def check_noise(data: pd.DataFrame, start: int, peak: int, decay: int) -> bool:
#     """Check if flare is greater than x1.75 the average noise across the flare."""
#     average_noise = abs(np.average(data.iloc[start:decay].flux_perr)) + abs(np.average(data.iloc[start:decay].flux_nerr))
#     flux_increase = data.iloc[peak].flux - data.iloc[start].flux
#     logger.debug(f"noise: {average_noise} | delta_flux: {flux_increase}")
#     return True if flux_increase > 1.75 * average_noise else False

# ##############################################################################################

# def check_above(data: pd.DataFrame, start: int, decay: int) -> bool:
#     """
#     Check the flare is above the (estimated) continuum.
    
#     We calculate the powerlaw continuum through the flare by solving a set of
#     power functions for (x, y) co-ordinates corresponding to the found flare
#     start and end times. The number of points above and below the slope can then
#     be found. If the fraction above the continuum is below 0.7, to allow some
#     variation through noise, we discard the flare.

#     """
#     # Check flare boundaries.
#     start = 0 if start == 0 else start - 1
#     decay = data.idxmax('index').time if decay == data.idxmax('index').time else decay + 1

#     x_coords = data['time'].iloc[start], data['time'].iloc[decay]
#     y_coords = data['flux'].iloc[start], data['flux'].iloc[decay]

#     ## Solving y = nx^a for start and stop.
#     alpha = np.emath.logn(x_coords[1]/x_coords[0], y_coords[1]/y_coords[0])
#     norm = y_coords[1] / x_coords[1] ** alpha
#     # logger.debug(f"ALPHA IS {alpha}")
#     # logger.debug(f"NORM IS {norm}")
    
#     points_above = sum(flux > (norm*time**alpha) for flux, time in zip(data['flux'].iloc[start:decay], data['time'].iloc[start:decay]))
#     num_points = len(data['flux'].iloc[start:decay])

#     logger.debug(f"points above/num_points => {points_above}/{num_points} = {points_above/num_points}")

#     # return True
#     return True if points_above/num_points >= 0.7 else False

# ##############################################################################################

# def check_decay_shape(data: pd.DataFrame, peak: int, decay: int):

#     decay_data = list(data.iloc[peak:decay].flux_perr)
#     count_decrease = sum(b < a for a, b in zip(decay_data, decay_data[1:]))

#     decay_shape = count_decrease / len(decay_data)
#     logger.debug(f"decay shape {decay_shape}")
    
#     if len(decay_data) < 4 and decay_shape > 0.1:
#         return True
    
#     # print("DECAY SHAPE VALUE", decay_shape)
#     return True if decay_shape >= 0.5 else False