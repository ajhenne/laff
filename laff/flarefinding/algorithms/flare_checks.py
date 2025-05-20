import logging
import pandas as pd
import numpy as np

##
import matplotlib.pyplot as plt
##

logger = logging.getLogger('laff')


def check_noise(data: pd.DataFrame, start: int, peak: int) -> bool:
    """Flare rise must be greater than 4x the local noise."""

    flare_rise = data['flux'].iloc[peak] - data['flux'].iloc[start]
    noise_level = np.average(data['flux_perr'].iloc[start:peak])

    return flare_rise > 3 * noise_level

def check_slopes(data: pd.DataFrame, start: int, peak: int, decay: int) -> bool:
    """Check the fraction of increases/decreases during the rise and de cay, respectively."""

    increase_threshold = 0.60
    decrease_threshold = 0.65

    rise_flux = data['flux'].iloc[start:peak+1]
    increase_fraction = np.sum(np.diff(rise_flux) > 0) / len(rise_flux)

    # decay_flux = data['flux'].iloc[peak:decay+1]
    # decrease_fraction = np.sum(np.diff(decay_flux) < 0) / len(decay_flux)

    return True
    return increase_fraction >= increase_threshold # `and decrease_fraction > decrease_threshold


def check_above(data: pd.DataFrame, start: int, decay: int) -> bool:
    """
    Check the flare is above the (estimated) continuum.
    
    We calculate the powerlaw continuum through the flare by solving a set of
    power functions for (x, y) co-ordinates corresponding to the found flare
    start and end times. The number of points above and below the slope can then
    be found. If the fraction above the continuum is below 0.7, to allow some
    variation through noise, we discard the flare.

    """

    above_threshold = 0.8

    x_coords = data['time'].iloc[start], data['time'].iloc[decay]
    y_coords = data['flux'].iloc[start], data['flux'].iloc[decay]

    ## Solving y = nx^a for start and stop.
    alpha = np.emath.logn(x_coords[1]/x_coords[0], y_coords[1]/y_coords[0])
    norm = y_coords[1] / x_coords[1] ** alpha
    
    points_above = sum(flux > (norm*time**alpha) for flux, time in zip(data['flux'].iloc[start:decay], data['time'].iloc[start:decay]))
    num_points = len(data['flux'].iloc[start:decay])

    logger.debug(f"\tpoints above/num_points => {points_above}/{num_points} = {points_above/num_points}")

    return points_above/num_points > above_threshold




# def check_rise(data: pd.DataFrame, start: int, peak: int) -> bool:
#     """Check the index of the rise, and that no points drop below flare start during the rise."""

#     # During rise, no point should be below the start.
#     start_flux = data['flux'].iloc[start]
#     rise_fluxes = data['flux'].iloc[start+1:peak]
#     if any(value < start_flux for value in rise_fluxes):
#         return False


def check_variance(data: pd.DataFrame, peak: int, decay: int) -> bool:
    """Check the variance of the decay phase."""

    variance_threshold = 0.75

    decay_flux = data['flux'].iloc[peak:decay+1]
    peak_flux  = data['flux'].iloc[peak]
    variance = np.var(decay_flux) / peak_flux

    return variance < variance_threshold