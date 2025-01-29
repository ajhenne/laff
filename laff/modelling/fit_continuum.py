import numpy as np
import logging
from scipy.odr import ODR, Model, RealData

from ..utility import calculate_fit_statistics, calculate_fluence

logger = logging.getLogger('laff')

#################################################################################
### AFTERGLOW MODEL
#################################################################################

def broken_powerlaw(params, x):

    n = (len(params)-2)/2

    x = np.array(x)
    slopes = params[0:n+1]
    breaks = params[n+1:-1]
    norm   = params[-1]

    mask = []

    for i in range(n):
        mask.append(x > breaks[i])

    if n >= 0:
        model = norm * (x**(-slopes[0]))
    if n >= 1:
        model[np.where(mask[0])] = norm * (x[np.where(mask[0])]**(-slopes[1])) * (breaks[0]**(-slopes[0]+slopes[1]))
    if n >= 2:
        model[np.where(mask[1])] = norm * (x[np.where(mask[1])]**(-slopes[2])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2]))
    if n >= 3:
        model[np.where(mask[2])] = norm * (x[np.where(mask[2])]**(-slopes[3])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3]))
    if n >= 4:
        model[np.where(mask[3])] = norm * (x[np.where(mask[3])]**(-slopes[4])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3])) * (breaks[3]**(-slopes[3]+slopes[4]))
    if n >= 5:
        model[np.where(mask[4])] = norm * (x[np.where(mask[4])]**(-slopes[5])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3])) * (breaks[3]**(-slopes[3]+slopes[4])) * (breaks[4]**(-slopes[4]+slopes[5]))

    return model

def broken_powerlaw_wrapper(params, x):
    """Wrapper for ODR to fit log10 break times. Intended for internal use only."""

    n = (len(params)-2)/2
    params[n+1:-1] = [10**val for val in params[n+1:-1]]

    return broken_powerlaw(params, x)
    

#################################################################################
### SCIPY ODR FITTING
#################################################################################

def odr_fitter(data, input_par, convert_to_std=1.0, t_max=0):
    """Fitting function for powerlaw."""
    data = RealData(data.time, data.flux, sx=data.time_perr*convert_to_std, sy=data.flux_perr*convert_to_std)
    model = Model(broken_powerlaw_wrapper)
    n = (len(input_par)-2)/2

    odr = ODR(data, model, beta0=input_par)

    output = odr.run()

    if output.info != 1:
        i = 1
        while output.info != 1 and not all(output.beta[n+1:-1] > 0) and not all(output.beta[n+1:-1] < t_max):
            # require 0 < break_time < t_max
            output = odr.restart()
            i += 1
            if i >= 100:
                raise ValueError("ODR failed converge within bounds")
            
    return output.beta, output.sd_beta


def find_afterglow_fit(data, conversion_to_std):

    data_start, data_end = data['time'].iloc[0], data['time'].iloc[-1]

    model_fits = []

    logger.debug('breaknum : fit_par / fit_err / fit_stats')
    for breaknum in range(0, 6):

        slope_guesses = [1.0] * (breaknum+1)
        break_guesses = list(np.linspace(np.log10(data_start), np.log10(data_end, num=breaknum+2)))[1:-1]
        normal_guess  = [data['flux'].iloc[0] * data['time'].iloc[0]]
        input_par = slope_guesses + break_guesses + normal_guess

        fit_par, fit_err = odr_fitter(data, input_par, n=breaknum, convert_to_std=conversion_to_std, t_max=data_end)
        fit_stats = calculate_fit_statistics(data, broken_powerlaw, fit_par)
        logger.debug('%s : %s / %s / %s', breaknum, fit_par, fit_err, fit_stats)

        # Check breaks are in order - remove later.
        if fit_par[breaknum+1:-1] != sorted(fit_par[breaknum+1:-1]):
            raise ValueError("Out of order breaks.")
        fit_par[breaknum+1:-1] = [10**x for x in fit_par[breaknum+1:-1]]

        model_fits.append([fit_par, fit_err, fit_stats])

    best_fit, best_err, best_stats = min(model_fits, key=lambda x: x[2]['deltaAIC'])
    breaknum = (len(fit_par)-2)/2
    logger.info('Afterglow fitted with %s breaks.', breaknum)

    return best_fit, best_err, best_stats, breaknum


#################################################################################
# CALCULATE FLUENCE
#################################################################################

def calculate_afterglow_fluence(data, breaknum, break_times, fit_par, count_to_flux_conversion):

    integral_boundaries = [data.iloc[0].time, *break_times, data.iloc[-1].time]
    afterglow_fluence = np.sum([calculate_fluence(broken_powerlaw, fit_par, integral_boundaries[i], integral_boundaries[i+1]) for i in range(breaknum)])
    afterglow_fluence *= count_to_flux_conversion
    logger.info('Afterglow fluence calculated as %s', afterglow_fluence)

    return afterglow_fluence