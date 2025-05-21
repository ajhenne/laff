import numpy as np
import logging
from scipy.optimize import fmin_slsqp
from ..utility import calculate_fit_statistics, calculate_par_err, calculate_fluence

logger = logging.getLogger('laff')

#################################################################################
### AFTERGLOW MODEL
#################################################################################

def broken_powerlaw(params, x):
    x = np.array(x)

    if type(params) in (list, np.ndarray):
        n = int((len(params)-2)/2)
        slopes = params[0:n+1]
        breaks = params[n+1:-1]
        norm   = params[-1]
    elif type(params) is dict:
        n      = params['break_num']
        slopes = params['slopes']
        breaks = params['breaks']
        norm   = params['normal']
    else:
        logger.critical('Input params not accepted type.')
        raise TypeError(f'params is not dict/list -> {type(params)}')
    
    breaks = [10**val for val in breaks]

    mask = []

    for i in range(n):
        mask.append(x > breaks[i])

    if n >= 0:
        model = norm * (x**(-slopes[0]))
    if n >= 1 and mask[0].any():
        model[np.where(mask[0])] = norm * (x[np.where(mask[0])]**(-slopes[1])) * (breaks[0]**(-slopes[0]+slopes[1]))
    if n >= 2 and mask[1].any():
        model[np.where(mask[1])] = norm * (x[np.where(mask[1])]**(-slopes[2])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2]))
    if n >= 3 and mask[2].any():
        model[np.where(mask[2])] = norm * (x[np.where(mask[2])]**(-slopes[3])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3]))
    if n >= 4 and mask[3].any():
        model[np.where(mask[3])] = norm * (x[np.where(mask[3])]**(-slopes[4])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3])) * (breaks[3]**(-slopes[3]+slopes[4]))
    if n >= 5 and mask[4].any():
        model[np.where(mask[4])] = norm * (x[np.where(mask[4])]**(-slopes[5])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3])) * (breaks[3]**(-slopes[3]+slopes[4])) * (breaks[4]**(-slopes[4]+slopes[5]))

    return model

def sum_residuals(params, *args):
    x, y, y_err, _, _ = args
    return np.sum(((y - broken_powerlaw(params, x)) / y_err) ** 2)    

#################################################################################
### FITTING
#################################################################################

def find_afterglow_fit(data, data_flare):

    data_start, data_end = data['time'].iloc[0], data['time'].iloc[-1]

    model_fits = []

    logger.debug('breaknum : fit_par / fit_err / fit_stats')
    for breaknum in range(0, 6):

        # Guess parameters.
        slope_guesses = [1.0] * (breaknum+1)
        break_guesses = list(np.linspace(np.log10(data_start), np.log10(data_end), num=breaknum+2))[1:-1]
        normal_guess  = [data['flux'].iloc[0]]
        input_par = slope_guesses + break_guesses + normal_guess

        # Parameter bounds.
        if breaknum == 0:
            bounds = ([-0.3, 6.0], [0, np.inf])
        else:
            bounds = tuple([[-0.3, 6.0]] * (breaknum + 1) + [[np.log10(data['time'].iloc[0])+0.05, np.log10(data['time'].iloc[-1])-0.05]] * breaknum + [[0, np.inf]])
        
        # Constraints.
        def all_constraints(params, *args):
            _, _, _, flare_x, flare_y = args
            ordered_breaks = np.diff(params[breaknum+1:-1]) - 0.05 # order breaks
            flare_limits = flare_y - broken_powerlaw(params, flare_x) # flares as upper lims
            return np.concatenate([ordered_breaks, flare_limits])
        
        fit_par = fmin_slsqp(sum_residuals, input_par, bounds=bounds, f_ieqcons=all_constraints, args=(data.time, data.flux, data.flux_perr, data_flare[0], data_flare[1]), iter=500, iprint=0)
        fit_stats = calculate_fit_statistics(data, broken_powerlaw, fit_par)
        model_fits.append([fit_par, fit_stats])
        
        #####

    # Assess best fit.
    best_fit, best_stats = min(model_fits, key=lambda x: x[1]['deltaAIC'])
    breaknum = int((len(best_fit)-2)/2)

    def chi2_wrapper(params):
        return sum_residuals(params, data['time'], data['flux'], data['flux_perr'], 1, 1)
    
    param_errors = calculate_par_err(best_fit, chi2_wrapper)
    
    logger.info('Afterglow fitted with %s breaks.', breaknum)
    logger.debug('slopes\t%s', list(round(x,2) for x in best_fit[0:breaknum+1]))
    logger.debug('slopes_err\t%s', list(round(x,2) for x in param_errors[0:breaknum+1]))
    logger.debug('breaks\t%s', list(round(10**x,2) for x in best_fit[breaknum+1:-1]))
    logger.debug('breaks_err\t%s', list(round(10**x,2) for x in param_errors[breaknum+1:-1]))
    logger.debug('norm\t%s', best_fit[-1])
    logger.debug('norm\t%s', param_errors[-1])

    return list(best_fit), list(param_errors), best_stats, breaknum

#################################################################################
# CALCULATE FLUENCE
#################################################################################

def calculate_afterglow_fluence(data, breaknum, break_times, fit_par, count_ratio):

    break_times = [10**(x) for x in break_times]

    integral_boundaries = [data.iloc[0].time, *break_times, data.iloc[-1].time]

    phase_fluences = [calculate_fluence(broken_powerlaw, fit_par, integral_boundaries[i], integral_boundaries[i+1], count_ratio) * count_ratio for i in range(breaknum)]
    total_fluence = np.sum(phase_fluences)
    logger.info('Afterglow fluence calculated as %s', total_fluence)

    return total_fluence, phase_fluences