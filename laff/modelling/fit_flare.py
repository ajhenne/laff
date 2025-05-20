import numpy as np
import logging
import emcee
from scipy.optimize import fmin_slsqp
from ..utility import calculate_fit_statistics, calculate_par_err, calculate_fluence 
from ..modelling import broken_powerlaw

logger = logging.getLogger('laff')

#################################################################################
### FRED MODEL
#################################################################################

def fred_flare(params, x):
    # J. P. Norris et al., ‘Attributes of Pulses in Long Bright Gamma-Ray Bursts’, The Astrophysical Journal, vol. 459, p. 393, Mar. 1996, doi: 10.1086/176902.
    
    x = np.array(x)
    t_max = params[0]
    rise = params[1]
    decay = params[2]
    sharpness = params[3]
    amplitude = params[4]

    model = amplitude * np.exp( -(abs(x - t_max) / rise) ** sharpness)
    model[np.where(x > t_max)] = amplitude * np.exp( -(abs(x[np.where(x > t_max)] - t_max) / decay) ** sharpness)

    return model

def sum_residuals(params, *args):
    x, y, y_err = args
    return np.sum(((y - fred_flare(params, x)) / y_err) ** 2)

#################################################################################
### SCIPY.ODR FITTING
#################################################################################

def flare_fitter(data, continuum, flares, model='fred'):
    """ 
    Flare fitting function. Takes already found flare indices and models them.

    Also runs:
      - 
    
    """

    # import matplotlib.pyplot as plt ## temp

    logger.info("Fitting flares...")

    data['residuals'] = data['flux'] - broken_powerlaw(continuum['params'], data['time'])

    # plt.figure(figsize=(10,8))
    # plt.scatter(data.time, data.residuals, marker='.', color='grey')
    # plt.axhline(0, color='grey', linestyle='--')
    # plt.semilogx()

    flareFits  = []
    flareStats = []
    flareErrs  = []

    for start, peak, end in flares:

        t_start = data['time'].iloc[start]
        t_peak = data['time'].iloc[peak]
        t_end = data['time'].iloc[end]

        # Parameter guesses.
        t_peak = t_peak
        rise  = (t_peak - t_start) / 4
        decay = (t_end - t_peak) / 2
        sharpness = decay / rise
        amplitude = data['flux'].iloc[peak]
        input_par = [t_peak, rise, decay, sharpness, amplitude]

        bounds = [t_start, t_end], [0.0, t_end-t_start], [0.0, t_end-t_start], [1.0, 10.0], [0.0, np.inf]

        fitted_flare = fmin_slsqp(sum_residuals, input_par, bounds=bounds, args=(data.time, data.residuals, data.flux_perr), iter=100)

        ## temp
        # plt.plot(data.time, fred_flare(fitted_flare, data.time))

        fitted_stats = calculate_fit_statistics(data, fred_flare, fitted_flare)

        # get errors
        def chi2_wrapper(params):
            return sum_residuals(params, data['time'], data['residuals'], data['flux_perr'])
        param_errors = calculate_par_err(fitted_flare, chi2_wrapper)
        
        data['residuals'] = data['residuals'] - fred_flare(fitted_flare, data.time)

        logger.debug(f"Flare {start}/{peak}/{end} fitted")
        logger.debug('params\t%s', list(round(x, 2) for x in fitted_flare))
        logger.debug('errors\t%s', list(round(x, 2) for x in param_errors))

        flareFits.append(list(fitted_flare))
        flareStats.append(list(fitted_stats))
        flareErrs.append(list(param_errors))

    logger.info("Flare fitting complete for all flares.")
    return flareFits, flareStats, flareErrs

#################################################################################
### NEAT PACKAGING
#################################################################################

def package_flares(data, fits, stats, errs, indices, count_ratio=1.0):

    flaresDict = []

    for idx, fit, stat, err in zip(indices, fits, stats, errs):

        # tmax, rise, decay, sharp, ampl
        # 0     1     2      3      4
        fitted_flare = fred_flare(fit, data['time'])

        start_time = fit[0] - (fit[1] * (-np.log(0.001)) **(1/fit[3]))
        peak_time  = fit[0]
        end_time = fit[0] + (fit[2] * (-np.log(0.001)) **(1/fit[3]))
        times = [start_time, peak_time, end_time]

        fluence_rise  = calculate_fluence(fred_flare, fit, start_time, peak_time, count_ratio=count_ratio)
        fluence_decay = calculate_fluence(fred_flare, fit, peak_time, end_time, count_ratio=count_ratio)
        fluences = [fluence_rise, fluence_decay, fluence_rise + fluence_decay]

        peak_flux = max(fitted_flare)
        
        flaresDict.append({'times': times, 'indices': idx, 'params': fit, 'stats': stat, 'errors': err, 'fluence': fluences, 'peak_flux': peak_flux})

    return flaresDict