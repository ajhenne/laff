import numpy as np
import logging
from ..utility import calculate_fit_statistics

logger = logging.getLogger('laff')

#################################################################################
### CONTINUUM MODEL
#################################################################################

def broken_powerlaw(x, params):
    
    x = np.array(x)

    if type(params) is dict:
        n = params['break_num']
        slopes = params['slopes']
        breaks = params['breaks']
        normal = params['normal']
    elif type(params) is np.ndarray or type(params) is list:
        nparam = len(params)
        n = int((nparam-2)/2)
        slopes = params[:n+1]
        breaks = params[n+1:-1]
        normal = params[-1]
    else:
        logger.critical("Input parameter is not correct type!")
        raise TypeError(f'params is not dict/list -> {type(params)}')
    
    mask = []

    for i in range(n):
        try:
            mask.append(x > breaks[i])
        except:
            logger.critical(i, 'too many?') # what's this for again?
            pass

    if n >= 0:
        model = normal * (x**(-slopes[0]))
    if n >= 1:
        model[np.where(mask[0])] = normal * (x[np.where(mask[0])]**(-slopes[1])) * (breaks[0]**(-slopes[0]+slopes[1]))
    if n >= 2:
        model[np.where(mask[1])] = normal * (x[np.where(mask[1])]**(-slopes[2])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2]))
    if n >= 3:
        model[np.where(mask[2])] = normal * (x[np.where(mask[2])]**(-slopes[3])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3]))
    if n >= 4:
        model[np.where(mask[3])] = normal * (x[np.where(mask[3])]**(-slopes[4])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3])) * (breaks[3]**(-slopes[3]+slopes[4]))
    if n >= 5:
        model[np.where(mask[4])] = normal * (x[np.where(mask[4])]**(-slopes[5])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3])) * (breaks[3]**(-slopes[3]+slopes[4])) * (breaks[4]**(-slopes[4]+slopes[5]))

    return model

def broken_powerlaw_wrapper(params, x):
    """Wrapper function for broken_powerlaw - for ODR fitting."""
    return broken_powerlaw(x, params)

#################################################################################
### SCIPY.ODR FITTING
#################################################################################

from scipy.odr import ODR, Model, RealData

def find_intial_fit(data):
    data_start, data_end = data['time'].iloc[0], data['time'].iloc[-1]
    model_fits = []

    for breaknum in range(0, 6, 1):

        # Guess parameters.
        slope_guesses = [1.0] * (breaknum+1)
        break_guesses = list(np.array(np.logspace(np.log10(data_start) * 1.1, np.log10(data_end) * 0.9, num=breaknum)))
        normal_guess  = [data['flux'].iloc[0] * data['time'].iloc[0]]
        input_par = slope_guesses + break_guesses + normal_guess

        # Perform fit.
        fit_par, fit_err = odr_fitter(data, input_par)

        # Ensure breaks are sorted.
        n = int((len(fit_par)-2)/2)
        fit_par[n+1:-1] = sorted(fit_par[n+1:-1])

        # Evaluate fit.
        fit_stats = calculate_fit_statistics(data, broken_powerlaw, fit_par)
        deltaAIC = fit_stats['deltaAIC']

        model_fits.append([fit_par, deltaAIC, fit_err, fit_stats])

    # Obtain best fitted mdel.
    best_fit, best_aic, best_err, best_stats = min(model_fits, key=lambda x: x[1])

    nparam = len(best_fit)
    n = int((nparam-2)/2)
    logger.info(f"Initial continuum fit found {n} breaks.")

    logger.debug("ODR initial fit parameters.")
    logger.debug(f'Slopes: {list(best_fit[:n+1])}')
    logger.debug(f'Breaks: {list(best_fit[n+1:-1])}')
    logger.debug(f'Normal: {best_fit[-1]}')

    return best_fit, best_err, best_stats

def odr_fitter(data, inputpar):
    data = RealData(data.time, data.flux, data.time_perr, data.flux_perr)
    model = Model(broken_powerlaw_wrapper)

    odr = ODR(data, model, beta0=inputpar)

    odr.set_job(fit_type = 0)
    output = odr.run()

    if output.info != 1:
        i = 1
        while output.info != 1 and i < 100:
            output = odr.restart()
            i += 1
            
    return output.beta, output.sd_beta

#################################################################################
### MCMC FITTING
#################################################################################

import emcee

def fit_continuum_mcmc(data, breaknum, init_param, init_err):

    ndim = 2 * breaknum + 2
    nwalkers = 25
    nsteps = 500

    nparam = len(init_param)
    n = int((nparam-2)/2)

    # Parameter priors.
    p0 = np.zeros((nwalkers, ndim))

    # Slopes.
    guess_slopes = init_param[:n+1]
    std_slopes = init_err[:n+1] / 3.4
    for i in range(0, breaknum+1):
        p0[:, i] = guess_slopes[i] + std_slopes[i] * np.random.randn(nwalkers)

    # Breaks.
    guess_breaks = init_param[n+1:-1]
    std_breaks = init_err[n+1:-1] / 3.4
    for breaknum, i in enumerate(range(breaknum+1, ndim-1)):
        p0[:, i] = guess_breaks[breaknum] + std_breaks[breaknum] * np.random.randn(nwalkers)

    # Normalisation.
    guess_norm = init_param[-1]
    std_norm = init_err[-1]
    p0[:, -1] = guess_norm + std_norm * np.random.randn(nwalkers)

    logger.info("Running continuum MCMC...")

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, \
        args=(data.time, data.flux, data.time_perr, data.flux_perr))
    try:
        sampler.run_mcmc(p0, nsteps)
    except:
        sampler.run_mcmc(p0, nsteps * 3)

    burnin = 100

    samples = sampler.chain[:, burnin:, :].reshape(-1, ndim)

    fitted_par = list(map(lambda v: np.median(v), samples.T))
    fitted_err = list(map(lambda v: np.std(v), samples.T))

    logger.info("Continuum fitting complete.")

    return fitted_par, fitted_err

def log_likelihood(params, x, y, x_err, y_err):
    model = broken_powerlaw(x, params)
    # residual = y - model
    # chi_squared = np.sum((residual/ y_err) ** 2)
    # log_likelihood = -0.5 * (len(x) * np.log(2*np.pi) + np.sum(np.log(y_err ** 2)) + chi_squared)
    chisq = np.sum(( (y-model)**2) / ((y_err)**2)) 
    log_likelihood = -0.5 * np.sum(chisq + np.log(2 * np.pi * y_err**2))
    return log_likelihood

def log_prior(params, TIME_END):

    nparam = len(params)
    n = int((nparam-2)/2)

    slopes = params[:n+1]
    breaks = params[n+1:-1]
    normal = params[-1]

    if not all(-2 < value < 4 for value in slopes):
        return -np.inf

    if any(value < 0 for value in breaks):
        return -np.inf

    if any(value > TIME_END for value in breaks):
        return -np.inf

    if (normal < 0):
        return -np.inf

    return 0.0

def log_posterior(params, x, y, x_err, y_err):
    lp = log_prior(params, x.iloc[-1])
    ll = log_likelihood(params, x, y, x_err, y_err)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ll