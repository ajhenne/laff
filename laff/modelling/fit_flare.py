import numpy as np
import logging
from ..utility import calculate_fit_statistics

logger = logging.getLogger('laff')

### run and i get initial set crap

#################################################################################
### FRED MODEL
#################################################################################


def fred_flare(x, params):
    x = np.array(x)

    t_start = params[0]
    rise = params[1]
    decay = params[2]
    amplitude = params[3]

    cond = x < t_start


    model = amplitude * np.sqrt(np.exp(2*(rise/decay))) * np.exp(-(rise/(x-t_start))-((x-t_start)/decay))
    model[np.where(cond)] = 0

    # for idx, number in enumerate(model):
    #     if np.isinf(number):
    #         raise ValueError('Infinite value calculated in function fred.')
    #     if np.isnan(number):
    #         raise ValueError('NaN value calculcated in function fred.')

    return model

def fred_flare_wrapper(params, x):
    return fred_flare(x, params)

def all_flares(x, params):
    x = np.array(x)

    flare_params = [params[i:i+4] for i in range(0, len(params), 4)]
    
    sum_all_flares = [0.0] * len(x)

    for flare in flare_params:
        fit_flare = fred_flare(x, flare)
        sum_all_flares = [prev + current for prev, current in zip(sum_all_flares, fit_flare)]

    return sum_all_flares
    

#################################################################################
### SCIPY.ODR FITTING
#################################################################################

from scipy.odr import ODR, Model, RealData

def flare_fitter(data, residual, flares):

    logger.info("Fitting flares...")

    flareFits = []
    flareErrs = []

    for start, peak, end in zip(flares[0], flares[1], flares[2]):

        data_flare = RealData(residual.time[start:end], residual.flux[start:end], residual.time_perr[start:end], residual.flux_perr[start:end])
        
        # Parameter estimates.
        t_peak = residual['time'].iloc[peak]
        t_start = residual['time'].iloc[start]
        rise = t_peak - t_start
        decay = (residual['time'].iloc[end] - t_peak)
        amplitude = residual['flux'].iloc[peak] - residual['flux'].iloc[start]
        input_par = [t_start, rise, decay, abs(amplitude)]

        # Perform fit.
        logger.debug(f"For flare indices {start}/{peak}/{end}:")
        fit_par, fit_err = odr_fitter(data_flare, input_par)
        logger.debug(f"ODR Par: {fit_par}")
        logger.debug(f"ODR Err: {fit_err}")

        # Perform MCMC fit.
        logger.debug(f"Performing MCMC fit...")
        final_par, final_err = fit_flare_mcmc(residual, fit_par, fit_err)
        logger.debug(f"MCMC Par: {final_par}")
        logger.debug(f"MCMC Err: {final_err}")

        # Remove from residuals.
        fitted_flare = fred_flare(data.time, final_par)
        residual['flux'] -= fitted_flare

        logger.debug("Flare complete")

        flareFits.append(final_par)
        flareErrs.append(final_err)

    return flareFits, flareErrs

def odr_fitter(data, inputpar):
    model = Model(fred_flare_wrapper)

    odr = ODR(data, model, beta0=inputpar)

    odr.set_job(fit_type=0)
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

def fit_flare_mcmc(data, init_param, init_err):

    ndim = len(init_param)
    nwalkers = 25
    nsteps = 250

    p0 = np.zeros((nwalkers, ndim))

    guess_tstart = init_param[0]
    std_tstart = init_err[0] / 3.4
    p0[:, 0] = guess_tstart + std_tstart * np.random.randn(nwalkers)

    guess_rise = init_param[1]
    std_rise = init_err[1] / 3.4
    p0[:, 1] = guess_rise + std_rise * np.random.randn(nwalkers)

    guess_decay = init_param[2]
    std_decay = init_err[2] / 3.4
    p0[:, 2] = guess_decay + std_decay * np.random.randn(nwalkers)

    guess_amplitude = init_param[3]
    std_ampltiude = init_err[3] / 3.4
    p0[:, 3] = guess_amplitude + std_ampltiude * np.random.randn(nwalkers)

    logger.debug("Running flare MCMC...")

    sampler = emcee.EnsembleSampler(nwalkers, ndim, fl_log_posterior, \
        args=(data.time, data.flux, data.time_perr, data.flux_perr))
    sampler.run_mcmc(p0, nsteps)

    burnin = 50

    samples = sampler.chain[:, burnin:, :].reshape(-1, ndim)

    fitted_par = list(map(lambda v: np.median(v), samples.T))
    fitted_err = list(map(lambda v: np.std(v), samples.T))

    logger.debug("MCMC run completed.")

    return fitted_par, fitted_err

def fl_log_likelihood(params, x, y, x_err, y_err):
    model = fred_flare(x, params)
    chisq = np.sum(( (y-model)**2) / ((y_err)**2)) 
    log_likelihood = -0.5 * np.sum(chisq + np.log(2 * np.pi * y_err**2))
    return log_likelihood

def fl_log_prior(params, TIME_END):

    t_start = params[0]
    rise = params[1]
    decay = params[2]
    amplitude = params[3]

    if not (t_start > 0) and (t_start < TIME_END):
        return -np.inf

    if rise > TIME_END or rise < 0:
        return -np.inf

    if decay > TIME_END or decay < 0:
        return -np.inf

    if amplitude < 0:
        return -np.inf

    return 0.0

def fl_log_posterior(params, x, y, x_err, y_err):
    lp = fl_log_prior(params, x.iloc[-1])
    ll = fl_log_likelihood(params, x, y, x_err, y_err)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ll