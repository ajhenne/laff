import numpy as np
import logging
from ..utility import calculate_fit_statistics

logger = logging.getLogger('laff')

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

def flare_fitter(data, residual, flares, use_odr=False):

    logger.info("Fitting flares...")

    flareFits = []
    flareErrs = []

    for start, peak, end in flares:

        data_flare = data.copy()
        data_flare['flux'] = 0
        data_flare.loc[start:end, 'flux'] = [float(x) for x in residual.loc[start:end, 'flux']]
        # Parameter estimates.
        t_peak = residual['time'].iloc[peak]
        t_start = residual['time'].iloc[start]
        rise = t_peak - t_start
        decay = (residual['time'].iloc[end] - t_peak)
        amplitude = abs(residual['flux'].iloc[peak] - residual['flux'].iloc[start])
        input_par = [t_start, rise, decay, amplitude]

        # Perform fit.
        logger.debug(f"For flare indices {start}/{peak}/{end}:")
        fit_par, fit_err = odr_fitter(data_flare, input_par)
        fit_par = [abs(x) for x in fit_par]
        odr_fit_par = calculate_fit_statistics(data, fred_flare, fit_par)
        odr_rchisq = odr_fit_par['rchisq']
        logger.debug(f"ODR Par: {fit_par}")
        logger.debug(f"ODR Err: {fit_err}")

        try:
            final_par, final_err = fit_flare_mcmc(data_flare, fit_par, fit_err)
            final_fit_statistics = calculate_fit_statistics(data, fred_flare, final_par)
            mcmc_rchisq = final_fit_statistics['rchisq']

            if mcmc_rchisq == 0 or mcmc_rchisq < 0.1 or mcmc_rchisq == np.inf or mcmc_rchisq == -np.inf:
                logger.debug(f'MCMC appears to be bad, using ODR fit for flare {start}-{end}.')
                final_par, final_err, final_fit_statistics = fit_par, fit_err, odr_fit_par

            elif abs(odr_rchisq-1) < abs(mcmc_rchisq-1):
                if abs(odr_rchisq) < 1.3 * abs(mcmc_rchisq-1):
                    logger.debug(f"ODR better than MCMC for flare {start}-{end}, using ODR.")
                    final_par, final_err, final_fit_statistics = fit_par, fit_err, odr_fit_par
                else:
                    logger.debug(f"ODR better than MCMC fit for flare {start}-{end}, but not significantly enough.")

        except ValueError:
            logger.debug(f'MCMC failed - using ODR fit.')
            final_par, final_err = fit_par, fit_err

        logger.debug(f"MCMC Par: {final_par}")
        logger.debug(f"MCMC Err: {final_err}")

        # Remove from residuals.
        fitted_flare = fred_flare(data.time, final_par)
        residual['flux'] -= fitted_flare

        logger.debug("Flare complete")

        flareFits.append(list(final_par))
        flareErrs.append(list(final_err))

    logger.info("Flare fitting complete for all flares.")
    return flareFits, flareErrs

def odr_fitter(data, inputpar):
    data = RealData(data.time, data.flux, data.time_perr, data.flux_perr)  
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
    nwalkers = 30
    nsteps = 200

    p0 = np.zeros((nwalkers, ndim))

    guess_tstart = init_param[0]
    std_tstart = init_param[0]
    p0[:, 0] = guess_tstart + std_tstart * np.random.randn(nwalkers)

    guess_rise = init_param[1]
    std_rise = init_param[1]
    p0[:, 1] = guess_rise + std_rise * np.random.randn(nwalkers)

    guess_decay = init_param[2]
    std_decay = init_param[2]
    p0[:, 2] = guess_decay + std_decay * np.random.randn(nwalkers)

    guess_amplitude = init_param[3]
    std_ampltiude = init_param[3]
    p0[:, 3] = guess_amplitude + std_ampltiude * np.random.randn(nwalkers)

    logger.debug("Running flare MCMC...")

    sampler = emcee.EnsembleSampler(nwalkers, ndim, fl_log_posterior, \
        args=(data.time, data.flux, data.time_perr, data.flux_perr))
    sampler.run_mcmc(p0, nsteps)

    burnin = 25

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

    if rise < 0:
        return -np.inf
    if rise > TIME_END:
        return -np.inf

    if decay < 0:
        return -np.inf
    if decay > TIME_END:
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