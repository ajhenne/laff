import numpy as np
import logging
import emcee
from intersect import intersection
from ..utility import calculate_fit_statistics
from ..modelling import broken_powerlaw

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

def flare_fitter(data, continuum, flares, use_odr=False):
    """ 
    Flare fitting function. Takes already found flare indices and models them.

    Also runs:
      - 
    
    """
    logger.info("Fitting flares...")

    logger.debug("Calculating residuals")    
    fitted_model = broken_powerlaw(data.time, continuum['parameters'])
    residuals = data.copy()
    residuals['flux'] = data.flux - fitted_model

    flareFits = []
    flareErrs = []

    for start, peak, end in flares:

        data_flare = data.copy()
        data_flare['flux'] = np.float64(0)
        data_flare.loc[start:end, 'flux'] = residuals.loc[start:end, 'flux']
        # Parameter estimates.
        t_peak = residuals['time'].iloc[peak]
        t_start = residuals['time'].iloc[start]
        rise = t_peak - t_start
        decay = (residuals['time'].iloc[end] - t_peak)
        amplitude = abs(residuals['flux'].iloc[peak] - residuals['flux'].iloc[start])
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

        logger.debug(f"Final par: {final_par}")
        logger.debug(f"Final err: {final_err}")

        # Remove from residuals.
        fitted_flare = fred_flare(data.time, final_par)
        residuals['flux'] -= fitted_flare

        logger.debug("Flare complete")

        flareFits.append(list(final_par))
        flareErrs.append(list(final_err))

    logger.info("Flare fitting complete for all flares.")
    return flareFits, flareErrs


def improved_end_time(data, flare_indices, flare_par, continuum_par):
    """
    Find the end of the flares based on flare converging to continuum.
    
    Calculate the value of the fitted continuum model to the flare+continuum
    model for a series of values until the models converge. Since the FRED
    model is asymptotic to 0, we apply 'factor' to the flare+continuum, an
    small effective downshift.
    
    Parameters:
    data (pd.DataFrame): GRB lightcurve data.
    flare_indices (List[int]): a nested list of indices corresponding to flare
                               start, stop and end times.
    flare_par (List[float]): a list of the flare model parameters.
    continuum_par (List[floart]): a list of the continuum model parameters.

    Returns:
    end_index (int): the newly found index for the end of the flare.
    end_time (float): the newly found time for the end of the flare.
    """

    logger.debug(f"Running improved_end_time for flare: {flare_indices}.")
    peak, end = flare_indices[1:]
    factor = 0.99 # seems to be the most consistent value.

    if end == len(data.time) - 1:
        logger.debug("End of flare already at end of data. Returning original inputs.")
        return end, data['time'].iloc[end]

    # Calculate value for each model at current flare end.
    current_continuum = broken_powerlaw([data['time'].iloc[end]], continuum_par)[0]
    current_flare = fred_flare([data['time'].iloc[end]], flare_par)[0]
    current_flare += (current_continuum * factor)

    # Loop until the flare+continuum model drops below the continuum.
    extend = 0
    while current_flare > current_continuum and end + extend < len(data.time) -1:
        extend += 1
        logger.debug(f"Conditions not yet met -- extending to {extend}")
        current_continuum = broken_powerlaw([data['time'].iloc[end+extend]], continuum_par)[0]
        current_flare = fred_flare([data['time'].iloc[end+extend]], flare_par)[0]
        current_flare += (current_continuum * factor)
    end_index = end + extend
    logger.debug(f"Conditions met! Calculating intercept.")
        
    # Calculate the intercept time.
    search_time = np.logspace(
        np.log10(data['time'].iloc[peak+1]), 
        np.log10(data['time'].iloc[end+extend+1]),
        num = 1000)
    
    continuum_model = broken_powerlaw(search_time, continuum_par)
    flare_model = fred_flare(search_time, flare_par) + (continuum_model * factor)
    end_time, _ = intersection(search_time, continuum_model, search_time, flare_model)

    if len(end_time) > 1:
        logger.warning("Multiple intercepts found for this burst - the flare \
                       may be incorrect or modelled badly.")
        end_time = end_time[-1]
        # import matplotlib.pyplot as plt
        # plt.plot(search_time, continuum_model, color='tab:orange')
        # plt.plot(search_time, flare_model, color='r')
        # plt.scatter(data.time, data.flux, color='b', marker='.')
        # plt.loglog()
        # plt.show()
        # raise ValueError("It appears two intercepts were found!")
        ## This can occur when the found flare isn't really a flare. Noisy data
        ## etc will cause the flare to be modelled wrong, and so multiple
        ## intersections happen. I can't think of a scenario otherwise? There
        ## should only ever be 2, so just select the later one in this case.
        ## Maybe I can remove this check with better flare finding?
    logger.debug(f"Found new end information: index {end_index}, end_time {end_time}.")

    return end_index, end_time


#################################################################################
### FITTING METHODS
#################################################################################

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