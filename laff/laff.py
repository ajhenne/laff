import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import warnings

# Ignore warnings.
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .flarefinding import sequential_findflares
from .modelling import find_intial_fit, fit_continuum_mcmc, flare_fitter, broken_powerlaw, fred_flare
from .utility import check_data_input, calculate_fit_statistics, calculate_fluence

# findFlares() -- locate the indices of flares in the lightcurve
# fitContinuum(flare_indices) -- use the indices to exclude data, then fit the continuum
# fitFlares(flares, continuum) -- use indices + continuum to fit the flares

# fitGRB() -- runs all 3 function in sequence, then does some final cleanups
#          -- final statistics of the whole fit
#          -- this is what the user should be running
#          -- outputs a dictionary with all useful statistics

#################################################################################
### LOGGER
#################################################################################

logging_level = 'INFO'
logger = logging.getLogger('laff')
logger.setLevel(logging_level)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def set_logging_level(level):
    if level.lower() in ['debug', 'info', 'warning', 'error', 'critical']:
        logging_level = level.upper()
        logger.setLevel(logging_level)
    elif level.lower() == 'none':
        logger.setLevel(60) # set to above all other levels
    else:
        raise ValueError("Invalid logging level. Please use DEBUG, INFO, WARNING, ERROR, CRITICAL or NONE.")

#################################################################################
### FIND FLARES
#################################################################################

def findFlares(data):
    logger.debug("Starting sequential_findflares()")
    check_data_input(data) # First check input format is good.

    # Cutoff late data.
    LATE_CUTOFF = True
    data = data[data.time < 2000] if LATE_CUTOFF else data

    

    # Run flare finding.
    flares = sequential_findflares(data)

    logger.info(f"Found {len(flares)} flare(s).")
    return flares if len(flares) else False

#################################################################################
### CONTINUUM FITTING
#################################################################################

def fitContinuum(data: pd.DataFrame, flare_indices: list, count_ratio: float) -> dict:
    logger.debug(f"Starting fitContinuum")

    # Remove flare data.
    if flare_indices:
        for start, _, end in flare_indices:
            data = data.drop(index=range(start, end))

    # Use ODR & AIC to find best number of powerlaw breaks.
    initial_fit, initial_fit_err, initial_fit_stats = find_intial_fit(data)
    break_number = int((len(initial_fit-2)/2)-1)

    # Try an MCMC fitting run.
    try:
        final_par, final_err = fit_continuum_mcmc(data, break_number, initial_fit, initial_fit_err)
    except ValueError:
        final_par, final_err = initial_fit, initial_fit_err
        logger.debug(f"MCMC failed, defaulting to ODR.")

    # Calculate fit statistics.
    final_fit_statistics = calculate_fit_statistics(data, broken_powerlaw, final_par)
    odr_rchisq = initial_fit_stats['rchisq']
    mcmc_rchisq = final_fit_statistics['rchisq']
    logger.debug(f'ODR rchisq: {odr_rchisq}')
    logger.debug(f'MCMC rchisq: {mcmc_rchisq}')

    # Compare MCMC and ODR fits.
    if mcmc_rchisq == 0 or mcmc_rchisq < 0.1 or mcmc_rchisq == np.inf or mcmc_rchisq == -np.inf:
        logger.debug('MCMC appears to be bad, using ODR fit.')
        final_par, final_err, final_fit_statistics = initial_fit, initial_fit_err, initial_fit_stats
    elif abs(odr_rchisq-1) < abs(mcmc_rchisq-1):
        if abs(odr_rchisq-1) < 1.3 * abs(mcmc_rchisq-1):
            logger.debug("ODR better than MCMC, using ODR fit.")
            final_par, final_err, final_fit_statistics = initial_fit, initial_fit_err, initial_fit_stats
        else:
            logger.debug("ODR better than MCMC fit, but not significantly enough.")

    slopes = list(final_par[:break_number+1])
    slopes_err = list(final_err[:break_number+1])
    breaks = list(final_par[break_number+1:-1])
    breaks_err = list(final_err[break_number+1:-1])
    normal = final_par[-1]
    normal_err = final_err[-1]

    if break_number == 0:
        breakpoints = [data.iloc[0].time, data.iloc[-1].time]
    elif (data.iloc[0].time < breaks[0]) and (breaks[-1] < data.iloc[-1].time):
        breakpoints = [data.iloc[0].time, *breaks, data.iloc[-1].time]
    elif (data.iloc[0].time > breaks[0]):
        breakpoints = [data.iloc[0].time, *breaks[1:], data.iloc[-1].time]
    elif (breaks[-1] > data.iloc[-1].time):
        breakpoints = [data.iloc[0].time, *breaks[:-1], data.iloc[-1].time]
    else:
        breakpoints = [data.iloc[0].time, *breaks[1:-1], data.iloc[-1].time]

    continuum_fluence = np.sum([calculate_fluence(broken_powerlaw, final_par, breakpoints[i], breakpoints[i+1], count_ratio) for i in range(len(breakpoints)-1)])

    return {'parameters': {
                'break_num': break_number,
                'slopes': slopes, 'slopes_err': slopes_err,
                'breaks': breaks, 'breaks_err': breaks_err,
                'normal': normal, 'normal_err': normal_err},
            'fluence': continuum_fluence,
            'fit_statistics': final_fit_statistics}

#################################################################################
### FIT FLARES
#################################################################################

def fitFlares(data, flares, continuum, count_ratio):

    if not flares:
        return False

    fitted_model = broken_powerlaw(data.time, continuum['parameters'])
    data_subtracted = data.copy()
    data_subtracted['flux'] = data.flux - fitted_model

    # Fit each flare.
    flare_fits, flare_errs = flare_fitter(data, data_subtracted, flares)

    fittedFlares = []
    for indices, par, err in zip(flares, flare_fits, flare_errs):
        times = [data.iloc[x].time for x in indices]
        fluence_rise = calculate_fluence(fred_flare, par, times[0], times[1], count_ratio)
        fluence_decay = calculate_fluence(fred_flare, par, times[1], times[2], count_ratio)
        fluence_total = fluence_rise + fluence_decay

        peak_flux = data.iloc[indices[1]].flux
        peak_flux_err = data.iloc[indices[1]].flux_perr

        fittedFlares.append({'times': times, 'indices': indices, 'par': par, 'par_err': err, 'fluence': [fluence_total, fluence_rise, fluence_decay], 'peak_flux': [peak_flux, peak_flux_err]})

    return fittedFlares

#################################################################################
### FIT GRB LIGHTCURVE
#################################################################################

def fitGRB(data, count_ratio=1):
    logger.debug(f"Starting fitGRB")
    check_data_input(data)

    flare_indices = findFlares(data) # Find flare deviations.
    continuum = fitContinuum(data, flare_indices, count_ratio) # Fit continuum.
    flares = fitFlares(data, flare_indices, continuum, count_ratio) # Fit flares.

    logger.info(f"LAFF run finished.")
    return {'flares': flares, 'continuum': continuum}

#################################################################################
### PLOTTING
#################################################################################

def plotGRB(data, fitted_grb):
    logger.info(f"Starting plotGRB.")

    # Plot lightcurve.
    logger.debug("Plotting lightcurve.")
    plt.errorbar(data.time, data.flux,
                xerr=[-data.time_nerr, data.time_perr], \
                yerr=[-data.flux_nerr, data.flux_perr], \
                marker='', linestyle='None', capsize=0, zorder=1)

    # For smooth plotting of fitted functions.
    max, min = np.log10(data['time'].iloc[0]), np.log10(data['time'].iloc[-1])
    constant_range = np.logspace(min, max, num=5000)

    # Plot continuum model.
    logger.debug('Plotting continuum model.')
    fittedContinuum = broken_powerlaw(constant_range, fitted_grb['continuum']['parameters'])
    total_model = fittedContinuum
    plt.plot(constant_range, fittedContinuum, color='c')

    # Overlay marked flares.
    if fitted_grb['flares'] is not False:
        logger.debug("Plotting flare indices and models.")

        for flare in fitted_grb['flares']:
            
            # Plot flare data.
            flare_data = data.iloc[flare['indices'][0]:flare['indices'][2]]
            plt.errorbar(flare_data.time, flare_data.flux,
                        xerr=[-flare_data.time_nerr, flare_data.time_perr], \
                        yerr=[-flare_data.flux_nerr, flare_data.flux_perr], \
                        marker='', linestyle='None', capsize=0, color='r', zorder=2)

            # Plot flare models.
            flare_model = fred_flare(constant_range, flare['par'])
            total_model += flare_model
            plt.plot(constant_range, fred_flare(constant_range, flare['par']), color='tab:green', linewidth=0.6, zorder=3)

    # Plot total model.
    logger.debug('Plotting total model.')
    plt.plot(constant_range, total_model, color='tab:orange', zorder=5)
    upper_flux, lower_flux = data['flux'].max() * 10, data['flux'].min() * 0.1
    plt.ylim(lower_flux, upper_flux)

    # Plot powerlaw breaks.
    logger.debug('Plotting powerlaw breaks.')
    for x_pos in fitted_grb['continuum']['parameters']['breaks']:
        plt.axvline(x=x_pos, color='grey', linestyle='--', linewidth=0.5, zorder=0)

    logger.info("Plotting functions done, displaying...")
    plt.loglog()
    plt.show()

    return
