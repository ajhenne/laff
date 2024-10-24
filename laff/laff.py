import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import warnings
import math
from intersect import intersection

# Ignore warnings.
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .flarefinding import flare_finding
from .modelling import find_intial_fit, fit_continuum_mcmc, flare_fitter, broken_powerlaw, fred_flare, gaussian_flare, improved_end_time
from .utility import check_data_input, calculate_fit_statistics, calculate_fluence, get_xlims

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
    """Set the desired logging level of the script.

    Args:
        level (string): from least to most verbose - 'none', 'debug', 'info', 'warning', 'error', 'critical'. The default level is normal.

    Raises:
        ValueError: Invalid logging level.
    """

    if level.lower() in ['debug', 'info', 'warning', 'error', 'critical']:
        logging_level = level.upper()
        logger.setLevel(logging_level)
    elif level.lower() in ['verbose']:
        logger.setLevel('DEBUG')
    elif level.lower() in ['none', 'quiet']:
        logger.setLevel(60) # set to above all other levels
    else:
        raise ValueError("Invalid logging level. Please use 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' or 'NONE'.")

#################################################################################
### FIND FLARES
#################################################################################

def findFlares(data, algorithm='sequential'):
    """Identify flares within datasets."""

    logger.debug(f"Starting findFlares - method {algorithm}")
    data = check_data_input(data) # First check input format is good.

    # Run flare finding.
    flares = flare_finding(data, algorithm)
    print(flares)
    logger.info(f"Found {len(flares)} flare(s).")

    return flares if len(flares) else False

#################################################################################
### CONTINUUM FITTING
#################################################################################

def fitContinuum(data: pd.DataFrame, flare_indices: list, count_ratio: float = 1, rich_output: bool = False, break_num: int = False) -> dict:
    logger.debug(f"Starting fitContinuum")

    # Remove flare data.
    if flare_indices:
        for start, _, end in flare_indices:
            data = data.drop(index=range(start+1, end))

    # Use ODR & AIC to find best number of powerlaw breaks.
    initial_fit, initial_fit_err, initial_fit_stats = find_intial_fit(data, rich_output, break_num)
    break_number = int((len(initial_fit-2)/2)-1)

    # Try an MCMC fitting run.
    try:
        raise ValueError
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

def fitFlares(data, flares, continuum, count_ratio, flare_model='fred', use_odr=False):

    if not flares:
        return False
    
    if flare_model == 'fred':
        model_flare = fred_flare
    elif flare_model == 'gauss':
        model_flare = gaussian_flare

    # Fit each flare.
    flare_fits, flare_errs = flare_fitter(data, continuum, flares, model=flare_model, use_odr=use_odr)

    # Format into dictionary nicely.
    fittedFlares = []
    for indices, par, err in zip(flares, flare_fits, flare_errs):
        # First run newly calculated end times.
        # indices[2], new_end_time = improved_end_time(data, indices, par, continuum['parameters'])
        times = [data.iloc[x].time for x in indices]
        # times[2] = new_end_time
        fluence_rise = calculate_fluence(model_flare, par, times[0], times[1], count_ratio)
        fluence_decay = calculate_fluence(model_flare, par, times[1], times[2], count_ratio)
        fluence_total = fluence_rise + fluence_decay

        peak_flux = data.iloc[indices[1]].flux
        peak_flux_err = data.iloc[indices[1]].flux_perr

        fittedFlares.append({'times': times, 'indices': indices, 'flare_model': flare_model, 'par': par, 'par_err': err, 'fluence': [fluence_total, fluence_rise, fluence_decay], 'peak_flux': [peak_flux, peak_flux_err]})

    return fittedFlares

#################################################################################
### FIT GRB LIGHTCURVE
#################################################################################

def fitGRB(data: pd.DataFrame, flare_model: str = 'fred', count_ratio: int = 1, rich_output: bool = False, use_odr: bool = False, break_num=False):
    # flare_model - use a certain flare model
    # use_odr - force use odr, disregard mcmc fitting
    # force_breaks - force a certain break_num

    # remove rich_output
    ## TODO ADD DESC HERE
    logger.debug(f"Starting fitGRB")
    data = check_data_input(data)

    flare_indices = findFlares(data) # Find flare deviations.
    continuum = fitContinuum(data, flare_indices, count_ratio, rich_output, break_num) # Fit continuum.
    flares = fitFlares(data, flare_indices, continuum, count_ratio, flare_model, use_odr=use_odr) # Fit flares.

    logger.info(f"LAFF run finished.")
    return {'flares': flares, 'continuum': continuum}

#################################################################################
### PLOTTING
#################################################################################

def plotGRB(data, fitted_grb, show=True, save_path=None):
    logger.info(f"Starting plotGRB.")

    plt.loglog()
    plt.xlabel("Time (s)")
    plt.ylabel("Flux (units)")

    # Plot lightcurve.
    logger.debug("Plotting lightcurve.")

    plt.errorbar(data.time, data.flux,
                xerr=[-data.time_nerr, data.time_perr], \
                yerr=[-data.flux_nerr, data.flux_perr], \
                marker='', linestyle='None', capsize=0, zorder=1)
    
    # Adjustments for xlims, ylims on a log graph.
    upper_flux, lower_flux = data['flux'].max() * 10, data['flux'].min() * 0.1
    plt.ylim(lower_flux, upper_flux)
    plt.xlim(get_xlims(data))

    ## Go over this logic -- why?
    if fitted_grb == None:
        # Guard clause to just plot the light curve.
        if show:
            plt.show()
            return
        else:
            return

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
            if flare['flare_model'] == 'fred': plotting_model = fred_flare
            if flare['flare_model'] == 'gauss': plotting_model = gaussian_flare

            flare_model = plotting_model(constant_range, flare['par'])
            total_model += flare_model
            plt.plot(constant_range, plotting_model(constant_range, flare['par']), color='tab:green', linewidth=0.6, zorder=3)

    # Plot total model.
    logger.debug('Plotting total model.')
    plt.plot(constant_range, total_model, color='tab:orange', zorder=5)

    # Plot powerlaw breaks.
    logger.debug('Plotting powerlaw breaks.')
    for x_pos in fitted_grb['continuum']['parameters']['breaks']:
        plt.axvline(x=x_pos, color='grey', linestyle='--', linewidth=0.5, zorder=0)

    if save_path:
        plt.savefig(save_path)
    logger.info("Plotting functions done, displaying...")
    if show == True:
        # what's the point ... just don't call the function?
        # unless i can return the plot object somehow?
        plt.show()

        
    return
