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
from .modelling import broken_powerlaw, find_afterglow_fit, calculate_afterglow_fluence
from .modelling import flare_fitter, fred_flare, gaussian_flare, improved_end_time
from .utility import check_data_input, calculate_fit_statistics, calculate_fluence, get_xlims

# findFlares() -- locate the indices of flares in the lightcurve
# fitContinuum(flare_indices) -- use the indices to exclude data, then fit the continuum
# fitFlares(flares, continuum) -- use indices + continuum to fit the flares

# fitGRB() -- runs all 3 function in sequence, then does some final cleanups
#          -- final statistics of the whole fit
#          -- this is what the user should be running
#          -- outputs a dictionary with all useful statistics

################################################################################
### LOGGER
################################################################################

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
        level (string): from least to most verbose - 'none', 'debug', 'info',
        'warning', 'error', 'critical'. The default level is normal.

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

################################################################################
### FIND FLARES
################################################################################

def findFlares(data, algorithm='sequential'):
    """Identify flares within datasets."""

    logger.debug(f"Starting findFlares - method {algorithm}")
    if check_data_input(data) == False:
        return  # First check input format is good.

    # Run flare finding.
    flares = flare_finding(data, algorithm)

    return flares if len(flares) else False

################################################################################
### CONTINUUM FITTING
################################################################################

def fitAfterglow(data: pd.DataFrame, flare_indices: list[list[int]] = None, *, errors_to_std: float = 1.0, count_flux_ratio: float = 1.0) -> dict:
    """Fits the afterglow of the light curve with a series of broken power laws.

    Args:
        data (pd.DataFrame):
            The dataset stored as a pandas dataframe.
        flare_indices (list[list[int]]):
            A nested list of 3 integers, the start/peak/end of flares as
            returned by laff.findFlares().
        errors_to_std (float, optional):
            The conversion factor to be applied to the x and y errors on data,
            the ODR fitter assumes there are 1-sigma standard deviations.
            Defaults to 1.0.
        count_flux_ratio (float, optional):
            The conversion factor to be applied to scale into flux, if the data
            provided is in count rate. Defaults to 1.0.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        dict: _description_
    """

    logger.debug('fitAfterglow()')

    # Remove flare data.
    if flare_indices:
        logger.debug('Removing indices of %s flares', len(flare_indices))
        for start, _, end in flare_indices:
            data = data.drop(index=range(start+1, end))

    afterglow_par, afterglow_err, afterglow_stats, breaknum = find_afterglow_fit(data, errors_to_std)

    slopes     = list(afterglow_par[:breaknum+1])
    slopes_err = list(afterglow_err[:breaknum+1])
    breaks     = list(afterglow_par[breaknum+1:-1])
    breaks_err = list(afterglow_err[breaknum+1:-1])
    normal     = afterglow_par[-1]
    normal_err = afterglow_err[-1]

    # Calculate fluence.
    afterglow_fluence = calculate_afterglow_fluence(data, breaknum, breaks, afterglow_par, count_flux_ratio)

    return {'parameters': {
                'break_num': breaknum,
                'slopes': slopes, 'slopes_err': slopes_err,
                'breaks': breaks, 'breaks_err': breaks_err,
                'normal': normal, 'normal_err': normal_err},
            'fluence': afterglow_fluence,
            'fit_statistics': afterglow_stats}

################################################################################
### FIT FLARES
################################################################################

def fitFlares(data, flares, continuum, count_ratio, flare_model='fred', skip_mcmc=False):

    if not flares:
        return False
    
    if flare_model == 'fred':
        model_flare = fred_flare
    elif flare_model == 'gauss':
        model_flare = gaussian_flare

    # Fit each flare.
    flare_fits, flare_errs = flare_fitter(data, continuum, flares, model=flare_model, skip_mcmc=skip_mcmc)

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

################################################################################
### FIT GRB LIGHTCURVE
################################################################################

def fitGRB(data: pd.DataFrame, *,
           flare_algorithm: str = 'sequential', flare_model: str = 'fred',
           errors_to_std: float = 1.0, count_flux_ratio: float = 1.0,
            rich_output: bool = False, skip_mcmc: bool = False, break_num=False):
    # flare_model - use a certain flare model
    # use_odr - force use odr, disregard mcmc fitting
    # force_breaks - force a certain break_num

    # remove rich_output
    ## TODO ADD DESC HERE
    logger.debug(f"Starting fitGRB")
    if check_data_input(data) == False:
        raise ValueError("check data failed")

    flare_indices = findFlares(data, algorithm=flare_algorithm) # Find flare deviations.
    afterglow = fitAfterglow(data, flare_indices, errors_to_std=errors_to_std, count_flux_ratio=count_flux_ratio) # Fit continuum.
    flares = fitFlares(data, flare_indices, afterglow, count_flux_ratio, flare_model, skip_mcmc=skip_mcmc) # Fit flares.

    logger.info(f"LAFF run finished.")
    return afterglow, flares

################################################################################
### PLOTTING
################################################################################

def plotGRB(data, afterglow, flares, show=True, save_path=None):
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

    # For smooth plotting of fitted functions.
    max, min = np.log10(data['time'].iloc[0]), np.log10(data['time'].iloc[-1])
    constant_range = np.logspace(min, max, num=5000)

    # Plot continuum model.
    logger.debug('Plotting continuum model.')
    fittedContinuum = broken_powerlaw(afterglow['parameters'], constant_range)
    total_model = fittedContinuum
    plt.plot(constant_range, fittedContinuum, color='c')

    # Overlay marked flares.
    if flares is not False:
        logger.debug("Plotting flare indices and models.")

        for flare in flares:
            
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
    for x_pos in afterglow['parameters']['breaks']:
        plt.axvline(x=x_pos, color='grey', linestyle='--', linewidth=0.5, zorder=0)

    if save_path:
        plt.savefig(save_path)
    logger.info("Plotting functions done, displaying...")
    if show == True:
        # what's the point ... just don't call the function?
        # unless i can return the plot object somehow?
        plt.show()

        
    return
