import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import warnings

# Ignore warnings.
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .flarefinding import flare_finding
from .modelling import broken_powerlaw, find_afterglow_fit, calculate_afterglow_fluence
from .modelling import flare_fitter, package_flares, fred_flare
from .utility import check_data_input, calculate_fit_statistics, calculate_fluence

## recent note:
## the conditions in fit_afterglow work but the fitter gets 'stuck' at that
## value without changing input variables.
## either remove conditions and filter after or find a force shuffle variables.

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

def findFlares(data, algorithm='savgol', **kwargs):
    """Identify flares within datasets."""

    logger.debug(f"Starting findFlares - method {algorithm}")
    if check_data_input(data) == False:
        return  # First check input format is good.

    # Run flare finding.
    flares = flare_finding(data, algorithm, **kwargs)

    return flares if len(flares) else False

################################################################################
### CONTINUUM FITTING
################################################################################

def fitAfterglow(data: pd.DataFrame, flare_indices: list[list[int]] = None, *, errors_to_std: float = 1.0, count_ratio: float = 1.0) -> dict:
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

    # Separate flare data.
    data_flare = [[], []]
    if flare_indices:
        logger.debug('Removing indices of %s flares', len(flare_indices))

        for start, _, end in flare_indices:
            data_flare[0].extend(data['time'].iloc[start:end])
            data_flare[1].extend(data['flux'].iloc[start:end])
            
        for start, _, end in flare_indices:
            data = data.drop(index=range(start, end))

    afterglow_par, afterglow_err, afterglow_stats, breaknum = find_afterglow_fit(data, data_flare)

    slopes     = list(afterglow_par[:breaknum+1])
    slopes_err = list(afterglow_err[:breaknum+1])
    breaks     = list(afterglow_par[breaknum+1:-1])
    breaks_err = list(afterglow_err[breaknum+1:-1])
    normal     = afterglow_par[-1]
    normal_err = afterglow_err[-1]

    # Calculate fluence.
    afterglow_fluence = calculate_afterglow_fluence(data, breaknum, breaks, afterglow_par, count_ratio)

    return {'params': {
                'break_num': breaknum,
                'slopes': slopes, 'slopes_err': slopes_err,
                'breaks': breaks, 'breaks_err': breaks_err,
                'normal': normal, 'normal_err': normal_err},
            'fluence': afterglow_fluence,
            'fit_statistics': afterglow_stats}

################################################################################
### FIT FLARES
################################################################################

def fitFlares(data, flare_indices, afterglow, *, count_ratio=1.0, flare_model='fred', skip_mcmc=False):

    if not flare_indices:
        return False
    
    flare_model = fred_flare

    # Fit each flare.
    flare_fits, flare_stats, flare_errs, flare_indices = flare_fitter(data, afterglow, flare_indices, model=flare_model)

    # Format, calculate times, fluence.
    flaresDict = package_flares(data, flare_fits, flare_stats, flare_errs, flare_indices, count_ratio=count_ratio)

    return flaresDict

################################################################################
### FIT GRB LIGHTCURVE
################################################################################

def fitGRB(data: pd.DataFrame, *,
           flare_algorithm: str = 'savgol', flare_model: str = 'fred',
           errors_to_std: float = 1.0, count_ratio: float = 1.0):
    # flare_model - use a certain flare model
    # force_breaks - force a certain break_num
    # cont_ratio

    # remove rich_output
    ## TODO ADD DESC HERE
    logger.debug(f"Starting fitGRB")
    if check_data_input(data) == False:
        raise ValueError("check data failed")

    flare_indices = findFlares(data, algorithm=flare_algorithm) # Find flare deviations.
    afterglow = fitAfterglow(data, flare_indices, errors_to_std=errors_to_std, count_ratio=count_ratio) # Fit continuum.
    flares = fitFlares(data, flare_indices, afterglow, count_ratio=count_ratio, flare_model=flare_model) # Fit flares.

    logger.info(f"LAFF run finished.")
    return afterglow, flares

################################################################################
### PLOTTING
################################################################################

def printGRB(data, afterglow, flares):

    print(f"\033[1m// Afterglow - {afterglow['params']['break_num']} breaks\033[0m")

    print("Breaks |", ", ".join(f"{10**x:.0f} \033[2m(-{10**x - 10**(x-dx):.0f},+{10**(x+dx) - 10**x:.0f})\033[0m" for x, dx in zip(afterglow['params']['breaks'], afterglow['params']['breaks_err'])))

    print("Slopes |", ", ".join(f"{x:.2f} \033[2m(+/-{dx:.2f})\033[0m" for x, dx in zip(afterglow['params']['slopes'], afterglow['params']['slopes_err'])))

    print("Normal |", f"{afterglow['params']['normal']:.2f} \033[2m({afterglow['params']['normal_err']:.2f})\033[0m")

    print(afterglow)

    return 

def plotGRB(data, afterglow, flares, show=True, save_path=None, bat=False):
    logger.info(f"Starting plotGRB.")

    plt.rcParams.update({'font.size': 16})

    plt.xlabel("Time (s)")
    plt.ylabel("Flux (units)")

    # Plot lightcurve.
    logger.debug("Plotting lightcurve.")

    plt.errorbar(data.time, data.flux,
                xerr=[-data.time_nerr, data.time_perr], \
                yerr=[-data.flux_nerr, data.flux_perr], \
                marker='', linestyle='None', capsize=0, zorder=1)
    
    if bat == False:
        # Adjustments for xlims, ylims on a log graph.
        upper_flux, lower_flux = data['flux'].max() * 10, data['flux'].min() * 0.1
        plt.ylim(lower_flux, upper_flux)

        lower_time = 0.8 * (data['time'].iloc[0] + data['time_nerr'].iloc[0])
        upper_time = 1.2 * (data['time'].iloc[-1] + data['time_perr'].iloc[-1])
        plt.xlim(lower_time, upper_time)
        plt.loglog()

    # For smooth plotting of fitted functions.
    max, min = np.log10(data['time'].iloc[0] + data['time_nerr'].iloc[0]), np.log10(data['time'].iloc[-1] + data['time_perr'].iloc[-1])
    constant_range = np.logspace(min, max, num=5000)

    # Plot continuum model.
    logger.debug('Plotting continuum model.')
    fittedContinuum = broken_powerlaw(afterglow['params'], constant_range)
    total_model = fittedContinuum
    plt.plot(constant_range, fittedContinuum, color='c')

    # Overlay marked flares.
    if flares is not False:
        logger.debug("Plotting flare indices and models.")

        for flare in flares:
            
            # Plot flare data.
            flare_data = data.iloc[flare['indices'][0]:flare['indices'][2]+1]
            plt.errorbar(flare_data.time, flare_data.flux,
                        xerr=[-flare_data.time_nerr, flare_data.time_perr], \
                        yerr=[-flare_data.flux_nerr, flare_data.flux_perr], \
                        marker='', linestyle='None', capsize=0, color='r', zorder=2)

            flare_model = fred_flare(flare['params'], constant_range)
            total_model += flare_model
            plt.plot(constant_range, fred_flare(flare['params'], constant_range), color='tab:green', linewidth=0.6, zorder=3)

    # Plot total model.
    logger.debug('Plotting total model.')
    plt.plot(constant_range, total_model, color='tab:orange', zorder=5)

    # Plot powerlaw breaks.
    logger.debug('Plotting powerlaw breaks.')
    for x_pos in afterglow['params']['breaks']:
        plt.axvline(x=10**x_pos, color='grey', linestyle='--', linewidth=0.5, zorder=0)

    if save_path:
        plt.savefig(save_path)
    logger.info("Plotting functions done, displaying...")
    if show == True:
        # what's the point ... just don't call the function?
        # unless i can return the plot object somehow?
        plt.show()

        
    return
