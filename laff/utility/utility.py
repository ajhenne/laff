import numpy as np
import pandas as pd
import scipy.integrate as integrate
import math
import logging

logger = logging.getLogger('laff')

PAR_NAMES_FLARE = ['t_start', 'rise', 'decay', 'amplitude']
PAR_NAMES_CONTINUUM = ['break_num', 'slopes', 'slopes_err', 'breaks', 'breaks_err', 'normal', 'normal_err']
STAT_NAMES_CONTINUUM = ['chisq', 'rchisq', 'n', 'npar', 'dof', 'deltaAIC']

def calculate_fit_statistics(data, model, params, temp_flare_shell=False):
    
    if temp_flare_shell:
        fitted_model = model(np.array(data.time), params)
    else:
        fitted_model = model(params, np.array(data.time))

    chisq = np.sum(((data.flux - fitted_model) ** 2) / (data.flux_perr ** 2))
    
    n = len(data.time)
    npar = len(params)
    dof = n - npar
    r_chisq = chisq / dof

    deltaAIC = (2 * npar) + (n * np.log(r_chisq))
    deltaAIC = deltaAIC if deltaAIC != -np.inf else np.inf # negative infinity check
        
    return {'chisq': chisq, 'rchisq': r_chisq, 'n': n, 'npar': npar, 'dof': dof, 'deltaAIC': deltaAIC}

def check_data_input(data):

    # if not isinstance(data, pd.DataFrame):
        # raise TypeError(f"Invalid input data type. Should be pandas dataframe.")

    if len(data) < 3:
        logger.critical("Too short.")
        raise ValueError(f"Data too short, require at least 3 data points - given data is {len(data)}")

    if not data.shape[1] == 4 and not data.shape[1] == 6:
        raise ValueError("Expected dataframe of shape (X, 4) or (X, 6)")

    # Check column names.
    expected_columns = ['time', 'time_perr', 'time_nerr', 'flux', 'flux_perr', 'flux_nerr']
    if data.shape[1] == 4:
        data.columns = ['time', 'time_perr', 'flux', 'flux_perr']
        data['time_nerr'] = data['time_perr']
        data['flux_nerr'] = data['flux_perr']
        data.columns = expected_columns
    elif data.shape[1] == 6:
        data.columns = expected_columns
    else:
        raise ValueError(f"Expected dataframe with 4 or 6 columns - got {data.shape[1]}.")
    
    data = data.reset_index(drop=True)


    logger.debug('Data input is good.')

    return True

def calculate_fluence(model, params, start, stop, count_ratio):
    """Given some model and range, calculate the fluence."""

    range = np.logspace(np.log10(start), np.log10(stop), num=2500)
    fitted_model = model(range, params)
    fluence = integrate.trapezoid(fitted_model, x=range)
    logger.debug('Fluence from %s to %s is %s', start, stop, fluence)

    return fluence * count_ratio

def get_xlims(data):
    """Calculating xlimits for a log graph.

    Calculate the xlimits for a log graph so there's no large spaces on either
    end. I'm probably missing an easy bit of math. If the value is too close,
    i.e. log10(time) = 2.05, then substracting 10**2 is too much, so included an
    edge case.

    Args:
        data (pd.Dataframe): GRB lightcurve dataframe

    Returns:
        float: lower x-axis limit
        float: upper x-axis limit
        _type_: _description_
    """
    lowest_time_val = data['time'].iloc[0] + data['time_nerr'].iloc[0]
    lowest_time_val_power = math.floor(np.log10(lowest_time_val))
    lowest_time_val_power = lowest_time_val_power-1 if lowest_time_val_power - math.floor(lowest_time_val) < 0.2 else lowest_time_val_power
    lower_lim = lowest_time_val - 10**lowest_time_val_power

    highest_time_val = (data['time'].max() + data['time_perr'].iloc[len(data.time)-1])
    upper_lim = highest_time_val + 10**(math.floor(np.log10(highest_time_val)))

    return lower_lim, upper_lim
