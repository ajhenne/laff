import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('laff')

PAR_NAMES_FLARE = ['t_start', 'rise', 'decay', 'amplitude']
PAR_NAMES_CONTINUUM = ['break_num', 'slopes', 'slopes_err', 'breaks', 'breaks_err', 'normal', 'normal_err']
STAT_NAMES_CONTINUUM = ['chisq', 'rchisq', 'n', 'npar', 'dof', 'deltaAIC']

def calculate_fit_statistics(data, model, params):
    
    fitted_model = model(np.array(data.time), params)
    chisq = np.sum(((data.flux - fitted_model) ** 2) / (data.flux_perr ** 2))
    
    n = len(data.time)
    npar = len(params)
    dof = n - npar
    r_chisq = chisq / dof

    deltaAIC = (2 * npar) + (n * np.log(r_chisq))
        
    return {'chisq': chisq, 'rchisq': r_chisq, 'n': n, 'npar': npar, 'dof': dof, 'deltaAIC': deltaAIC}

def check_data_input(data):

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Invalid input data type. Should be pandas dataframe.")

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

    logger.debug('Data input is good.')

    return

    