import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.optimize import approx_fprime
import math
import logging

logger = logging.getLogger('laff')

PAR_NAMES_FLARE = ['t_start', 'rise', 'decay', 'amplitude']
PAR_NAMES_CONTINUUM = ['break_num', 'slopes', 'slopes_err', 'breaks', 'breaks_err', 'normal', 'normal_err']
STAT_NAMES_CONTINUUM = ['chisq', 'rchisq', 'n', 'npar', 'dof', 'deltaAIC']

def calculate_fit_statistics(data, model, params, y_col='flux'):
    
    # if temp_flare_shell:
        # fitted_model = model(np.array(data.time), params)
    # else:
        # fitted_model = model(params, np.array(data.time))

    chisq = np.sum(((data[y_col] - model(params, data['time'])) / data['flux_perr']) ** 2)  
    n = len(data['time'])
    npar = len(params)
    
    dof = n - npar
    r_chisq = chisq / dof

    deltaAIC = (2 * npar) + (n * np.log(r_chisq))
    deltaAIC = deltaAIC if deltaAIC != -np.inf else np.inf # negative infinity check

    residuals = data[y_col] - model(params, data['time'])
    variance = data['flux_perr'] ** 2
    lnL = -0.5 * np.sum((residuals**2) / variance + np.log(2 * np.pi * variance))

    BIC = (np.log(n) * npar) - 2 * lnL
        
    # return {'n': n, 'npar': npar, 'dof': dof, 'deltaAIC': deltaAIC}
    return {'chisq': chisq, 'rchisq': r_chisq, 'n': n, 'npar': npar, 'dof': dof, 'deltaAIC': deltaAIC, 'BIC': BIC}

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
    fitted_model = model(params, range)
    fluence = integrate.trapezoid(fitted_model, x=range)
    logger.debug('Fluence from %s to %s is %s', start, stop, fluence)

    return fluence * count_ratio

def calculate_par_err(params, chi_wrapper):

    epsilon = [max(x * 1e-4, 1e-8) for x in params]
    n = len(params)
    hessian = np.zeros((n, n))

    for i in range(n):
        # vary
        x1 = np.array(params, copy=True)
        x1[i] = params[i] + epsilon[i]
        grad1 = approx_fprime(x1, chi_wrapper, epsilon)
        x1[i] = params[i] - epsilon[i]
        grad2 = approx_fprime(x1, chi_wrapper, epsilon)
        hessian[:, i] = (grad1 - grad2) / (2 * epsilon[i])

    try:
        covariance_matrix = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance_matrix = np.linalg.pinv(hessian) # Moore–Penrose pseudo-inverse
        logger.warning("\tusing pseudo inverse cov matrix - par likely poorly constrained")
    errors = np.sqrt(np.diag(covariance_matrix))

    return 3 * errors