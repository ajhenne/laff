import emcee
import numpy as np

def power_break_1(x, index1, index2, break1, normal):

    mask = x < break1

    y = np.empty_like(x)

    y[mask] = normal * (x[mask] ** (-index1))
    y[~mask] = normal * (break1 ** (index2 - index1)) * (x[~mask] ** (-index2))

    return y

def log_likelihood(params, x, y, x_err, y_err):

    slope1, slope2, breakpoint, normal = params

    model = power_break_1(x, slope1, slope2, breakpoint, normal)
    
    log_likelihood = -0.5 * np.sum(((y - model) / y_err) ** 2) - np.sum(np.log(y_err))

    return log_likelihood

def log_prior(params):
    slope1, slope2, breakpoint, normal = params

    if not (-3 < slope1 < 3):
        return -np.inf
    if not (-3 < slope2 < 3):
        return -np.inf

    if (breakpoint < 0):
        return -np.inf

    return 0.0

def log_posterior(params, x, y, x_err, y_err):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, x, y, x_err, y_err)