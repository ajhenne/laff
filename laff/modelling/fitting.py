import emcee
import numpy as np

def broken_powerlaw(x, params):

    x = np.array(x)

    nparam = len(params)
    n = int((nparam-2)/2)

    slopes = params[:n+1]
    breaks = params[n+1:-1]
    normal = params[-1]

    mask = []

    for i in range(n):
        try:
            mask.append(x > breaks[i])
        except:
            print(i, 'too many?')
            pass

    if n >= 0:
        model = normal * (x**(-slopes[0]))
    if n >= 1:
        mask_0 = np.array(mask[0])
        model[np.where(mask_0)] = normal * (x[np.where(mask_0)]**(-slopes[1])) * (breaks[0]**(-slopes[0]+slopes[1]))
    if n >= 2:
        mask[1] = np.array(mask[1])
        model[np.where(mask[1])] = normal * (x[np.where(mask[1])]**(-slopes[2])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2]))
    if n >= 3:
        model[np.where(mask[2])] = normal * (x[np.where(mask[2])]**(-slopes[3])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3]))
    if n >= 4:
        model[np.where(mask[3])] = normal * (x[np.where(mask[3])]**(-slopes[4])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3])) * (breaks[3]**(-slopes[3]+slopes[4]))
    if n >= 5:
        model[np.where(mask[4])] = normal * (x[np.where(mask[4])]**(-slopes[5])) * (breaks[0]**(-slopes[0]+slopes[1])) * (breaks[1]**(-slopes[1]+slopes[2])) * (breaks[2]**(-slopes[2]+slopes[3])) * (breaks[3]**(-slopes[3]+slopes[4])) * (breaks[4]**(-slopes[4]+slopes[5]))

    return model

def power_break_1(x, index1, index2, break1, normal):
    mask = x < break1

    y = np.empty_like(x)
    y[mask] = normal * (x[mask] ** (-index1))
    y[~mask] = normal * (break1 ** (index2 - index1)) * (x[~mask] ** (-index2))
    return y

def log_likelihood(params, x, y, x_err, y_err):
    model = broken_powerlaw(x, params)
    
    residual = y - model
    chi_squared = np.sum((residual / y_err) ** 2)
    log_likelihood = -0.5 * (len(x) * np.log(2*np.pi) + np.sum(np.log(y_err ** 2)) + chi_squared)

    return log_likelihood

def log_prior(params):

    nparam = len(params)
    n = int((nparam-2)/2)

    slopes = params[:n+1]
    breaks = params[n+1:-1]
    normal = params[-1]

    if not all(-1 < value < 2.5 for value in slopes):
        return -np.inf

    if any(value < 0 for value in breaks):
        return -np.inf
    #  any breakpoint > data.time max ?

    if (normal < 0):
        return -np.inf

    return 0.0

def log_posterior(params, x, y, x_err, y_err):
    lp = log_prior(params)
    ll = log_likelihood(params, x, y, x_err, y_err)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ll