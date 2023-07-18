import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner

from .flarefinding import (
    _find_deviations,
    _find_deviations2,
    _find_minima,
    _find_maxima,
    _find_end,
    _remove_Duplicates,
    _check_AverageNoise,
    _check_FluxIncrease,
    _check_PulseShape )

from .modelling import (
 power_break_1,
 log_likelihood,
 log_posterior,
 log_prior   
)

def findFlares(data):
    """
    Find flares within a GRB lightcurve.

    Longer description.
    
    [Parameters]
        data
            A pandas table containing the light curve data. Columns named [time,
            time_perr, time_nerr, flux, flux_perr, flux_nerr].
            
    [Returns]
        flares
            A nested list of flare start, stop, end indices.
    """

    # Check data is correct input format.
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

    # Cutoff late data.
    LATE_CUTOFF = True
    data = data[data.time < 2000] if LATE_CUTOFF else data

    # Find deviations, or possible flares.
    deviations = _find_deviations2(data)

    # Refine deviations by looking for local minima, or flare starts.
    starts = _find_minima(data, deviations)

    # For each flare start, find the corresponding peak.
    peaks = _find_maxima(data, starts)

    # Combine any duplicate start/peaks.
    starts, peaks = _remove_Duplicates(data, starts, peaks)

    # For each flare peak, find the corresponding flare end.
    DECAYPAR = 3
    ends = _find_end(data, starts, peaks, DECAYPAR)

    # Perform some checks to ensure the found flares are valid.
    flare_start, flare_peak, flare_end = [], [], []
    for start, peak, end in zip(starts, peaks, ends):
        check1 = _check_AverageNoise(data, start, peak, end)
        check2 = _check_FluxIncrease(data, start, peak)
        check3 = _check_PulseShape(data, start, peak, end)
        if check1 and check2 and check3:
            flare_start.append(int(start))
            flare_peak.append(int(peak))
            flare_end.append(int(end))
    return [flare_start, flare_peak, flare_end]

def fitContinuum(data, flare_indices):

    print(flare_indices)
    # Remove flare data.
    # flare_indices = sorted(flare_indices, key=lambda x: x[1], reverse=True)
    # for start, end in zip(flare_indices[0], flare_indices[2]):
    #     print(start, end)
    #     del data[start:end]
    
    print('here1')
    ndim = 4
    nwalkers = 25
    nsteps = 500
    
    # How to calculate these more effectively. Slopes just keep as constant?
    slope1_guess = 1.0
    slope2_guess = 2.0

    SLOPE_RANGE = 3
    # Break point use a spacing algorithm?
    break_point_guess = 100

    BREAKPOINT_RANGE = 200

    normal_guess = 1e-7


    p0 = np.zeros((nwalkers, ndim))
    p0[:, 0] = slope1_guess + SLOPE_RANGE * np.random.randn(nwalkers)
    p0[:, 1] = slope2_guess + SLOPE_RANGE * np.random.randn(nwalkers)
    p0[:, 2] = break_point_guess + BREAKPOINT_RANGE * np.random.randn(nwalkers)
    p0[:, 3] = normal_guess + (normal_guess*10) * np.random.randn(nwalkers)


    print('here2')

    x = data.time
    y = data.flux
    x_err = data.time_perr
    y_err = data.flux_perr

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y, x_err, y_err))
    sampler.run_mcmc(p0, nsteps)
    print('here3')

    burnin = 100

    samples = sampler.chain[:, burnin:, :].reshape(-1, ndim)

    slope1_est, slope2_est, break_point_est, normal_est = map(lambda v: np.median(v), samples.T)
    slope1_err, slope2_err, break_point_err, normal_err = map(lambda v: np.std(v), samples.T)

    print(slope1_est, slope1_err)
    print(slope2_est, slope2_err)
    print(break_point_est, break_point_err)
    print(normal_est, normal_err)

    fig = corner.corner(samples, labels=["slope1", "slope2", "break_point", "normal"])
    plt.show()

    fitted_model_y = power_break_1(data.time, slope1_est, slope2_est, break_point_est, normal_est)

    plt.scatter(data.time, data.flux)
    plt.plot(data.time, fitted_model_y)
    plt.loglog()

    plt.show()
    return

def plotResults(data, flares):

    flare_data = []
    cont_data = data.copy()

    for start, peak, end in zip(*flares):
        flare_data.append(data.iloc[start:end+1])   
        cont_data = cont_data.drop(cont_data.index[start:end+1])
    
    try:
        flare_data = pd.concat(flare_data) # concat rather than each flare in a separate list element??
    except ValueError:
        pass

    plt.errorbar(cont_data.time, cont_data.flux,
    xerr=[-cont_data.time_nerr, cont_data.time_perr], \
    yerr=[-cont_data.flux_nerr, cont_data.flux_perr], \
    marker='', linestyle='None', capsize=0)

    try:
        for start, peak, end in zip(*flares):
            plt.axvspan(data.iloc[start].time, data.iloc[end].time, color='r', alpha=0.25)
            plt.axvline(data.iloc[peak].time, color='g')
    except AttributeError:
        pass

    try:
        plt.errorbar(flare_data.time, flare_data.flux,
        xerr=[-flare_data.time_nerr, flare_data.time_perr], \
        yerr=[-flare_data.flux_nerr, flare_data.flux_perr], \
        marker='', linestyle='None', capsize=0, color='r')
    except AttributeError:
        pass

    plt.loglog()
    plt.show()

    return