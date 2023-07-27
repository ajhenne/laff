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
 broken_powerlaw,
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

    # Remove flare data.
    for start, end in zip(reversed(flare_indices[0]), reversed(flare_indices[2])):
        data = data.drop(index=range(start, end))


    def fitPowerlaws(data, breaknum):
        
        ndim = 2 * breaknum + 2
        nwalkers = 40
        nsteps = 1000

        # Initialise guesses.
        guess_slopes = [1] * (breaknum+1)
        std_slopes = [0.6] * (breaknum+1)

        normal_guess = np.logspace(0, -9, base=10, num=nwalkers)

        guess_breaks = np.logspace(
                        np.log10(data['time'].iloc[0]) * 1.1, 
                        np.log10(data['time'].iloc[-1]) * 0.9, breaknum)

        std_breaks = [(x + x * np.random.randn(nwalkers)) for x in list(guess_breaks)]

        p0 = np.zeros((nwalkers, ndim))
        for i in range(0, breaknum+1): # First n+1 are slopes.
            p0[:, i] = guess_slopes[i] + std_slopes[i] * np.random.randn(nwalkers)
        for breaknum, i in enumerate(range(breaknum+1, ndim-1)): # n+2 to penultimate are breaks.
            p0[:, i] = std_breaks[breaknum]
            # p0[:, i] = np.exp(np.random.uniform(np.log(data['time'].iloc[0]), np.log(data['time'].iloc[-1]), nwalkers)) * 0.1
        p0[:, -1] = normal_guess # Final point is normal.

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, \
            args=(data.time, data.flux, data.time_perr, data.flux_perr))    
        sampler.run_mcmc(p0, nsteps)

        burnin = 200

        samples = sampler.chain[:, burnin:, :].reshape(-1, ndim)

        fitted_par = list(map(lambda v: np.median(v), samples.T))
        fitted_err = list(map(lambda v: np.std(v), samples.T))

        return [fitted_par, fitted_err]

    # for n in range(1, 6):
        # print(n, fitPowerlaws(data, n))

    number = 3

    fit = fitPowerlaws(data, number)

    return fit[0], fit[1]

        # append each powerlaw fit to a list

    # Determine the best model.
    # def functionDetermineBestModel():

    # Do a more precise fit.
    # def functionFitFinalContinuum():

    ####


    # fig = corner.corner(samples, labels=["slope1", "slope2", "break_point", "normal"])
    # plt.show()

    # Perform all fits.
    # Evaluate each one.

    # Run the best fit for a bit longer.

    # Return model and parameters.

    # return power_break_1, fitted_par, fitted_err

def plotGRB(data, flares=None, continuum=None):

    data_continuum = data.copy()
    flare_data = []

    if flares:
        for start, peak, end in zip(*flares):
            flare_data.append(data.iloc[start:end+1])
            data_continuum = data_continuum.drop(data.index[start:end+1])
            plt.axvspan(data.iloc[start].time, data.iloc[end].time, color='r', alpha=0.25)
            plt.axvline(data.iloc[peak].time, color='g')
        flare_data = pd.concat(flare_data)
        plt.errorbar(flare_data.time, flare_data.flux,
            xerr=[-flare_data.time_nerr, flare_data.time_perr], \
            yerr=[-flare_data.flux_nerr, flare_data.flux_perr], \
            marker='', linestyle='None', capsize=0, color='r')

    # Plot lightcurve.
    plt.errorbar(data_continuum.time, data_continuum.flux,
    xerr=[-data_continuum.time_nerr, data_continuum.time_perr], \
    yerr=[-data_continuum.flux_nerr, data_continuum.flux_perr], \
    marker='', linestyle='None', capsize=0)

    if continuum:
        modelpar, modelerr = continuum

        nparam = len(modelpar)
        n = int((nparam-2)/2)

        slopes = modelpar[:n+1]
        breaks = modelpar[n+1:-1]
        normal = modelpar[-1]

        print('Slopes:', *slopes)
        print('Breaks:', *breaks)
        print('Normal:', normal)

        max, min = np.log10(data['time'].iloc[0]), np.log10(data['time'].iloc[-1])
        constant_range = np.logspace(min, max, num=2000)
        fittedModel = broken_powerlaw(constant_range, modelpar)


        plt.plot(constant_range, fittedModel)

        for x_pos in breaks:
            plt.axvline(x=x_pos, color='grey', linestyle='--', linewidth=0.75)


    plt.loglog()
    plt.show()

    return