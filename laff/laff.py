import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner

from .modelling import broken_powerlaw

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

    from .flarefinding import (
        _find_deviations,
        _find_minima,
        _find_maxima,
        _find_end,
        _remove_Duplicates,
        _check_AverageNoise,
        _check_FluxIncrease,
        _check_PulseShape )

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
    deviations = _find_deviations(data)

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
    return [flare_start, flare_peak, flare_end] if len(flare_start) else None

def fitContinuum(data, flare_indices):

    from .modelling import find_intial_fit, fitMCMC

    # Remove flare data.
    if flare_indices:
        for start, end in zip(reversed(flare_indices[0]), reversed(flare_indices[2])):
            data = data.drop(index=range(start, end))

    # Use AIC to find best number of powerlaw breaks.
    initial_fit, initial_fit_err = find_intial_fit(data)
    break_number = int((len(initial_fit-2)/2)-1)

    # Run MCMC to refine fit.
    final_par, final_err = fitMCMC(data, break_number, initial_fit, initial_fit_err)

    return final_par, final_err

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
    
        slopes_Err = modelerr[:n+1]
        breaks_Err = modelerr[n+1:-1]
        normal_Err = modelerr[-1]

        print('Slopes:', *["{:.2f}".format(number) for number in slopes])
        print('Slopes Err:', *["{:.2f}".format(number) for number in slopes_Err])
        print('Breaks:', *["{:.2f}".format(number) for number in breaks])
        print('Breaks Err:', *["{:.2f}".format(number) for number in breaks_Err])
        print('Normal:', "{:.2e}".format(normal))
        print('Normal Err:', "{:.2e}".format(normal_Err))

        max, min = np.log10(data['time'].iloc[0]), np.log10(data['time'].iloc[-1])
        constant_range = np.logspace(min, max, num=5000)
        fittedModel = broken_powerlaw(constant_range, modelpar)


        plt.plot(constant_range, fittedModel)

        for x_pos in breaks:
            plt.axvline(x=x_pos, color='grey', linestyle='--', linewidth=0.75)


    plt.loglog()
    plt.show()

    return