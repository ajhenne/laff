import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import warnings
import emcee
import corner


warnings.filterwarnings("ignore", category=RuntimeWarning)

### Logging ###
logger = logging.getLogger('laff')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
###

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
    logger.debug("Starting findFlares")

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

    logger.info(f"Found {len(flare_start)} flares.")
    return [flare_start, flare_peak, flare_end] if len(flare_start) else False

def fitContinuum(data, flare_indices):
    logger.debug(f"Starting fitContinuum")

    from .modelling import find_intial_fit, fit_continuum_mcmc

    # Remove flare data.
    if flare_indices:
        for start, end in zip(reversed(flare_indices[0]), reversed(flare_indices[2])):
            data = data.drop(index=range(start, end))

    # Use ODR & AIC to find best number of powerlaw breaks.
    initial_fit, initial_fit_err = find_intial_fit(data)
    break_number = int((len(initial_fit-2)/2)-1)

    # Run MCMC to refine fit.
    final_par, final_err = fit_continuum_mcmc(data, break_number, initial_fit, initial_fit_err)

    from .utility import calculate_fit_statistics
    final_fit_statistics = calculate_fit_statistics(data, broken_powerlaw, final_par)

    return {'parameters': final_par, 'errors': final_err, **final_fit_statistics}

def plotGRB(data, flares=False, continuum=False):
    logger.debug(f"Starting plotGRB")
    logger.debug(f"Input flares: {flares}")
    logger.debug(f"Input continuum: {continuum}")

    data_continuum = data.copy()
    flare_data = []

    if flares:
        for start, peak, end in zip(*flares):
            flare_data.append(data.iloc[start:end+1])
            data_continuum = data_continuum.drop(data.index[start:end+1])
            plt.axvspan(data.iloc[start].time, data.iloc[end].time, color='r', alpha=0.25)
            # plt.axvline(data.iloc[peak].time, color='g')
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
        modelpar, modelerr = continuum['parameters'], continuum['errors']

        nparam = len(modelpar)
        n = int((nparam-2)/2)

        slopes = modelpar[:n+1]
        slopes_Err = modelerr[:n+1]
        slopes_info = [f"{slp:.2f} ({slp_err:.2f})" for slp, slp_err in zip(slopes, slopes_Err)]

        breaks = modelpar[n+1:-1]
        breaks_Err = modelerr[n+1:-1]
        breaks_info = [f"{brk:.3g} ({brk_err:.3g})" for brk, brk_err in zip(breaks, breaks_Err)]
        
        normal = modelpar[-1]
        normal_Err = modelerr[-1]
        normal_info = f"{normal:.2e} ({normal_Err:.2e})"

        logger.info("Slopes: {}".format(', '.join(slopes_info)))
        logger.info("Breaks: {}".format(', '.join(breaks_info)))
        logger.info(f"Normal: {normal_info}")

        max, min = np.log10(data['time'].iloc[0]), np.log10(data['time'].iloc[-1])
        constant_range = np.logspace(min, max, num=5000)
        fittedModel = broken_powerlaw(constant_range, modelpar)


        plt.plot(constant_range, fittedModel)

        for x_pos in breaks:
            plt.axvline(x=x_pos, color='grey', linestyle='--', linewidth=0.75)


    plt.loglog()
    plt.show()


    if flares and continuum:

        #########################
        #### Temporary block ####

        flr_strt, flr_end = flares[0][0], flares[-1][2]

        plt.scatter(data.time, data.flux - broken_powerlaw(data.time, modelpar))   
        plt.axhline(y=0, color='grey', linewidth=1, linestyle='--')
        plt.xlim(data.iloc[flr_strt].time * 0.9, data.iloc[flr_end].time * 1.1)
        plt.semilogx()
        # plt.show()

        # Take data - model
        # For each flare, I can estimate rthe starting parameters.
        # Centre = peak of flares.
        # Rise/decay indices as a simple gradient calc?


        #### Temporary block ####
        #########################

    return