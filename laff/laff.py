import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import warnings

import emcee
# do I need emcee in this main module?

# Ignore warnings.
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .utility import check_data_input

#################################################################################
### LOGGER
#################################################################################

logger = logging.getLogger('laff')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

#################################################################################
### FIND FLARES
#################################################################################

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

    # Check data is correct input format.
    check_data_input(data)

    # Cutoff late data.
    LATE_CUTOFF = True
    data = data[data.time < 2000] if LATE_CUTOFF else data

    from .flarefinding import possible_flares, _check_AverageNoise, _check_FluxIncrease, _check_PulseShape, _check_AboveContinuum

    starts, peaks, ends = possible_flares(data) # Find possible flares.

    # Perform some checks to ensure the found flares are valid.
    flare_start, flare_peak, flare_end = [], [], []
    for start, peak, end in zip(starts, peaks, ends):
        check1 = _check_AverageNoise(data, start, peak, end)
        check2 = _check_FluxIncrease(data, start, peak)
        check3 = _check_PulseShape(data, start, peak, end)
        check4 = _check_AboveContinuum(data, start, peak, end)
        logger.debug(f"Flare {round(data['time'].iloc[start],1)}-{round(data['time'].iloc[end],1)}s checks: {check1}/{check2}/{check3}/{check4}")
        if check1 and check2 and check3 and check4:
            flare_start.append(int(start))
            flare_peak.append(int(peak))
            flare_end.append(int(end))

    logger.info(f"Flare finder found {len(flare_start)} flares.")
    return [flare_start, flare_peak, flare_end] if len(flare_start) else False

#################################################################################
### CONTINUUM FITTING
#################################################################################

def fitContinuum(data, flare_indices, use_odr=False):
    logger.debug(f"Starting fitContinuum")

    from .modelling import find_intial_fit, fit_continuum_mcmc

    # Remove flare data.
    if flare_indices:
        logger.critical(f"{flare_indices}")
        for start, end in zip(reversed(flare_indices[0]), reversed(flare_indices[2])):
            data = data.drop(index=range(start, end))

    # Use ODR & AIC to find best number of powerlaw breaks.
    initial_fit, initial_fit_err, initial_fit_stats = find_intial_fit(data)
    break_number = int((len(initial_fit-2)/2)-1)

    # Run MCMC to refine fit.
    if use_odr == True:
        final_par, final_err = initial_fit, initial_fit_err
        logger.debug("Forcing ODR, skipping MCMC fitting.")
    else:
        try:
            final_par, final_err = fit_continuum_mcmc(data, break_number, initial_fit, initial_fit_err)
        except ValueError:
            final_par, final_err = initial_fit, initial_fit_err
            logger.debug(f"MCMC failed, defaulting to ODR.")

    from .utility import calculate_fit_statistics
    final_fit_statistics = calculate_fit_statistics(data, broken_powerlaw, final_par)

    odr_rchisq = initial_fit_stats['rchisq']
    mcmc_rchisq = final_fit_statistics['rchisq']
    logger.debug(f'ODR rchisq: {odr_rchisq}')
    logger.debug(f'MCMC rchisq: {mcmc_rchisq}')

    if mcmc_rchisq == 0 or mcmc_rchisq < 0.1 or mcmc_rchisq == np.inf or mcmc_rchisq == -np.inf:
        logger.debug('MCMC appears to be bad, using ODR fit.')
        final_par, final_err, final_fit_statistics = initial_fit, initial_fit_err, initial_fit_stats

    elif abs(odr_rchisq-1) < abs(mcmc_rchisq-1):
        if abs(odr_rchisq-1) < 1.3 * abs(mcmc_rchisq-1):
            logger.debug("ODR better than MCMC, using ODR fit.")
            final_par, final_err, final_fit_statistics = initial_fit, initial_fit_err, initial_fit_stats
        else:
            logger.debug("ODR better than MCMC fit, but not significantly enough.")

    return {'parameters': final_par, 'errors': final_err, 'fit_statistics': final_fit_statistics}

#################################################################################
### FIT FLARES
#################################################################################

def fitFlares(data, flares, continuum):

    from .modelling import flare_fitter, fred_flare

    if not flares:
        return False

    fitted_model = broken_powerlaw(data.time, continuum['parameters'])
    data_subtracted = data.copy()
    data_subtracted['flux'] = data.flux - fitted_model

    # Fit each flare.
    flare_fits, flare_errs = flare_fitter(data, data_subtracted, flares)

    return {'parameters': flare_fits, 'errors': flare_errs}

#################################################################################
### FIT WHOLE GRB
#################################################################################

def fitGRB(data, flare_indices=None, continuum=None, flares=None):

    check_data_input(data)

    logger.debug(f"Starting fitGRB")

    if flare_indices is None:
        logger.debug(f"Flares not provided - running findFlares function.")
        flare_indices = findFlares(data)

    if continuum is None:
        logger.debug(f"Continuum not provided - running fitContinuum function.")
        continuum = fitContinuum(data, flare_indices)

    if flares is None:
        logger.debug(f"Flare models not provided - running fitFlares function.")
        flares = fitFlares(data, flare_indices, continuum)

    return flare_indices, continuum, flares

#################################################################################
### PLOTTING
#################################################################################

def plotGRB(data, flare_indices=None, continuum=None, flares=None):
    logger.debug(f"Starting plotGRB")
    logger.debug(f"Input flares: {flare_indices}")
    logger.debug(f"Input continuum: {continuum}")

    data_continuum = data.copy()
    flare_data = []

    # For smooth plotting of fitted functions.
    max, min = np.log10(data['time'].iloc[0]), np.log10(data['time'].iloc[-1])
    constant_range = np.logspace(min, max, num=5000)

    if flare_indices:
        for start, peak, end in zip(*flare_indices):
            flare_data.append(data.iloc[start:end+1])
            data_continuum = data_continuum.drop(data.index[start:end+1])
            # plt.axvspan(data.iloc[start].time, data.iloc[end].time, color='r', alpha=0.25)
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
        if flare_indices is None:
            raise ValueError("Cannot input a continuum without flare indices.")

        modelpar, modelerr = continuum['parameters'], continuum['errors']

        nparam = len(modelpar)
        n = int((nparam-2)/2)

        # Print continuum fits.
        slopes = modelpar[:n+1]
        slopes_Err = modelerr[:n+1]
        slopes_info = [f"{slp:.2f} ({slp_err:.2f})" for slp, slp_err in zip(slopes, slopes_Err)]
        breaks = modelpar[n+1:-1]
        breaks_Err = modelerr[n+1:-1]
        breaks_info = [f"{brk:.3g} ({brk_err:.3g})" for brk, brk_err in zip(breaks, breaks_Err)]
        normal = modelpar[-1]
        normal_Err = modelerr[-1]
        normal_info = f"{normal:.2e} ({normal_Err:.2e})"

        logger.info("[ CONTINUUM PARAMETERS ]")
        logger.info("Slopes: {}".format(', '.join(slopes_info)))
        logger.info("Breaks: {}".format(', '.join(breaks_info)))
        logger.info(f"Normal: {normal_info}")

        # Plot continuum model.
        fittedContinuum = broken_powerlaw(constant_range, modelpar)
        total_model = fittedContinuum
        plt.plot(constant_range, fittedContinuum, color='c')

        # Plot powerlaw breaks.
        for x_pos in breaks:
            plt.axvline(x=x_pos, color='grey', linestyle='--', linewidth=0.5)

    if flares:
        if not flare_indices:
            raise ValueError("Cannot input flares without flare_indices.")
        if not continuum:
            raise ValueError("Cannot input flares without continuum.")
        
        from .modelling import fred_flare

        logger.info("[ FLARE PARAMETERS ] - t_start, rise, decay, amplitude")

        for fit, err in zip(flares['parameters'], flares['errors']):
            logger.info(f'Flare - {fit[0]:.2f} ({err[0]:.2f}) / {fit[1]:.2f} ({err[1]:.2f}) / {fit[2]:.2f} ({err[2]:.2f}) / {fit[3]:.2g} ({err[3]:.2g})')
            flare = fred_flare(constant_range, fit)
            total_model += flare # Add flare to total model.
            plt.plot(constant_range, fred_flare(constant_range, fit), color='tab:green', linewidth=0.6) # Plot each flare.

        # Plot total model.
        plt.plot(constant_range, total_model, color='tab:orange')
        upper_flux, lower_flux = data['flux'].max() * 10, data['flux'].min() * 0.1
        plt.ylim(lower_flux, upper_flux)
        plt.title('test')

    plt.loglog()
    plt.show()

    return