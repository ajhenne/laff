import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import warnings

# Ignore warnings.
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .utility import check_data_input


# findFlares() -- locate the indices of flares in the lightcurve
# fitContinuum(flare_indices) -- use the indices to exclude data, then fit the continuum
# fitFlares(flares, continuum) -- use indices + continuum to fit the flares

# fitGRB() -- runs all 3 function in sequence, then does some final cleanups
#          -- final statistics of the whole fit
#          -- this is what the user should be running
#          -- outputs a dictionary with all useful statistics

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
    logger.debug("Starting sequential_findFlares()")
    check_data_input(data) # First check input format is good.

    # Cutoff late data.
    LATE_CUTOFF = True
    data = data[data.time < 2000] if LATE_CUTOFF else data

    from .flarefinding import findFlares

    # Run flare finding.
    flares = findFlares(data)

    logger.info(f"Found {len(flares)} flare(s).")
    return flares if len(flares) else False

#################################################################################
### CONTINUUM FITTING
#################################################################################

def fitContinuum(data, flare_indices, use_odr=False):
    logger.debug(f"Starting fitContinuum")

    from .modelling import find_intial_fit, fit_continuum_mcmc

    # Remove flare data.
    if flare_indices:

        for start, peak, end in flare_indices:
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

    slopes = final_par[:break_number+1]
    slopes_err = final_err[:break_number+1]
    breaks = final_par[break_number+1:-1]
    breaks_err = final_err[break_number+1:-1]
    normal = final_par[-1]
    normal_err = final_err[-1]

    return {'parameters': {
                'break_num': break_number,
                'slopes': slopes, 'slopes_err': slopes_err,
                'breaks': breaks, 'breaks_err': breaks_err,
                'normal': normal, 'normal_err': normal_err},
            'fit_statistics': final_fit_statistics}

#################################################################################
### FIT FLARES
#################################################################################

def fitFlares(data, flares, continuum):

    from .modelling import flare_fitter

    if not flares:
        return False

    fitted_model = broken_powerlaw(data.time, continuum['parameters'])
    data_subtracted = data.copy()
    data_subtracted['flux'] = data.flux - fitted_model

    # Fit each flare.
    flare_fits, flare_errs = flare_fitter(data, data_subtracted, flares)

    ### TODO Calculate fluence.

    fittedFlares = []
    for indices, par, err in zip(flares, flare_fits, flare_errs):
        fittedFlares.append({'times': [data.iloc[x].time for x in indices], 'par': par, 'par_err': err})

    return fittedFlares

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

    logger.info(f"LAFF run finished.")

    return {'flares': flares, 'continuum': continuum}

#################################################################################
### PLOTTING
#################################################################################

def plotGRB(data, flare_indices=None, continuum=None, flares=None):
    logger.info(f"Starting plotGRB")
    logger.debug(f"Input flares: {flare_indices}")
    logger.debug(f"Input continuum: {continuum}")

    data_continuum = data.copy()
    flare_data = []

    # For smooth plotting of fitted functions.
    max, min = np.log10(data['time'].iloc[0]), np.log10(data['time'].iloc[-1])
    constant_range = np.logspace(min, max, num=5000)

    if flare_indices:
        for start, peak, end in flare_indices:
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
    plt.loglog()
    plt.show()

    return
