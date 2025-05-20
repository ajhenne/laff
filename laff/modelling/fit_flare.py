import numpy as np
import logging
from scipy.optimize import fmin_slsqp
from scipy.signal import find_peaks
from scipy.stats import f
from ..utility import calculate_fit_statistics, calculate_par_err, calculate_fluence 
from ..modelling import broken_powerlaw

logger = logging.getLogger('laff')

#################################################################################
### FRED MODEL
#################################################################################

def fred_flare(params, x):
    # J. P. Norris et al., ‘Attributes of Pulses in Long Bright Gamma-Ray Bursts’, The Astrophysical Journal, vol. 459, p. 393, Mar. 1996, doi: 10.1086/176902.
    
    x = np.array(x)

    flr_params = [params[i:i+5] for i in range(0, len(params), 5)]

    total_model = [0.0] * len(x)

    for flr in flr_params:
        t_max, rise, decay, sharpness, amplitude = flr

        flr_model = amplitude * np.exp( -(abs(x - t_max) / rise) ** sharpness)
        flr_model[np.where(x > t_max)] = amplitude * np.exp( -(abs(x[np.where(x > t_max)] - t_max) / decay) ** sharpness)

        total_model += flr_model

    return total_model

def sum_residuals(params, *args):
    x, y, y_err = args
    return np.sum(((y - fred_flare(params, x)) / y_err) ** 2)

#################################################################################
### SCIPY.ODR FITTING
#################################################################################

def flare_fitter(data, continuum, flares, model='fred'):
    """ 
    Flare fitting function. Takes already found flare indices and models them.

    Also runs:
      - 
    
    """

    logger.info("Fitting flares...")

    data['residuals'] = data['flux'] - broken_powerlaw(continuum['params'], data['time'])

    flareFits    = []
    flareStats   = []
    flareErrs    = []
    flareIndices = []

    for start, peak, end in flares:

        try_another = True
        flare_count = 1

        found_maxima, properties = find_peaks(data['residuals'].iloc[start:end], prominence=data['residuals'].iloc[start])
        ranked_maxima_idx = np.argsort(properties['prominences'][::-1])

        t_start = data['time'].iloc[start]
        t_peak  = data['time'].iloc[peak]
        t_end   = data['time'].iloc[end]

        while try_another == True:

            maximums = [found_maxima[i]+start for i in ranked_maxima_idx[:flare_count]]

            if len(maximums) and flare_count == 1:
                maximums = [peak]

            if len(maximums) != flare_count:
                try_another = False
                continue

            input_par = []

            for i, pk in enumerate(maximums):

                # Parameter guesses.
                t_peak = t_peak
                rise  = (t_peak - t_start)
                decay = (t_end - t_peak)
                sharpness = 2
                amplitude = data['flux'].iloc[pk]

                input_par.extend((t_peak, rise, decay, sharpness, amplitude))

            bounds = flare_count * ([t_start, t_end], [rise/10, t_end-t_start], [decay/10, t_end-t_start], [1.0, 5.0], [data['flux'].iloc[start], data['flux'].iloc[peak]*3])

            def all_constraints(params, *args):
                x_points = [data['time'].iloc[x] for x in (start, peak, end)]
                y_points = [data['residuals'].iloc[x] for x in (start, peak, end)]

                upper_limits = y_points - fred_flare(params, x_points)
                return upper_limits

            fitted_flare = fmin_slsqp(sum_residuals, input_par, bounds=bounds, f_ieqcons=all_constraints, args=(data.time, data.residuals, data.flux_perr), iter=100, iprint=0)

            if flare_count == 1:
                chi_1, dof_1 = determine_additional_flare(fitted_flare, data, flare_count, None, None)
                flare_count += 1
                flare_fit = fitted_flare
                prev_fit = fitted_flare
                continue
            else:
                p_value, chi_2, dof_2 = determine_additional_flare(fitted_flare, data, flare_count, chi_1, dof_1)
                if p_value < 0.0027:
                    chi_1 = chi_2
                    dof_1 = dof_2
                    flare_count += 1
                    prev_fit = fitted_flare
                    continue
                else:
                    flare_fit = prev_fit
                    flare_count -= 1
                    try_another = False

        logger.debug(f"Flare {start}/{peak}/{end} fitted")
        logger.debug('\tconsists of %s flares', flare_count)

        for i in range(0, len(flare_fit), 5):
        
            individual_par = flare_fit[i:i+5]
            
            def chi2_wrapper(individual_par):
                return sum_residuals(individual_par, data['time'], data['residuals'], data['flux_perr'])
            individual_errors = calculate_par_err(individual_par, chi2_wrapper)
        
            data['residuals'] = data['residuals'] - fred_flare(individual_par, data.time)

            individual_stats = calculate_fit_statistics(data, fred_flare, fitted_flare)

            flareFits.append(list(individual_par))
            flareErrs.append(list(individual_errors))
            flareStats.append(list(individual_stats))
            flareIndices.append([start, peak, end])

            logger.debug('\tparams\t%s', list(round(x, 2) for x in individual_par))
            logger.debug('\terrors\t%s', list(round(x, 2) for x in individual_errors))

    logger.info("Flare fitting complete for all flares.")
    return flareFits, flareStats, flareErrs, flareIndices

def determine_additional_flare(par, data, count, chi_1, dof_1):

    chi_2 = sum_residuals(par, data['time'], data['residuals'], data['flux_perr'])
    dof_2 = len(data['flux']) - len(par)
    
    if count == 1:
        return chi_2, dof_2
    
    F = ((chi_1 - chi_2) / (dof_1 - dof_2)) / (chi_2/dof_2)
    p_value = 1 - f.cdf(F, dof_1-dof_2, dof_2)

    return p_value, chi_2, dof_2

#################################################################################
### NEAT PACKAGING
#################################################################################

def package_flares(data, fits, stats, errs, indices, count_ratio=1.0):

    flaresDict = []

    for idx, fit, stat, err in zip(indices, fits, stats, errs):

        # tmax, rise, decay, sharp, ampl
        # 0     1     2      3      4
        fitted_flare = fred_flare(fit, data['time'])

        start_time = fit[0] - (fit[1] * (-np.log(0.001)) **(1/fit[3]))
        peak_time  = fit[0]
        end_time = fit[0] + (fit[2] * (-np.log(0.001)) **(1/fit[3]))
        times = [start_time, peak_time, end_time]

        fluence_rise  = calculate_fluence(fred_flare, fit, start_time, peak_time, count_ratio=count_ratio)
        fluence_decay = calculate_fluence(fred_flare, fit, peak_time, end_time, count_ratio=count_ratio)
        fluences = [fluence_rise, fluence_decay, fluence_rise + fluence_decay]

        peak_flux = max(fitted_flare)
        
        flaresDict.append({'times': times, 'indices': idx, 'params': fit, 'stats': stat, 'errors': err, 'fluence': fluences, 'peak_flux': peak_flux})

    return flaresDict