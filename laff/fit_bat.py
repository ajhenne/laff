"""
fit_bat.py

This module provides light curve fitting routines for Swift-BAT data, specifically
tuned for 64 ms data. The data is filtered to first remove noise, and then identify
deviation regions above this. Peaks above a certian prominence are found, and a FRED
flare fitted to each. It also includes a plotting function.

Functions:
    - fitPrompt: Entire fitting routine that outputs fitted pulses.
    - plotPrompt: Plotting function for the modelled light curve.

TODO:
    - t_peak_bound minimum to be determined dynamically based on bin length
    - window lengths to be dynamic in case of different lengthed data?
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import label
from scipy.optimize import least_squares, fmin_slsqp
from scipy.stats import f
from .modelling import fred_flare, sum_residuals
from .utility import calculate_fit_statistics, calculate_fluence, calculate_par_err

def fitPrompt(data):

    data = data[data['flux'] != 0.0].reset_index(drop=True)

    data, pulses = filter_data(data)

    # continuum = find_continuum(data, pulses)

    pulses = fit_pulses(data, pulses)

    return {'data': data, 'pulses': pulses}

################################################################################
# DATA FILTERING
################################################################################

def filter_data(data):

    data['avg'] = data['flux_perr'].rolling(window=500, min_periods=1).mean()
    data['flux_broad'] = data['flux'].rolling(window=50, center=True).mean()
    data['flux_fine'] = data['flux'].rolling(window=10, center=True).mean()

    data['flux_err_std'] = data['flux_perr'].rolling(window=500, min_periods=1).std()

    data['deviation'] = data[(data['flux_broad'] > 5 * data['flux_err_std']) | (data['flux_fine'] > 7.5 * data['flux_err_std'])]['flux_broad']

    # Aggregate deviation data into regions.
    labelled_array, num_features = label(data['deviation'] > 0)
    intial_flare_regions = []

    for i in range(1, num_features+1):
        indices = np.where(labelled_array == i)[0]
        if not indices[0] == indices[-1]:
            intial_flare_regions.append((indices[0], indices[-1]))

    return data, intial_flare_regions

################################################################################
# FLARE FITTING
################################################################################

def fit_pulses(data, region_indices):

    region_peaks = []

    for idx_start, idx_end in region_indices:

        region_data = data.iloc[idx_start:idx_end]

        flare_peaks, _ = find_peaks(
            region_data['flux_fine'],
            height = 1.5*np.average(region_data['flux_err_std']),
            distance = 2,
            width = 3,
            prominence = 0.5 * np.median(region_data['avg'])
        )

        flare_peaks = [x + idx_start for x in flare_peaks]
        region_peaks.append(flare_peaks)

    data['residuals'] = data['flux_fine'].copy()

    pulses = []

    for (idx_start, idx_end), peaks in zip(region_indices, region_peaks):

        if len(peaks) == 0:
            continue

        flare_count = len(peaks)

        input_par = []
        bound_par = []

        for flare_peak in peaks:

            region_width = data['time'].iloc[idx_end] - data['time'].iloc[idx_start].item()
            estimate_flare_width = region_width / (flare_count * 1) # initial guess of evenly spread pulses

            t_peak = data['time'].iloc[flare_peak].item()
            rise   = estimate_flare_width / 2 * 2.3 # d ~ actual time width / 2.3
            decay  = estimate_flare_width / 2 * 2.3
            sharp  = 2.0
            amplitude = data['flux_fine'].iloc[flare_peak].item()

            input_par.extend((t_peak, rise, decay, sharp, amplitude))

            t_peak_bound = [t_peak-2, t_peak+2] # appropriate for late pulses?
            rise_bound = [0.032, estimate_flare_width/2] # ~64 ms
            decay_bound = [0.032, estimate_flare_width/2]
            sharp_bound = [1.0, 5.0]
            amplitude_bound = [0, data['flux_fine'].iloc[idx_start:idx_end].max()]

            bound_par.extend((t_peak_bound, rise_bound, decay_bound, sharp_bound, amplitude_bound))

        def all_constraints():
            pass

        fitted_pulses = fmin_slsqp(sum_residuals, input_par, bounds=bound_par, args=(data.time, data.residuals, data.flux_perr), iter=200, iprint=0)

        fitted_stats = calculate_fit_statistics(data, fred_flare, fitted_pulses)

        for i in range(0, len(fitted_pulses), 5):
            
            data['residuals'] -= fred_flare(fitted_pulses[i:i+5], data['time'])

            start_time = fitted_pulses[i] - (fitted_pulses[i+1] * (-np.log(0.01)) **(1/fitted_pulses[i+3]))
            end_time = fitted_pulses[i] + (fitted_pulses[i+2] * (-np.log(0.01)) **(1/fitted_pulses[i+3]))

            fluence_rise = calculate_fluence(fred_flare, fitted_pulses[i:i+5], start_time, fitted_pulses[i], 1)
            fluence_decay = calculate_fluence(fred_flare, fitted_pulses[i:i+5], fitted_pulses[i], end_time, 1)  
            fluence_total = fluence_rise + fluence_decay
            
            pulses.append({'indices': (start_time, end_time), 'parameters': fitted_pulses[i:i+5], 'fluence': [fluence_rise, fluence_decay, fluence_total], 'fit_statistics': fitted_stats})


    return pulses

    
################################################################################
# PLOTTING
################################################################################

def plotPrompt(prompt_fit, **kwargs):
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        prompt_fit (_type_): _description_
    
    Kwargs:
        grb_name (str):   grb name string to title the plot.
        zero_lines (bool): plot the y=0 dashed lines.
        main_data (bool): plot the main data, default true
        residuals (bool): plot the residuals, default true
        flare_spans (bool): plot the flare spans, default true.
        flare_fit
        total fit
        save
        TODO savepath
    """

    data = prompt_fit['data']

    residuals = kwargs.get('residuals', True)

    if residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,6), gridspec_kw={'hspace': 0})
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))

    constant_range = np.linspace(data['time'].iloc[0], data['time'].iloc[-1], num=len(data['time']) * 10)

    # Plot title.
    if (grb_name := kwargs.get('grb_name')):
        fig.suptitle(str(grb_name))
    
    # Zero lines.
    if kwargs.get('zero_lines', True):
        ax1.axhline(y=0, linestyle='--', color='grey', linewidth=0.5)
        if residuals:
            ax2.axhline(y=0, linestyle='--', color='grey', linewidth=0.5)

    # Main data points.
    if kwargs.get('main_data', True):
        ax1.errorbar(data['time'], data['flux'], yerr=data['flux_perr'], linestyle='None', marker='', color='grey', linewidth=0.3, alpha=0.2, zorder=-1)

    # Savgol filter line.
    if kwargs.get('savgol', True):
        ax1.plot(data['time'], data['flux_fine'], color='black', linewidth=0.5, linestyle='--')
        ax1.plot(data['time'], data['flux_broad'], color="#408EC2", linewidth=1)
        ax1.plot(data['time'], data['flux_err_std']*5.0, color='r', linewidth=0.5)
        ax1.plot(data['time'], data['flux_err_std']*7.5, color='r', linewidth=0.5)


    total_model = [0.0] * constant_range

    for flare in prompt_fit['pulses']:

        srt, end = flare['indices']

        if kwargs.get('flare_spans', True):
            # ax1.axvspan(srt, end, color='b', alpha=0.2)
            pass

        flare_model = fred_flare(flare['parameters'], constant_range)
        total_model += flare_model

        if kwargs.get('flare_fit', True):
            ax1.plot(constant_range, flare_model, color='#2274A5', linewidth=2)
            pass

    if kwargs.get('total_fit', True):
        ax1.plot(constant_range, total_model, color='tab:orange', linewidth=2)
        pass

    plt.xlabel('Time since trigger (s)', fontsize=16)
    plt.ylabel('Count rate (counts/s)', fontsize=16)

    # plt.axvline(data['time'].iloc[0] * 0.95, color='r')
    # plt.axvline(data['time'].iloc[-1] * 0.95, color='r')
    # plt.plot(data['time'], data['negative_noise'], color='m')

    # plt.xlim(-50, 200)
    # plt.ylim(-0.2, 0.4)
    if (save_path := kwargs.get('save')):
        plt.savefig(save_path + grb_name + '.png', bbox_inches='tight')
    if kwargs.get('show', True):
        plt.show()