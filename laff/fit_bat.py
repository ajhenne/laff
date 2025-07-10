import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import label
from scipy.optimize import least_squares, fmin_slsqp
from scipy.stats import f
from .modelling import fred_flare, sum_residuals
from .utility import calculate_fit_statistics, calculate_fluence, calculate_par_err

def fitPrompt(data):

    data, flares = filter_data(data)

    # continuum = find_continuum(data, flares)

    flares = fit_flares(data, flares)

    return {'data': data, 'flares': flares}

################################################################################
# DATA FILTERING
################################################################################

def filter_data(data):

    data['savgol'] = savgol_filter(data['flux'], window_length=31, polyorder=3)

    # Filter out negatives.
    filtered_data = data['savgol'].copy()
    filtered_data = filtered_data[filtered_data > 0]
    avg_positive = np.average(filtered_data)
    
    # Filter out big peaks.
    filtered_data = filtered_data[filtered_data < 3 * avg_positive]
    avg_filter = np.average(filtered_data)

    # Calculate residuals.
    data['savgol_residuals'] = data['savgol'] - 3 * avg_filter
    data['savgol_residuals'] = data['savgol_residuals'].apply(lambda x: max(x, 0))
    
    # Calculate rolling std.
    # Cutoff higher peaks.
    modified_flux = data['flux'].copy()
    prev_valid = None
    for i in range(len(modified_flux)):
        if modified_flux[i] > 3 * avg_filter:
            if prev_valid is not None:
                modified_flux[i] = prev_valid
        else:

            prev_valid = modified_flux[i]
    data['moving_std'] = modified_flux.rolling(window=501, min_periods=1).std()
    data.loc[0, 'moving_std'] = data['moving_std'].iloc[1]

    # Gather all flare region indices from residuals.
    labelled_array, num_features = label(data['savgol_residuals'] > 0)
    intial_flare_regions = []

    for i in range(1, num_features+1):
        indices = np.where(labelled_array == i)[0]
        if not indices[0] == indices[-1]:
            intial_flare_regions.append((indices[0], indices[-1]))

    flare_indices = []

    for a, b in intial_flare_regions:
        peak_index = data['savgol'].iloc[a:b].idxmax()
        amplitude = data['savgol'].iloc[peak_index]

        if amplitude < 2 * np.average(data['moving_std'].iloc[a:b]):
            continue
        if len(range(a, b)) <= 1:
            continue

        flare_indices.append((a, b))

    flare_indices = sorted(set(flare_indices))

    return data, flare_indices


################################################################################
# FLARE FITTING
################################################################################

def fit_flares(data, flare_indices):

    flare_data = data.copy()
    flare_data['untouched_flux'] = flare_data['flux']
    flare_data['flux'] -= flare_data['moving_std']
    flare_data['flux'] = flare_data['flux'].apply(lambda x: max(x, 0))

    flares = []

    for dev_start, dev_end in flare_indices:

        try_another = True
        flare_count = 1

        found_maxima, properties = find_peaks(flare_data['flux'].iloc[dev_start:dev_end+1], prominence=flare_data['moving_std'].iloc[dev_start])            
        ranked_maxima_idx = np.argsort(properties['prominences'])[::-1]

        t_start = data['time'].iloc[dev_start]
        t_end   = data['time'].iloc[dev_end]

        while try_another == True:

            input_par = []
            bounds = []

            peaks = [found_maxima[i]+dev_start for i in ranked_maxima_idx[:flare_count]]

            if len(peaks) == 0 and flare_count == 1:
                peaks = [data['savgol'].iloc[dev_start:dev_end].idxmax()]

            if len(peaks) != flare_count:
                try_another = False
                continue

            for i, peak in enumerate(peaks):

                # Parameter guesses
                t_peak    = data['time'].iloc[peak]
                rise      = (t_peak - t_start) / (6 * i+1)
                decay     = (t_end - t_peak)   / (4 * i+1)
                sharp     = 2
                amplitude = data['savgol'].iloc[peak]

                input_par.extend((t_peak, rise, decay, sharp, amplitude))
            
                # Parameter bounds
                t_peak_bound    = [data['time'].iloc[dev_start], data['time'].iloc[dev_end]]
                rise_bound      = [rise/10, t_end-t_start]
                decay_bound     = [decay/10, t_end-t_start]
                sharp_bound     = [1.0, 5.0]
                amplitude_bound = [0, 2 * data['flux'].max()]

                bounds.extend((t_peak_bound, rise_bound, decay_bound, sharp_bound, amplitude_bound))

            def all_constraints():
                pass

            print('yes')
            fitted_flare = fmin_slsqp(sum_residuals, input_par, bounds=bounds, args=(flare_data.time, flare_data.untouched_flux, flare_data.flux_perr), iter=200, iprint=0)

            fitted_stats = calculate_fit_statistics(flare_data, fred_flare, fitted_flare)

            if flare_count == 1:
                prev_fits = fitted_flare
                prev_stat = fitted_stats
                flare_count += 1
                continue

            else:
                if fitted_stats['BIC'] + 6 < prev_stat['BIC']:
                    prev_fits = fitted_flare
                    prev_stat = fitted_stats
                    flare_count += 1
                    continue
                else:
                    fitted_flare = prev_fits
                    fitted_stats = prev_stat
                    flare_count -= 1
                    try_another = False

        flare_data['flux'] -= fred_flare(fitted_flare, flare_data['time'])
        flare_data['flux'] = flare_data['flux'].apply(lambda x: max(x, 0))

        for i in range(0, len(fitted_flare), 5):

            fluence_rise = calculate_fluence(fred_flare, fitted_flare[i:i+5], t_start, fitted_flare[i], 1)
            fluence_decay = calculate_fluence(fred_flare, fitted_flare[i:i+5], fitted_flare[i], t_end, 1)  
            fluence_total = fluence_rise + fluence_decay
            
            flares.append({'indices': (dev_start, dev_end), 'parameters': fitted_flare[i:i+5], 'fluence': [fluence_rise, fluence_decay, fluence_total], 'fit_statistics': fitted_stats})

    return flares

    
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
        ax1.errorbar(data['time'], data['flux'], yerr=data['flux_perr'], linestyle='None', marker='.', color='grey', linewidth=0.5, alpha=0.4, zorder=-1)

    # Savgol filter line.
    if kwargs.get('savgol', True):
        ax1.plot(data['time'], data['savgol'], color='black', linewidth=0.5, linestyle='--')

    # Residuals.
    if residuals:
        ax2.plot(data['time'], data['savgol_residuals'], linewidth=0.5, color='tab:green')


    total_model = [0.0] * len(data['flux'])

    for flare in prompt_fit['flares']:

        srt, end = flare['indices']

        if kwargs.get('flare_spans', True):
            ax1.axvspan(data['time'].iloc[srt], data['time'].iloc[end], color='b', alpha=0.2)
            if residuals:
                ax2.axvspan(data['time'].iloc[srt], data['time'].iloc[end], color='b', alpha=0.2)

        flare_model = fred_flare(flare['parameters'], data['time'])
        total_model += flare_model

        if kwargs.get('flare_fit', True):
            ax1.plot(data['time'], flare_model, color='#2274A5', linewidth=1)
            if residuals:
                ax2.plot(data['time'], flare_model, color='cyan', linewidth=1)

    if kwargs.get('total_fit', True):
        ax1.plot(data['time'], total_model, color='tab:orange', linewidth=2)

    plt.xlabel('Time since trigger (s)', fontsize=16)
    plt.ylabel('Count rate (counts/s)', fontsize=16)

    # plt.xlim(-50, 200)
    # plt.ylim(-0.2, 0.4)
    if (save_path := kwargs.get('save')):
        plt.savefig(save_path + grb_name + '.png', bbox_inches='tight')
    if kwargs.get('show', True):
        plt.show()