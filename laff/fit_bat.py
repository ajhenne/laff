import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import label
from scipy.optimize import least_squares
from scipy.stats import f

################################################################################
# FUNCTIONS
################################################################################

def fred_flares(params, x):
    # J. P. Norris et al., ‘Attributes of Pulses in Long Bright Gamma-Ray Bursts’, The Astrophysical Journal, vol. 459, p. 393, Mar. 1996, doi: 10.1086/176902.

    x = np.array(x)

    flr_params = [params[i:i + 5] for i in range(0, len(params), 5)]

    total_model = [0.0] * len(x)

    for flr in flr_params:

        t_max, rise, decay, sharpness, amplitude = flr

        flr_model = amplitude * np.exp( -(abs(x - t_max) / rise) ** sharpness)
        flr_model[np.where(x > t_max)] = amplitude * np.exp( -(abs(x[np.where(x > t_max)] - t_max) / decay) ** sharpness)

        total_model += flr_model

    return total_model

def fred_resids(params, x, y):
    return fred_flares(params, x) - y


################################################################################
# DATA FILTERING
################################################################################

def fitPrompt(data):

    data, flares = filter_data(data)

    # continuum = find_continuum(data, flares)

    flares = fit_flares(data, flares)

    return {'data': data, 'flares': flares}


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

        if amplitude < 1.5 * np.average(data['moving_std'].iloc[a:b]):
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
    flare_data['flux'] -= flare_data['moving_std']
    flare_data['flux'] = flare_data['flux'].apply(lambda x: max(x, 0))

    flares = []

    for a, b in flare_indices:

        add_flare = True
        count_flare = 1

        found_peaks, properties = find_peaks(flare_data['flux'].iloc[a:b+1], prominence=flare_data['moving_std'].iloc[a])            
        ranked_indices = np.argsort(properties['prominences'])[::-1]

        while add_flare == True:

            peaks = [found_peaks[i]+a for i in ranked_indices[:count_flare]]

            if len(peaks) == 0 and count_flare == 1:
                peaks = [data['savgol'].iloc[a:b].idxmax()]

            if len(peaks) != count_flare:
                add_flare = False
                continue

            input_par = []

            for peak in peaks:

                t_peak    = data['time'].iloc[peak]
                rise      = (data['time'].iloc[peak]-data['time'].iloc[a]) / 4 or (data['time'].iloc[b] - np.average((data['time'].iloc[a], data['time'].iloc[b]))) / 4
                decay     = (data['time'].iloc[b]-data['time'].iloc[peak]) / 2 or (data['time'].iloc[b] - np.average((data['time'].iloc[a], data['time'].iloc[b]))) / 2
                sharp     = min(max(decay/rise, 1), 10)
                amplitude = data['savgol'].iloc[peak]

                input_par.extend((t_peak, rise, decay, sharp, amplitude))
            
            bound_par = count_flare * [data['time'].iloc[a], 0.0,            0.0,            1.0,  0.0   ], \
                        count_flare * [data['time'].iloc[b], 2*(rise+decay), 2*(rise+decay), 10.0, np.inf]

            flare_fit = least_squares(fred_resids, input_par, bounds=bound_par, args=(flare_data['time'], flare_data['flux'])).x

            filt_data = data[data['flux'] != 0]

            chi_2 = np.sum( ((filt_data['flux'] - fred_flares(flare_fit, filt_data['time']))**2) / (filt_data['flux_perr'] ** 2) )
            dof_2 = len(data['flux']) - len(flare_fit)

            if count_flare == 1:
                chi_1 = chi_2
                dof_1 = dof_2
                count_flare += 1
                prev_fit = flare_fit
                continue

            F = ((chi_1 - chi_2) / (dof_1 - dof_2)) / (chi_2/dof_2)
            p_value = 1 - f.cdf(F, dof_1-dof_2, dof_2)

            if p_value < 0.0027:
                chi_1 = chi_2
                dof_1 = dof_2
                count_flare += 1
                prev_fit = flare_fit
            else:
                flare_fit = prev_fit
                add_flare = False

        flare_data['flux'] -= fred_flares(flare_fit, data['time'])
        flare_data['flux'] = flare_data['flux'].apply(lambda x: max(x, 0))

        for i in range(0, len(flare_fit), 5):
            flares.append({'indices': (a, b), 'parameters': flare_fit[i:i+5]})

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
        ax1.errorbar(data['time'], data['flux'], yerr=data['flux_perr'], linestyle='None', marker='.', color='grey', linewidth=0.5, alpha=0.3, zorder=-1)

    # Savgol filter line.
    if kwargs.get('savgol', True):
        ax1.plot(data['time'], data['savgol'], color='tab:green', linewidth=0.5)

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

        flare_model = fred_flares(flare['parameters'], data['time'])
        total_model += flare_model

        if kwargs.get('flare_fit', True):
            ax1.plot(data['time'], flare_model, color='m', linewidth=0.5, linestyle='--')
            if residuals:
                ax2.plot(data['time'], flare_model, color='m', linewidth=0.5, linestyle='--')

    if kwargs.get('total_fit', True):
        ax1.plot(data['time'], total_model, color='black', linewidth=0.7)

    plt.xlabel('Time since trigger (seconds)')
    plt.ylabel('Count rate (/seconds)')

    if (save_path := kwargs.get('save')):
        plt.savefig(save_path + grb_name + '.png', bbox_inches='tight')

    if kwargs.get('show', True):
        plt.show()