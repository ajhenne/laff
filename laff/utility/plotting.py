import matplotlib.pyplot as plt
import logging
import numpy as np
from .utility import get_xlims
from ..modelling import broken_powerlaw, fred_flare

logger = logging.getLogger('laff')



def plotGRB(data, grb_fit, save_path=None, show_fig=True, **plot_components):
    """Plot the results of LAFF fit.

    Take the output of a LAFF run and plot the flares, continuum and lightcurve
    of a GRB.

    Args:
        data (pd.Dataframe): GRB lightcurve dataframe.
        grb_fit (dict): the output of laff.fitGRB containing all the fit data.
        save_path (string): default do not save, if provided it will save to given path.
        show (bool): defaults to true, if false do not call plt.show().
        **kwargs:
            lightcurve, continuums, flares, total_model, powerlaw_breaks

    Returns:
        N/A
        TODO - can I return the plotted object?
    """

    # Light curve.
    if not plot_components.get('lightcurve', False):
        logger.debug("Plottting lightcurve")
        plt.errorbar(data.time, data.flux, xerr=[-data.time_nerr, data.time_perr], 
                     yerr=[-data.flux_nerr, data.flux_perr], marker='', linestyle='None', capsize=0, zorder=1)

        upper_flux, lower_flux = data['flux'].max() * 10, data['flux'].min() * 0.1
        
    # For smooth plotting of fitted functions.
    max, min = np.log10(data['time'].iloc[0]), np.log10(data['time'].iloc[-1])
    constant_range = np.logspace(min, max, num=5000)

    # Continuum model.
    if not plot_components.get('continuum', False):
        logger.debug('Plotting continuum model')
        fitted_continuum = broken_powerlaw(constant_range, grb_fit['continuum']['parameters'])
        plt.plot(constant_range, fitted_continuum, color='c')

    try:
        total_model = fitted_continuum
    except NameError:
        total_model = [0] * len(data.time)

    # Flare models.
    if grb_fit['flares'] is not False:
        for flare in grb_fit['flares']:
            flare_model = fred_flare(constant_range, flare['par'])
            total_model += flare_model
            if not plot_components.get('flares', False):
                plt.plot(constant_range, flare_model, color='tag:green', linewidth=0.6, zorder=3)

    # Total model.
    if not plot_components.get('total_model', False):
        plt.plot(constant_range, total_model, color='tab:orange', zorder=5)

    # Powerlaw breaks.
    if not plot_components.get('powerlaw_breaks', False):
        for x_pos in grb_fit['continuum']['parameters']['breaks']:
            plt.axvline(x=x_pos, color='grey', linestyle='--', linewidth=0.5, zorder=0)

    # Adjust limits.
    plt.ylim(lower_flux, upper_flux)
    plt.xlim(get_xlims(data))
    plt.loglog()
    plt.xlabel("Time (s)")
    plt.ylabel("Flux (units)")

    if save_path is not None:
        plt.savefig(save_path)

    if show_fig:
        plt.show()

    return


def saveGRB(data, grb_fit, save_path):
    """Wrapper function to plot and save the LAFF results without showing them."""
    plotGRB(data, grb_fit, save_path, show=False)
    return