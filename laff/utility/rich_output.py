import matplotlib.pyplot as plt
import numpy as np
import logging
from .utility import get_xlims

logger = logging.getLogger('laff')

def plot_all_break_fits(data, model_fits, broken_powerlaw):
    """Plot all powerlaw break fits."""

    logger.debug("Running plot_all_break_fits()")

    minval, maxval = np.log10(data['time'].iloc[0]), np.log10(data['time'].iloc[-1])
    constant_range = np.logspace(minval, maxval, num=250)

    fig, axs = plt.subplots(6, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, left=0, bottom=0, right=1, top=1)
    
    for fit, ax in zip(model_fits, axs):
        fit = list(fit[0])
        ax.errorbar(data.time, data.flux,
                xerr=[-data.time_nerr, data.time_perr], \
                yerr=[-data.flux_nerr, data.flux_perr], \
                marker='', linestyle='None', capsize=0, zorder=1, color='k')

        fittedContinuum = broken_powerlaw(constant_range, fit)
        ax.plot(constant_range, fittedContinuum, color='c')

        n = int((len(fit)-2)/2)
        
        if n > 0:
            for xpos in fit[n+1:-1]:
                ax.axvline(x=xpos, color='grey', linestyle='--', linewidth=0.5, zorder=0)
        ax.loglog()
        
    plt.xlim(get_xlims(data))
    plt.margins(x=0, y=0)
    plt.show()




    return