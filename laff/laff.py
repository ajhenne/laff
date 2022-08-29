
"""laff.laff: provides entry point main()."""

__version__ = "0.2.0"

import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from operator import itemgetter

from .models import Models
from .lcimport import Imports

# Silence the double scalar warnings.
warnings.filterwarnings("ignore")

# laff <data_filepath> <output_filepath> <print_dirname> <rise_condition> <decay_condition>

parser = argparse.ArgumentParser(description="Lightcurve and flare fitter for GRBs", prog='laff')
                                # epilog="")

parser.add_argument('data', nargs=1, metavar='data_filepath', type=str, help='Path to the input datafile.')

parser.add_argument('-o', '--output', nargs=1, metavar='output', type=str, help='Output file to write results to.')

parser.add_argument('-r', '--rise', nargs=1, metavar='rise_condition', type=float, help="Condition to alter the flare finding algorithm. A higher value makes it stricter (default: 2).")
parser.add_argument('-d', '--decay', nargs=1, metavar='decay_condition', type=float, help="Condition to alter the decay finding algorithm. A higher vaules makes it stricter (default: 4).")

parser.add_argument('-p', '--print', action='store_true', help='Show the fitted lightcurve.')
parser.add_argument('-f', '--filetype', nargs=1, metavar='filetype', help='Changed the input filetype/mission (default: Swift/XRT .qdp).')
parser.add_argument('-v', '--verbose', action='store_true', help="Produce more detailed text output in the terminal window.")

args = parser.parse_args()

### Data filepath.
input_path = args.data[0]

### Write output to table.
if args.output:
    output_path = args.output[0]

### Rise variable.
if args.rise:
    RISECONDITION = args.rise[0]
else:
    RISECONDITION = 2

### Decay variable.
if args.decay:
    DECAYCONDITION = args.decay[0]
else:
    DECAYCONDITION = 4

### Filetype.
swiftxrt = ('swift', 'xrt', 'swiftxrt') # Supported missions/filetypes.
filetype = swiftxrt # Default filetype.

if args.filetype:
    if args.filetype[0] in swiftxrt:
        filetype = swiftxrt
    else:
        print("ERROR: filetype '%s' not supported." % args.filetype[0])
        sys.exit(1)

###############################################################
### MAIN
###############################################################

def main():

    data = importData(input_path, filetype)

    #### FIND FLARES
    fl_start, fl_peak, fl_end = FlareFinder(data)

    ### FIT CONTINUUM
    data_excluded = data[data['flare'] == False]
    models, fits, pars = FitContinuum(data_excluded)

    ### FIND BEST FIT
    powerlaw, parameters, stats = BestContinuum(data_excluded,models,pars)
    residuals = data.flux - powerlaw(parameters, np.array(data.time))

    ## TEMPORARY
    fl_start, fl_peak, fl_end = fl_start[0:3], fl_peak[0:3], fl_end[0:3]
    
    flareParams = []
    for start, peak, end in zip(fl_start, fl_peak, fl_end):
        flare, residuals = FitFlare(data, start, peak, end, residuals)
        flareParams.append(flare)

    constant_range = np.logspace(np.log10(data['time'].iloc[0]),
                                 np.log10(data['time'].iloc[-1]), num=2000)

    finalModel = powerlaw(parameters, np.array(data.time))
    for flare in flareParams:
        finalModel += Models.flare_gaussian(flare, np.array(data.time))

    finalRange = powerlaw(parameters, constant_range)
    for flare in flareParams:
        finalRange += Models.flare_gaussian(flare, constant_range)

    if not args.verbose:
        printResults(fl_start, parameters, stats)
    else:
        printResults_verbose(data, fl_start, fl_peak, fl_end, powerlaw, parameters, stats)

    ### PLOT RESULTS
    if args.print:
        plotResults(data, finalRange, parameters)

###############################################################
### GENERAL FUNCTIONS
###############################################################

def tableValue(data, index, column):
    return data['%s' % column].iloc[index]

def uniqueList(duplicate_list):
    unique_list = list(set(duplicate_list))
    unique_list.sort()
    return unique_list

def slope(data, p1, p2):
    deltaFlux = tableValue(data,p2,"flux") - tableValue(data,p1,"flux")
    deltaTime = tableValue(data,p2,"time") - tableValue(data,p1,"time")
    return deltaFlux/deltaTime

def importData(data, filetype):
    if filetype == swiftxrt:
        data = Imports.swift_xrt(input_path)
    else:
        data = Imports.other() # eventually support other missions.
    return data
    

###############################################################
### FLARE FINDER
###############################################################

def FlareFinder(data):
    
    # Identify deviations as possible flares.
    possible_flares = []
    for index in data.index[data.time < 2000]:
        if potentialflare(data, index) == True:
            possible_flares.append(index)
    possible_flares = uniqueList(possible_flares)

    # Refine flares starts.
    index_start = []
    for start in possible_flares:
        index_start.append(findstart(data,start))
    index_start = uniqueList(index_start)

    # Look for flares peaks.
    index_peak  = []
    for start in index_start:
        index_peak.append(findpeak(data,start,index_start))
    index_peak = uniqueList(index_peak)

    # Look for end of flares.
    index_decay = []
    for peak in index_peak:
        index_decay.append(finddecay(data,peak,index_start))
    index_decay = uniqueList(index_decay)

    # Assign flare times in table.
    for start, decay in zip(index_start, index_decay):
        beginning = data.index >= start
        end = data.index <= decay
        data.loc[beginning & end, 'flare'] = True

    return index_start, index_peak, index_decay

def potentialflare(data,index):

    consecutive = []
    for i in range(8):
        try:
            consecutive.append(tableValue(data,index+i,"flux"))
        except:
            pass

    counter = 0
    for check in consecutive:
        if tableValue(data,index,"flux") + \
            ((tableValue(data,index,"flux_perr") * RISECONDITION)) < check:
            counter += 1
        if (tableValue(data,index,"flux") + \
            (tableValue(data,index,"flux_perr") * 3 * RISECONDITION) < check):
            counter += 6
    if counter >= 6:
        return True
    else:
        counter = 0
        return False   

def findstart(data,possiblestart):
    minval = min([tableValue(data,possiblestart+i,"flux") \
        for i in range(-30,1) if (possiblestart+i >= 0)])
    start = data[data['flux'] == minval].index.values[0]
    return start

def findpeak(data,start,index_start):
    j = 0
    while (max([tableValue(data,start+j+i,"flux") for i in range(0,4)]) > \
            tableValue(data,start,"flux")) and (start+j+4 not in index_start):
            j += 1
    maxval = max([tableValue(data,start+j,"flux") for i in range (-5,5)])
    peak = data[data['flux'] == maxval].index.values[0]
    return peak

def finddecay(data,peak,index_start):
    condition = 0
    peak = peak
    i = peak + 1
    while condition < DECAYCONDITION:
        i += 1
        # If all three conditions are met, add to counter.
        if slope(data, i, i+1) > slope(data, peak, i):
            if slope(data, i, i+1) > slope(data, i-1, i):
                if slope(data, peak, i) > slope(data, peak, i-1):
                    condition += 1
        # If too late, or it reaches a flare start, end immediately.
        if (tableValue(data,i,'time') > 2000) or (i in index_start):
            condition = DECAYCONDITION
    return i

###############################################################
### CONTINUUM FITTING
###############################################################

def modelfitter(data, model, inputpar):
    model = Model(model)
    odr = ODR(data, model, inputpar)
    odr.set_job(fit_type=0)
    output = odr.run()
    if output.info != 1:
        i = 1
        while output.info != 1 and i < 100:
            output = odr.restart()
            i += 1
    return output, output.beta

def FitContinuum(data):
    # Initialise defeault index parameters at logarithmic intervals.
    b1, b2, b3, b4, b5 = np.logspace(np.log10(data['time'].iloc[0] ) * 1.1, \
                                     np.log10(data['time'].iloc[-1]) * 0.9, \
                                     num=5)
    data = RealData(data.time, data.flux, data.time_perr, data.flux_perr)
    a1, a2, a3, a4, a5, a6 = 1, 1, 1, 1, 1, 1
    norm = 1e-7
    brk1_fit, brk1_param = modelfitter(data, Models.powerlaw_1break, \
        [a1,a2,b3,norm])
    brk2_fit, brk2_param = modelfitter(data, Models.powerlaw_2break, \
        [a1,a2,a3,b2,b4,norm])
    brk3_fit, brk3_param = modelfitter(data, Models.powerlaw_3break, \
        [a1,a2,a3,a4,b2,b3,b4,norm])
    brk4_fit, brk4_param = modelfitter(data, Models.powerlaw_4break, \
        [a1,a2,a3,a4,a5,b1,b2,b4,b5,norm])
    brk5_fit, brk5_param = modelfitter(data, Models.powerlaw_5break, \
        [a1,a2,a3,a4,a5,a6,b1,b2,b3,b4,b5,norm])

    models = [Models.powerlaw_1break, Models.powerlaw_2break, \
              Models.powerlaw_3break, Models.powerlaw_4break, \
              Models.powerlaw_5break]
    fits = [brk1_fit, brk2_fit, brk3_fit, brk4_fit, brk5_fit]
    pars = [brk1_param, brk2_param, brk3_param, brk4_param, brk5_param]

    return models, fits, pars

def BestContinuum(data,models,pars):

    evaluate_continuum = []
    for model, parameters in zip(models, pars):
        values = model(parameters, np.array(data.time))

        # Calculate Chi2, reduced Chi2 and AIC parameters.
        chisq = np.sum(((data.flux - values) ** 2)/(data.flux_perr**2))
        k = len(parameters)
        numdata = len(data.flux)
        red_chisq = chisq/(numdata - k)
        AIC = 2 * k + numdata * np.log(chisq)

        evaluate_continuum.append((model, parameters, chisq, red_chisq, AIC))

    # Find best fitting model as lowest AIC.
    gen = (x for x in evaluate_continuum)
    best_model, best_par, chisq, red_chisq, AIC = min(gen, key=itemgetter(4))

    statistics = [chisq, red_chisq, AIC]
    return best_model, best_par, statistics



def FitFlare(data, start, peak, stop, residuals):
    
    data_flare = RealData(data.time[start:stop], residuals[start:stop], \
                    data.time_perr[start:stop], data.flux_perr[start:stop])    
    flare_fit, flare_param = modelfitter(data_flare, Models.flare_gaussian, \
            [tableValue(data,peak,"flux"), tableValue(data,peak,"time"), \
            tableValue(data,stop,"time") - tableValue(data,start,"time")])
    residuals = residuals - Models.flare_gaussian(flare_param, np.array(data.time))
    return flare_param, residuals


###############################################################
### OUTPUT
###############################################################

def printResults(fl_start, parameters, stats):

    line = "//-============================================================================="

    print(line,"\nLightcurve and Flare Fitter | Version %s" % __version__)
    print("Contact: Adam Hennessy (ah724@leicester.ac.uk)")
    print(line,"\nInput data: %s" % input_path)

    print(line)

    N = len(parameters)
    print("%s flares found" % len(fl_start))
    print("%s powerlaw breaks found" % len(parameters[0:int(N/2)]))
    print("Chi-square:", round(stats[0],2))
    print("Reduced chi-square:", round(stats[1],2))

    print(line)

    print("LAFF complete.")

    print(line)

def printResults_verbose(data,fl_start, fl_peak, fl_end, powerlaw, parameters, stats):

    line = "//-============================================================================="

    print(line,"\nLightcurve and Flare Fitter | Version %s" % __version__)
    print("Contact: Adam Hennessy (ah724@leicester.ac.uk)")
    print(line,"\nInput data: %s" % input_path)

    print(line)

    print("[[ Flares (sec) ]]")
    print("Start\t\tPeak\t\tEnd\t\t")
    for start, peak, decay in zip(fl_start, fl_peak, fl_end):
        times = [round(tableValue(data,x,'time'),2) for x in (start,peak,decay)]
        print(*times, sep='\t\t')

    print(line)

    print("[[ Bestfit model - %s ]]" % powerlaw.__name__)

    N = len(parameters)

    print("Indices:\t\t", end=' ')
    print(*[round(x,2) for x in parameters[0:int(N/2)]], sep=', ')
    print("Breaks:\t\t\t", end=' ')
    print(*[round(x,2) for x in parameters[int(N/2):int(N-1)]])
    print("Normalisation:\t\t", end=' ')
    print(*[float("{:.2e}".format(parameters[-1]))])
    print("Chi-square:\t\t", round(stats[0],2))
    print("Reduced chi-square:\t", round(stats[1],2))
    print("AIC:\t\t\t", round(stats[2],2))
    print(line)

def plotResults(data, model, power_pars):

    constant_range = np.logspace(np.log10(data['time'].iloc[0]),
                                 np.log10(data['time'].iloc[-1]), num=2000)

    # Plot main lightcurve.
    plt.errorbar(data.time, data.flux, \
        xerr=[-data.time_nerr, data.time_perr], \
        yerr=[-data.flux_nerr, data.flux_perr], \
    marker='', linestyle='None', capsize=0)

    # Plot flare data.
    flaretime = data.flare == True
    plt.errorbar(data.time[flaretime], data.flux[flaretime],\
        xerr=[-data.time_nerr[flaretime], data.time_perr[flaretime]], \
        yerr=[-data.flux_nerr[flaretime], data.flux_perr[flaretime]], \
        marker='', linestyle='None', capsize=0, color='red')

    # Plot continuum fits.
    # for model, parameters in zip(models, pars):
    #     plt.plot(constant_range, model(parameters, constant_range), \
    #         linestyle='--', linewidth=0.75)

    plt.plot(constant_range, model)

    # # Plot powerlaw breaks.
    # N = len(power_pars[0])
    # for broken in power_pars[0][int(N/2):int(N-1)]:
    #     plt.axvline(broken, color='darkgrey', linestyle='--', linewidth=0.5)

    

    plt.loglog()
    plt.show()