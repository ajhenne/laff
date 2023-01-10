
"""laff.laff: provides entry point main()."""

__version__ = "0.5.0"

import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from csv import writer
from os.path import exists
import scipy.integrate as integrate
from lmfit import Minimizer, Parameters, report_fit

# Import local laff modules.
from .models import Models
from .lcimport import Imports
from .classes import Flare

# Silence the double scalar warnings.
warnings.filterwarnings("ignore")

###############################################################
### USER ARGUMENTS
###############################################################

parser = argparse.ArgumentParser(description="Lightcurve and flare fitter for GRBs", prog='laff')
parser.add_argument('--version', action='version', version=__version__)
parser.add_argument('data', nargs=1, metavar='data_filepath', type=str, help='Path to the input datafile.')

parser.add_argument('-n', '--name', nargs=1, metavar='name', type=str, help='User specific name for the run, perhaps the name of the GRB.')
parser.add_argument('-o', '--output', nargs=1, metavar='output', type=str, help='Output file to write results to.')

parser.add_argument('-r', '--rise', nargs=1, metavar='rise_condition', type=float, help="Condition to alter the flare finding algorithm. A higher value makes it stricter (default: 2).")
parser.add_argument('-d', '--decay', nargs=1, metavar='decay_condition', type=float, help="Condition to alter the decay finding algorithm. A higher vaules makes it stricter (default: 4).")
parser.add_argument('-b', '--breaks', nargs=1, metavar='force_fit', help="Force a specific number of powerlaw breaks")

parser.add_argument('-s', '--show', action='store_true', help='Show a plot of the fitted lightcurve.')
parser.add_argument('-v', '--verbose', action='store_true', help="Produce more detailed text output in the terminal window.")
parser.add_argument('-q', '--quiet', action='store_true', help="Don't produce any terminal output.")

parser.add_argument('-m', '--mission', nargs=1, metavar='mission', help='Changed the input mission/filetype (default: Swift/XRT .qdp).')

args = parser.parse_args()

###############################################################
### ARGUMENTS
###############################################################

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
    DECAYCONDITION = 3

### Filetype.
swiftxrt  = ('swift', 'xrt', 'swiftxrt')     # Swift-XRT
swiftbulk = ('swiftbulk', 'bulk', 'xrtbulk') # Swift-XRT (bulk)
mission = swiftxrt # Default filetype.

if args.mission:
    if args.mission[0] in swiftxrt:
        mission = swiftxrt
    if args.mission[0] in swiftbulk:
        mission = swiftbulk
    else:
        raise ValueError("ERROR: filetype '%s' not supported." % args.mission[0])

### Force certain fits.
if args.breaks:
    # Force number of powerlaws.
    try:
        force = int(args.force)
    except:
        raise ValueError("--force breaks argument invalid: must be integer in range 1 to 5.")
    if not 1 <= force <= 5:
        raise ValueError("--force breaks argument invalid: must be in range 1 to 5.")
else:
    force = False

###############################################################
### MISC SETUP
###############################################################  

### Flare function shape. Default to gaussian.
flareFunction = Models.flareFred

# Decorative line.
line = "//-===============-"

###############################################################
### MAIN
###############################################################

def main():

    ###### [ IMPORT DATA ] ######
    try:
        data = importData(input_path, mission)
    except:
        raise ValueError('Could not find valid file at %s.' % input_path)

    ###### [ FIND FLARES ] ######

    # Flare finding function.
    fl_start, fl_peak, fl_end = FlareFinder(data)

    # List to contain Flare objects.
    FlareList = []

    # Assign each flare to Flare class.
    for start, peak, decay in zip(fl_start, fl_peak, fl_end):        
        FlareList.append(Flare(start, peak, decay))

    # Assign flare times in table.
    for flare in FlareList:
        start, peak, decay = flare.returnTimes()

        beg = data.index >= start
        end = data.index <= decay
        data.loc[beg & end, 'flare'] = True

        beg = data.index > start
        end = data.index < decay

        data.loc[beg & end, 'flare_ext'] = True

    ###### [ FIT CONTINUUM ] ######

    # Ignore the flare data.
    data_continuum = data[data['flare_ext'] == False]

    # Assign a weighting function.
    sigma = calculateSigma(data, continuum=True)

    # Setup parameters object.
    params = Parameters()
    mintime = tableValue(data, 0,'time')
    maxtime = tableValue(data,-1,'time')

    # Fit powerlaws of breaks 0 through 5.
    continuum_fits = fitContinuum(data_continuum, sigma, mintime, maxtime)

    # Calculate set of parameters with lowest AIC as the best fit solution.
    gen = (x for x in continuum_fits)
    ContinuumParameters = min(gen, key=itemgetter(1))[0]

    # Calculate residuals - perfect case is just flares leftover.
    continuum_residuals = data.flux - Models.powerlaw(ContinuumParameters, np.array(data.time))

    ###### [ FIT FLARES ] ######

    sigma = calculateSigma(data, reset=True)

    FlareParameters, flare_residuals = fitFlares(data, continuum_residuals, FlareList)
    # Is flare_residuals needed?

    constant_range = np.logspace(np.log10(data['time'].iloc[0]),
                                 np.log10(data['time'].iloc[-1]), num=2000)

    # Sum all model components.
    finalModel = Models.powerlaw(ContinuumParameters, constant_range)
    for flare in FlareParameters:
        finalModel += flareFunction(flare, constant_range)


    ### CALCULATE FLUENCES

    # func_powerlaw = lambda x: powerlaw(parameters, x) # define powerlaw function

    # def createFlare(flarefits): # define flare functions
    #     return lambda x: Models.flare_gaussian(flarefits, x)
    # func_flare = []
    # for flare in flareParams:
    #     func_flare.append(createFlare(flare))

    # fluence_full = calculateFluence(func_powerlaw, func_flare, tableValue(data,0,"time"), tableValue(data,-1,"time"))

    # fluence_flare = []
    # for beg, stop, par in zip(fl_start, fl_end, flareParams):
    #     fluence_flare.append(calculateFluence(func_powerlaw, func_flare, (par[1]-10*par[2]), (par[1]+10*par[2])))

    # # Neaten up fluences.
    # flarecount = len(fluence_flare)
    # fluences = [fluence_full[0], *[fluence_flare[i][i+1] for i in range(flarecount)], fluence_full[0]+sum([fluence_flare[i][i+1] for i in range(flarecount)])]

    ###### [ FINISHING UP ] ######

    # Print results to terminal.
    if not args.quiet:
        if not args.verbose:
            printResults(ContinuumParameters, FlareParameters)
            # also print stats
        else:
            printResults_verbose(data, ContinuumParameters, FlareList)
            # also print stats and fluences

    # # Show the lightcurve plots.
    if args.show:
        plotResults(data, finalModel, [ContinuumParameters, FlareParameters])

    # # Write output to table.
    # if args.output:
    #     produceOutput(data, fl_start, fl_peak, fl_end, fluences)

    print("//- LAFF run finished successfully.")

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

def importData(data, mission):
    if mission == swiftxrt:
        data = Imports.swift_xrt(input_path)
    elif mission == swiftbulk:
        data = Imports.swift_bulk(input_path)
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

    return index_start, index_peak, index_decay

def potentialflare(data,index):
    # Store ranges of 8 consecutive values at a time.
    consecutive = []
    for i in range(8):
        try:
            consecutive.append(tableValue(data,index+i,"flux"))
        except:
            pass
    # Check conditions.
    counter = 0
    for check in consecutive:
        if tableValue(data,index,"flux") + \
            ((tableValue(data,index,"flux_perr") * RISECONDITION)) < check:
            counter += 1
        if (tableValue(data,index,"flux") + \
            (tableValue(data,index,"flux_perr") * 3 * RISECONDITION) < check):
            counter += 3
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
    """Look for the peak by cycling through data and looking for a maximum.
    
    Start at point i+1, while he points in range n->n+4 have a maximum greater than the start flux
    and n+1 hasn't reached the start of the next flare, keep cycling through i + 1. Once one of these
    conditions are met we can end the loop.

    At this point we're definitely past the peak, so we look through all values start -> i and look for
    the maximum flux value: our peak, and return this.

    """
    i = 1

    cond1 = max([tableValue(data,start+i+j,"flux") for j in range(0,4)]) > tableValue(data,start,"flux")
    cond2 = start + i + 1 not in index_start
    cond3 = start + i + 4 in data.index

    while cond1 and cond2 and cond3:
        i += 1
        # Need to re-evaluate the conditions - better way surely?
        cond1 = max([tableValue(data,start+i+j,"flux") for j in range(0,4)]) > tableValue(data,start,"flux")
        cond2 = start + i + 1 not in index_start
        cond3 = start + i + 4 in data.index

    max_value = max([tableValue(data,start+k,"flux") for k in range(0, i+1)])

    peak = data[data['flux'] == max_value].index.values[0]

    return peak


def finddecay(data,peak,index_start):
    condition = 0
    i = peak + 1
    while condition < DECAYCONDITION: #default: 3
        # If too late, or it reaches a flare start, end immediately.
        if (tableValue(data,i,'time') > 2000) or (i in index_start):
            return i - 2

        # If all three conditions are met, add to counter.

        con_1 = slope(data, i, i+1) > slope(data, peak, i)      # gradient from i->i+1 is shallower than peak->i (shallower than line to peak)
        con_2 = slope(data, i, i+1) > slope(data, i-1, i)       # gradient from i->i+1 is shallower than i-1->i (shallower point to point)
        con_3 = slope(data, peak, i) > slope(data, peak, i-1)   # gradient from peak->i is shallower than peak->i-1 (shallower peak to i than peak to previous i)

        if con_1 and con_2 and con_3:
            condition += 1
        elif con_1 and not con_2 and con_3:
            condition -= 0.5
        i += 1
    return i

###############################################################
### CONTINUUM FITTING
###############################################################

def fitContinuum(data, weights, mintime, maxtime):
    """Perform continuum fits for breaks 0 through 5."""
    continuum_fits = []

    for N in range(6):
        # Initialise the parameters.
        params = initParams(N, mintime, maxtime)
        
        # Perform the fit.
        fitter = Minimizer(Models.powerlaw, params, fcn_args=(np.array(data.time), np.array(data.flux), weights))
        results = fitter.least_squares()

        # Calculate fit statistics - (relative) AIC
        k = len(results.params) - 1
        numdata = len(data.flux)
        chisq = np.sum((results.residual ** 2)/(data.flux_perr ** 2))
        AIC = 2 * k + (numdata * np.log(chisq))

        # Store parameters.
        continuum_fits.append([results.params, AIC])
    
    return continuum_fits

def calculateSigma(data, continuum=False, reset=False):
    # Assign a weight based on uncertainty - less sigma is a more precise datapoint.
    sigma = np.array(data.flux_perr)

    if reset:
        return sigma

    # Assign less sigma to points at the start and end of flare.
    sigma[np.where(data.flare == True)] *= 1e-5

    # Remove flare data if required.
    sigma_red = sigma[data['flare_ext'] == False]

    return sigma_red if continuum else sigma

def initParams(number_breaks, mintime, maxtime):
    """Initiliase parameters based on the number of powerlaw breaks."""

    params = None
    params = Parameters()
    # Create break number parameter.
    params.add('num_breaks', value=number_breaks, vary=False)
    # Create index parameters.
    init_params_indices(number_breaks, params)
    # Create break parameters.
    init_params_breaks(number_breaks, mintime, maxtime, params)
    # Create normalisation parameters.
    params.add('normal', value=1)
    return params

def init_params_indices(number_breaks, params):
    """Initialise the index parameters."""
    indices = []
    # Setup parameter names.
    for i in range(number_breaks+1):
        indices.append('index%s' % (i+1))
    # Initilise with standard variable ranges.
    for index in indices:
        params.add(index, value=1, min=0, max=5)
    return params

def init_params_breaks(number_breaks, mintime, maxtime, params):
    """Initialise the break parameters."""
    powerlawbreaks = []
    # Initial value estimates.
    initial_breaks = [mintime, *np.logspace(np.log10(mintime * 1.1), np.log10(maxtime) * 0.9, num=number_breaks), maxtime]
    # Setup parameter names.
    for i in range(number_breaks):
        powerlawbreaks.append('break%s' % (i+1))
    # Assign initial parameter values and bounds.
    for i in range(len(powerlawbreaks)):
        # params.add(powerlawbreaks[i], value=initial_breaks[i+1], \
        #         min=initial_breaks[i], max=initial_breaks[i+2])
        params.add(powerlawbreaks[i], value=initial_breaks[i+1], \
            min = initial_breaks[0], max=initial_breaks[-1])
    return params

###############################################################
### FLARE FITTING
###############################################################

def fitFlares(data, residuals, flarelist):

        flare_parameters = []
        flare_residuals = 0

        for flare in flarelist:
            start, peak, end = flare.returnTimes()

            params = initFlare(data, start, peak, end) # Initialise flare parameters.

            # Fit to the residuals only in the highlighted flare time.
            time = np.array(data.time[start:end])
            y = np.array(residuals[start:end])

            # Run fitting routine.
            fitter = Minimizer(flareFunction, params, fcn_args=(time, y))
            results = fitter.least_squares()
            fittedFlare = flareFunction(results.params, np.array(data.time))

            # Store parameters and add residuals.
            flare_parameters.append(results.params)
            flare.setParameters(results.params)
            residuals -= flareFunction(results.params, np.array(data.time))
            flare_residuals += fittedFlare

        return flare_parameters, residuals

def initFlare(data, start, peak, end):

    # Calculate initial input values.
    time_start = tableValue(data,start,"time")
    time_end = tableValue(data,end,"time")

    height = tableValue(data,peak,"flux")
    centre = tableValue(data,peak,"time")
    width  = time_end - time_start

    # Setup parameters according to flare model.
    if flareFunction == Models.flareGaussian:
        params = Parameters()
        params.add('height', value=height, min=0)
        params.add('centre', value=centre)
        params.add('width',  value=width, min=0, max=2*width)
        return params
    elif flareFunction == Models.flareFred:
        params = Parameters()
        params.add('peak', value=centre, vary=False)
        params.add('rise', value=width*0.3, min=0, max=width*3)
        params.add('decay', value=width*0.7, min=0, max=width*4)
        params.add('amp', value=height, min=0)
        params.add('sharp', value=1, min=0, max=15)
        return params
    elif flareFunction == Models.flareFred_archive:
        params = Parameters()
        params.add('rise',  value=0.25*width, min=0.1*width, max=1*width)
        params.add('decay', value=0.75*width, min=0.1*width, max=2*width)
        params.add('time',  value=centre, min=time_start, max=time_end)
        params.add('amp',   value=height)
        return params
    else:
        raise ValueError('an unacceptable flare model appears to have been selected.')

def calculateFluence(powerlaw_func, flare_funclist, start, stop):
    comp_power = integrate.quad(powerlaw_func, start, stop)[0] # Calculate component from powerlaw.
    comp_flare = []
    for anotherone in flare_funclist:
        comp_flare.append(integrate.quad(anotherone, start, stop)[0]) # Calculate component(s) from flares.
    total = comp_power + np.sum(comp_flare) # Total.
    return comp_power, *comp_flare, total

###############################################################
### OUTPUT
###############################################################

def printResults(continuumParams, FlareList):

    print(line,"\nLightcurve and Flare Fitter | Version %s" % __version__)
    print("Contact: Adam Hennessy (ah724@leicester.ac.uk)")
    print(line,"\nInput data: %s" % input_path)

    print(line)

    N = len(continuumParams)
    print("%s flares found" % len(FlareList))
    print("%s powerlaw breaks found" % continuumParams['num_breaks'].value)
    # print("Chi-square:", round(stats[0],2))
    # print("Reduced chi-square:", round(stats[1],2))

    print(line)
    print("LAFF complete.")
    print(line)


def printResults_verbose(data, continuumParams, FlareList):

    print(line,"\nLightcurve and Flare Fitter | Version %s" % __version__)
    print("Contact: Adam Hennessy (ah724@leicester.ac.uk)")
    print(line,"\nInput data: %s" % input_path)

    print(line)

    print("[[ Flares (sec) -- %s found ]] \t\t\t \t\t[[ Fitted Parameters ]]" % len(FlareList))
    print("Start\t\tPeak\t\tEnd\t\t \t\tT_Centre\tRise\t\tDecay\t\tAmplitude\tSharpness")

    # ROUND THE PARAMETERS AND THEN ADD TITLE ROW
    for flare in FlareList:
        times = [round(tableValue(data,x,'time'),2) for x in (flare.start, flare.peak, flare.end)]
        params = flare.returnParameters(pretty=True)

        print(*times,' ',*params, sep='\t\t')

    # for start, peak, decay in zip(start, peak, decay):
    #     times = [round(tableValue(data,x,'time'),2) for x in (start,peak,decay)
    #     print(*times,"rg", sep='\t\t')

    print(line)

    print("[[ Best fit broken powerlaw -- %s breaks ]]" % continuumParams['num_breaks'].value)

    N = len(continuumParams)
    parameters = list(continuumParams)

    print("Indices \t\t", end=' ')
    print(*[round(continuumParams[x].value,2) for x in parameters[1:int((N/2)+1)]], sep=', ')

    print("Breaks (sec)\t\t", end=' ')
    print(*[round(continuumParams[x].value,2) for x in parameters[int((N/2)+1):int(N-1)]], sep=', ')

    print("Normalisation \t\t", end=' ')
    print(*[float("{:.3e}".format(continuumParams[parameters[-1]].value))])

    # print(*[round(x,2) for x in parameters[0:int(N/2)]].value, sep=', ')
    # print("Breaks:\t\t\t", end=' ')
    # print(*[round(x,2) for x in parameters[int(N/2):int(N-1)]])
    # print("Normalisation:\t\t", end=' ')
    # print(*[float("{:.1}".format(parameters[-1]))])
    # print("Chi-square:\t\t", round(stats[0],2))
    # print("Reduced chi-square:\t", round(stats[1],2))
    # print("AIC:\t\t\t", round(stats[2],2))
    # print(line)

    # # Print fluences.
    # print("[[ Fluences ]] ")
    # print("Continuum:\t\t", "{:.4e}".format(fluences[0]))

    # for count, fl in enumerate(fluences[1:-1], start=1):
    #     print(f"Flare {count}:\t\t","{:.4e}".format(fl))

    # print("Total model:\t\t","{:.4e}".format(fluences[-1]))

    print(line)


def plotResults(data, finalModel, finalParameters):

    continuumParams, flareParams = finalParameters

    constant_range = np.logspace(np.log10(data['time'].iloc[0]),
                                 np.log10(data['time'].iloc[-1]), num=2000)

    plt.figure(figsize=(10,7))

    # Plot lightcurve data.
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

    # Plot fitted model.
    plt.plot(constant_range, finalModel)

    # Plot powerlaw breaks.
    N = len(continuumParams)
    for breakpoint in list(continuumParams)[int((N/2+1)):int(N-1)]:
        plt.axvline(continuumParams[breakpoint].value, color='darkgrey', linestyle='--', linewidth=0.5)

    # Plot flare model components.
    for flare in flareParams:
        plt.plot(constant_range, flareFunction(flare, constant_range), \
            linestyle='--', linewidth=0.5)

    plt.plot(constant_range, Models.powerlaw(continuumParams, constant_range), \
            linestyle='-', linewidth=0.75)

    y_bottom = data.flux.min()*0.5
    y_top = data.flux.max()*3

    plt.ylim(y_bottom, y_top)

    plt.loglog()
    # plt.savefig(f'results/{input_path[0:-4]}.png')
    plt.show()


def produceOutput(data, start, peak, end, fluences):

    # If file doesn't exist yet, add header row.
    headerList = ['name','flare_num','start','peak','end','fluence','fluence_err']
    if not exists(output_path):
        with open(output_path, 'w') as file:
            object = writer(file)
            object.writerow(headerList)
            file.close()

    # Default name if not user-specified.
    if args.name:
        out_name = args.name[0]
    else:
        out_name = 'laff_run'

    i = 1

    # Write flare times and fluences to table.
    for strt, peek, eend in zip(start, peak, end):
        out_flno = i
        out_srt, out_peak, out_end = [tableValue(data,x,'time') for x in (strt, peek, eend)]
        outputline = out_name, out_flno, out_srt, out_peak, out_end, fluences[i]

        with open(output_path, 'a') as file:
            object = writer(file)
            object.writerow(outputline)
            file.close()
        i += 1