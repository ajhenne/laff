from astropy.table import Table, vstack
import pandas as pd
import logging
import numpy as np

from .exceptions import NoFileError
from .runparameters import RiseCondition, DecayCondition
from .models import FlareObject

logger = logging.getLogger(__name__)


class Imports(object):

    def swift_xrt(filepath) -> pd.DataFrame:

        """
        Import a lightcurve from Swift-XRT.
        
        This function takes the standard .qdp lightcurve data available on the
        Swift online archive, and outputs the formatted table ready for LAFF.
        XRT lightcurves can sometimes contian upper limits, this funciton will
        ignore this data.
        
        [Parameters]
            filepath (str):
                Filepath to lightcurve data.
                
        [Returns]
            data (pandas dataframe)
                Formatted data table object.
        """

        logger.info("Starting Imports.swift_xrt import")
        qdptable = []
        i = 0
        # Haven't seen a .qdp table with more than 4, but check 6 anyway.
        while True:
            try:
                import_table = Table.read(filepath, format='ascii.qdp', table_id=i)

                if import_table.meta['comments'] in (['WTSLEW'], ['WT'], ['PC_incbad']):
                    logger.debug(f"Opening table index {i}")
                    qdptable.append(import_table)
                i += 1
            except:
                if i > 0:
                    logger.debug(f"Stopping at index {i} - no more tables to import")
                    break
                logger.critical(f"Unable to import data for given filepath")
                raise NoFileError(filepath)
            
        # Combine data into one table and format into pandas.
        logger.debug("Merging and formatting tables")
        data = vstack(qdptable).to_pandas()
        data = data.sort_values(by=['col1'])
        data = data.reset_index(drop=True)
        data = data.rename(columns={
            'col1': 'time', 'col1_perr': 'time_perr', 'col1_nerr': 'time_nerr',
            'col2': 'flux', 'col2_perr': 'flux_perr', 'col2_nerr': 'flux_nerr'})
        data['flare'] = False
        data['flare_ext'] = False

        logger.info("Data successfully imported")
        return data


def IdentifyDeviations(data) -> list:

    deviations = []

    for index in data.index[data.time < 2000]:

        # Check if 5 out of 8 consecutive data points rise above original.
        # if FlareMethods.cond_Consecutive(data.iloc[index:index+8]):
        #     logger.debug(f"Deviation found at {index}")

        if FlareMethods.cond_DoubleRise(data, index):
            logger.debug(f"Deviation found at {index}")
            # Look for a minima at the datapoints preceding - this is the start of deviation.
            new_index = FlareMethods.find_PrevMinima(data, index)

            if new_index != index:
                logger.debug(f"Deviation found at {index} actually starts at {new_index}")

            deviations.append(new_index)
    
    # Cleanup deviations - remove duplicates.
    deviations = [*set(deviations)]
    deviations.sort()

    logger.info(f"{len(deviations)} possible flares found")

    return deviations

def FindPeaks(data, startidx) -> list:

    peaks = []

    for idx, start in enumerate(startidx):
        logger.debug(f"Looking for peak after deviation start at {start}")
        try:
            nextidx = startidx[idx+1]
            endlook = (nextidx-1) if (nextidx in np.arange(start, start+30)) else (start+30)
        except:
            # Exception to catch the last startidx value - expected.
            endlook = (start+30)

        flarepeak = FlareMethods.find_NextMaxima(data, start, endlook)
        logger.debug(f"Peak for {start} found at {flarepeak}")

        if flarepeak == start:
            logger.error(f"Start {start} is equal to peak {flarepeak}")
        
        peaks.append(flarepeak)
    
    logger.info(f"Flare peaks found")

    return peaks

def FindEnds(data, startidx, peakidx) -> list:

    ends = []

    for start, peak in zip(startidx, peakidx):

        logger.debug(f"Looking for end of flare after peak {start}:{peak}")
        idx = peak
        condition = 0

        while condition < DecayCondition:
            if (data.iloc[idx]['time'] > 2000):
                logger.debug(f"- ending due to t > 2000s")
                break
            if (idx in startidx):
                logger.debug(f"- ending due to index in start index list")
                break
            idx += 1

            # Compute gradients and conditions.
            NextAlong = (data.iloc[idx+1]['flux'] - data.iloc[idx]['flux']) / (data.iloc[idx+1]['time'] - data.iloc[idx]['time'])
            PrevAlong = (data.iloc[idx]['flux'] - data.iloc[idx-1]['flux']) / (data.iloc[idx]['time'] - data.iloc[idx-1]['time'])
            Peak2Next = (data.iloc[idx]['flux'] - data.iloc[peak]['flux']) / (data.iloc[idx]['time'] - data.iloc[peak]['time'])
            Peak2Prev = (data.iloc[idx-1]['flux'] - data.iloc[peak]['flux']) / (data.iloc[idx-1]['time'] - data.iloc[peak]['time'])
            cond_1 = NextAlong > Peak2Next
            cond_2 = NextAlong > PrevAlong
            cond_3 = Peak2Next > Peak2Prev

            logger.debug(f"- conditions {cond_1} {cond_2} {cond_3}")
            if cond_1 and cond_2 and cond_3:
                condition += 1
            if cond_1 and cond_3 and not cond_2:
                condition -= 0.5
        else:
            logger.debug(f"- decay condition met")

        ends.append(idx)

    logger.info(f"Flare ends found")

    return ends

def VerifyFlares(data, FlareList) -> list:

    for i, flare in enumerate(FlareList):
        logger.debug(f"Performing checks for Flarelist[{i}]")
        conditions = flare.performChecks(data)

        if all(conditions):
            continue

        if conditions[0] == False and i != 0: # Can't alter a previous flare if it's the first!
            if flare.start == FlareList[i-1].decay:
                FlareList[i-1].decay = flare.decay

    # Filter out bad flares.
    print(len(FlareList))
    FlareList = [x for x in FlareList if x.keep == True]

    print(len(FlareList))

    return FlareList


class FlareMethods:

    def cond_Consecutive(data) -> bool:
        """Count and see if subsequent values are larger than some intial value."""
        data = np.array(data.flux) # Easier to handle as an np.array.
        initial = data[0] # First value
    
        counter = 0
        # If subsequent values are greater than initial, add to counter.
        for compare in data[1:]:
            if initial < compare:
                counter += 1

        if counter > 5:
            return True
        else:
            return False

    def cond_DoubleRise(data, idx) -> bool:

        # Guard - if doesn't increase, return false.
        if data.iloc[idx+1]['flux'] < data.iloc[idx]['flux']:
            return False

        fluxcompare = data.iloc[idx]['flux'] + (RiseCondition * data.iloc[idx]['flux_perr'])

        # If increases two in a row.
        if data.iloc[idx+2]['flux'] > data.iloc[idx+1]['flux']:
            if data.iloc[idx+2]['flux'] > fluxcompare:
                return True
        return False


    def find_PrevMinima(data, idx) -> int:
        """From deviation, look at previous 30 points for a minima - the flare start.."""

        MAXIDX = data.idxmax('index').time

        if idx < 30: # If close to beginning of data
            points = data.iloc[:35-idx]
        elif (idx+30) - MAXIDX >= 0: # If near the end of data range
            points = data.iloc[idx-30:]
        else:
            points = data.iloc[idx-30:idx+5]

        minima_index = data[data['flux'] == min(points.flux)].index.values[0]
        
        return minima_index

    def find_NextMaxima(data, idx, endidx) -> int:
        """From start, look for a preceding peak - the flare peak."""

        MAXIDX = data.idxmax('index').time

        if endidx > MAXIDX: # If near the end of data range
            points = data.iloc[idx:MAXIDX]
        else:
            points = data.iloc[idx:endidx]

        maxima_index = data[data['flux'] == max(points.flux)].index.values[0]

        return maxima_index