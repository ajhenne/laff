"""!!!Not really usable yet, more for personal use.!!!"""


import swifttools.ukssdc.data.GRB as udg
import swifttools.ukssdc.query as uq
import pandas as pd
import argparse
from astropy.table import Table

"""Wrapper around swifttools module to bulk download GRB light curves in a LAFF suitable format."""


parser = argparse.ArgumentParser(description='Bulk download Swift XRT GRB light curves.', prog='laff-bulk')

parser.add_argument('output', nargs=1, metavar='output_directory', type=str, help='Output directory path for downloaded files.')

args = parser.parse_args()

output_path = args.output[0]

# Setup query object.
q = uq.GRBQuery(cat='SDC_GRB')
q.addCol('Name')
q.addCol('TriggerNumber')

# Add filters.
filter1 = ('Name', '>', 'GRB 190912A')
filter2 = ('BAT_T90', '>', '2')
q.addFilter(filter1)
q.addFilter(filter2)

# Search.
q.submit()
grb_names, grb_triggers = list(q.results['Name']), list(q.results['TriggerNumber'])

# Store lightcurves.
lightcurves = udg.getLightCurves(targetID=grb_triggers[0:5],
                    incbad='yes',
                    saveData=False,
                    returnData = True,
                    silent=False)

# Organise observation modes and save to file.
for name, trigID in zip(grb_names, lightcurves):

    try:    # Look for slew mode data.
        lightcurves[trigID]['WTSLEW_incbad']
        sl = True
    except:
        sl = False

    try:    # Look for WT mode data.
        lightcurves[trigID]['WT_incbad']
        wt = True
    except:
        wt = False

    try:    # Look for PC mode data.
        lightcurves[trigID]['PC_incbad']
        pc = True
    except:
        pc = False

    columns = ['Time','TimePos','TimeNeg','Rate','RatePos','RateNeg']
    table = []
    if sl:
        table.append(lightcurves[trigID]['WTSLEW_incbad'][columns])
    if wt:
        table.append(lightcurves[trigID]['WT_incbad'][columns])
    if pc:
        table.append(lightcurves[trigID]['PC_incbad'][columns])

    if sl or wt or pc:
        fulltable = pd.concat([mode for mode in table])
        fulltable = fulltable.sort_values(by=['Time'])
        fulltable = fulltable.reset_index(drop=True)

        grb_lc = Table.from_pandas(fulltable)
        filepath = output_path+'/'+name.replace(' ','')+'.qdp'
        print(filepath)
        grb_lc.write(filepath, err_specs={'terr': [1, 2]})

def getSwiftLC():

    """

    This function takes the standard .qdp lightcurve data available on the 
    Swift online archive, and outputs the formatted table ready for LAFF.
    XRT Lightcurves can sometimes contain upper limits, this function also
    excludes importing this data.

    [Parameters]
        filepath (str):
            Filepath to lightcurve data.

    [Returns]
        data (pandas table): 
            Formatting data table object ready for LAFF.
    """

    return