################################################################################
# [ LAFF.LCIMPORT ]
################################################################################
# A collection of functions to import GRB lightcurves from various missions and
# file formats.
################################################################################
# Currently supported:
# > swift-xrt.qdp files from Swift Online Archive
################################################################################

import pandas as pd
from astropy.table import Table, vstack

def import_swift_xrt(filepath) -> pd.DataFrame:
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

    qdptable = []
    i = 0

    while i < 10:
        try:
            import_table = Table.read(filepath, format='ascii.qdp', table_id=i)
            if import_table.meta['comments'] in (['WTSLEW'], ['WT'], ['PC_incbad']):
                qdptable.append(import_table)
            i += 1
        except:
            if i > 0: break # No more tables.