################################################################################
# [ LAFF.LCIMPORT.SWIFT ]
################################################################################
# A collection of functions for the Swift mission.
################################################################################

import pandas as pd
from astropy.table import Table, vstack

def lc_swift_online_archive(filepath) -> pd.DataFrame:
    """
    Import a lightcurve from the Swift-XRT online archive page.
    
    This function takes the standard .qdp lightcurve data available on the
    Swift online archive GRB pages, and outputs the formatted table ready 
    for LAFF. XRT lightcurves can sometimes contian upper limits, this 
    function will ignore this data.
    
    [Parameters]
        filepath (str):
            Filepath to lightcurve data.
            
    [Returns]
        data (pandas dataframe)
            Formatted data table object.
    """

    qdptable = []
    i = 0

    # Import tables from qdp.
    while i < 10:
        try:
            import_table = Table.read(filepath, format='ascii.qdp', table_id=i)
            if import_table.meta['comments'] in (['WTSLEW'], ['WT'], ['PC_incbad']):
                qdptable.append(import_table)
            i += 1
        except FileNotFoundError:
            raise FileNotFoundError(f"No file found at '{filepath}'.")
        except IndexError:
            break

    # Prepare data to pandas frame.
    data = vstack(qdptable).to_pandas()
    data = data.sort_values(by=['col1'])
    data = data.reset_index(drop=True)

    return data