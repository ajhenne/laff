import pandas as pd
from astropy.table import Table, vstack

def lc_import(filepath, format="swift"):
    """
    Import a lightcurve to a format ready for LAFF.
    
    This function takes the filepath of a GRB lightcurve and will convert it into a
    format ready to use by LAFF. If this is not used, the user input is assumed to
    be structured as X.
    
    [Parameters]
        filepath (str):
            Filepath to lightcurve data.
            
    [Returns]
        data (pandas dataframe)
            Formatted pandas datatable.
    """

    if format == "swift":
        # data = _swift_online_archive(filepath)
        data = 1
    
    else:
        return "error"
        
    return data


def _swift_online_archive(data_filepath):
    qdptable = []
    i = 0

    # Import tables from qdp.
    while i < 10:
        try:
            import_table = Table.read(data_filepath, format='ascii.qdp', table_id=i)
            if import_table.meta['comments'] in (['WTSLEW'], ['WT'], ['PC_incbad']):
                qdptable.append(import_table)
            i += 1
        except FileNotFoundError:
            raise FileNotFoundError(f"No file found at '{data_filepath}'.")
        except IndexError:
            break

    # Prepare data to pandas frame.
    data = vstack(qdptable).to_pandas()
    data = data.sort_values(by=['col1'])
    data = data.reset_index(drop=True)
    data.columns = ['time', 'time_perr', 'time_nerr', 'flux', 'flux_perr', 'flux_nerr']

    data['flare'] = False

    return data