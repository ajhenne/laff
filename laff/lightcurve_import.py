import pandas as pd
from astropy.table import Table, vstack
import logging

logger = logging.getLogger('laff')

def lcimport(filepath, format="online_archive"):
    """
    Import a lightcurve to a format ready for LAFF.
    
    This function takes the filepath of a GRB lightcurve and will convert it into a
    format ready to use by LAFF. If this is not used, the user input is assumed to
    be structured as X.
    
    [Parameters]
        filepath (str):
            Filepath to lightcurve data.
        format (str):
            Format of incoming file.
                - 'online_archive': .qdp from Swift online archive containing XRT data
                - 'python_query': .qdp file generated through swifttools python modules containing BAT and XRT data.
            
    [Returns]
        data (pandas dataframe)
            Formatted pandas datatable.
    """

    if format == "online_archive":
        data = _swift_online_archive(filepath)

    elif format == "python_query":
        data = _swift_python_query(filepath)

    elif format == "bat":
        data = _bat_data(filepath)
    
    else:
        raise ValueError("Invalid format parameter.")

    logger.info('Data import successful.')
    logger.debug(f'Data filepath: {filepath}')
        
    return data


def _swift_online_archive(data_filepath, incbad=False):
    """Data obtained directly from the Swift Online Archives."""
    qdptable = []
    i = 0

    allowed_modes = ['WTSLEW', 'WT', 'WT_incbad', 'PC_incbad',
                    'batSNR5flux', 'xrtwtslewflux', 'xrtwtflux', 'xrtpcflux_incbad']
    allowed_modes += 'xrtpcflux_nosys_incbad' if incbad == True else ''
    allowed_modes = [[item] for item in allowed_modes]

    # Import tables from qdp.
    while i < 10:
        try:
            import_table = Table.read(data_filepath, format='ascii.qdp', table_id=i)
            if import_table.meta['comments'] in allowed_modes:
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

    return data

def _swift_python_query(data_filepath):
    """Data obtained through the Swift python module."""

    qdptable = []
    i = 0

    while i < 10:
        try:
            import_table = Table.read(data_filepath, format='ascii.qdp', table_id=i)
            qdptable.append(import_table)
            logger.debug(f'Appending i = {i}')
            i += 1
        except FileNotFoundError:
            raise FileNotFoundError(f"No file found at {data_filepath}.")
        except IndexError:
            break

    data = vstack(qdptable).to_pandas()
    data = data.sort_values(by=['col1'])
    data = data.reset_index(drop=True)
    data.columns = ['time', 'time_perr', 'time_nerr', 'flux', 'flux_perr', 'flux_nerr']

    return data

def _bat_data(data_filepath):
    """Data obtained from lc observation files for BAT."""

    df = pd.read_csv(data_filepath)

    data = pd.DataFrame({
        'time': df['TIME'],
        'time_perr': 0,
        'time_nerr': 0,
        'flux': df['RATE'],
        'flux_perr': df['ERROR'],
        'flux_nerr': df['ERROR']
    })

    data = data.sort_values(by=['time'])
    data = data.reset_index(drop=True)

    return data