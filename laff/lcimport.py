from astropy.table import Table, vstack
import pandas as pd


"""laff.lcimport: lightcurve importing module within the laff package."""

class Imports(object):

    def swift_xrt(filepath):

        """
        Import a lightcurve from Swift-XRT.

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
        acceptable_modes = (['WTSLEW'], ['WT'], ['PC_incbad'])

        qdptable = []

        # I haven't seen more than 4 tables, so 6 should be fine.
        for i in range(6):
            try:
                import_table = Table.read(filepath, format='ascii.qdp', table_id=i)
            except:
                continue

            # Exclude data tables that aren't in the correct observing mode.
            # Check first for _incbad mode (usually default).
            if import_table.meta['comments'] in acceptable_modes:
                qdptable.append(import_table)
            i += 1

        # Combine data into one table, format
        data = vstack(qdptable).to_pandas()
        data = data.sort_values(by=['col1'])
        data = data.reset_index(drop=True)
        data = data.rename(columns={
            'col1': 'time', 'col1_perr': 'time_perr', 'col1_nerr': 'time_nerr',
            'col2': 'flux', 'col2_perr': 'flux_perr', 'col2_nerr': 'flux_nerr'})
        data['flare'] = False

        return data

    def swift_bulk(filepath):

        """
        Import a lightcurve from Swift-XRT downloaded through laff.bulklc.

        This function imports standard .qdp lightcurve data from the Swift XRT
        telescope. This function is likely temporary and for personal use and
        the main function will fully support various formats of LC. The bulklc
        module imports data into one large table removing the various
        observation modes, hence the need for a slightly different import
        method.

        [Parameters]
            filepath (str):
                Filepath to lightcurve data.

        [Returns]
            data (pandas table): 
                Formatting data table object ready for LAFF.
        """

        import_table = Table.read(filepath, format='ascii.qdp', table_id=0)
        data = vstack(import_table).to_pandas()
        data = data.sort_values(by=['col1'])
        data = data.reset_index(drop=True)
        data = data.rename(columns={
            'col1': 'time', 'col1_perr': 'time_perr', 'col1_nerr': 'time_nerr',
            'col2': 'flux', 'col2_perr': 'flux_perr', 'col2_nerr': 'flux_nerr'})
        data['flare'] = False

        data = None if len(data) <= 3 else data # Seems to break if data <= 3.

        return data

    def other():
        """Temporary object, eventually will contain other lc formats."""
        return None
    
    pass
