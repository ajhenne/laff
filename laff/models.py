import numpy as np

"""laff.models: lightcurve and flare models for the laff package."""

class Continuum(object):

    def powerlaw(params, x, data=None, sigma=None):

        breaknum = params['breaknum']
        normal = params['normal']

        # Initialise all possible parameters such they exist.
        # Required as all indices wouldn't exist if called by low breaknum.
        index1, index2, index3, index4, index5, index6 = 0, 0, 0, 0, 0, 0
        break1, break2, break3, break4, break5         = 0, 0, 0, 0, 0

        if breaknum >= 0:
            index1 = params['index1']
        if breaknum >= 1:
            index2 = params['index2']
            break1 = params['break1']
        if breaknum >= 2:
            index3 = params['index3']
            break2 = params['break2']
        if breaknum >= 3:
            index4 = params['index4']
            break3 = params['break3']
        if breaknum >= 4:
            index5 = params['index5']
            break4 = params['break4']
        if breaknum >= 5:
            index6 = params['index6']
            break5 = params['break5']

        # Boundary conditions for powerlaw breaks.
        cond = [x > break1, x > break2, x > break4, x > break4, x > break5]

        # Define functions according to breaknum and conditions.
        if breaknum >= 0:
            model = normal * (x**(-index1))
        if breaknum >= 1:
            model[np.where(cond[0])] = normal * (x[np.where(cond[0])]**(-index2)) * (break1**(-index1+index2))
        if breaknum >= 2:
            model[np.where(cond[1])] = normal * (x[np.where(cond[1])]**(-index3)) * (break1**(-index1+index2)) * (break2**(-index2+index3))
        if breaknum >= 3:
            model[np.where(cond[2])] = normal * (x[np.where(cond[2])]**(-index4)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4))
        if breaknum >= 4:
            model[np.where(cond[3])] = normal * (x[np.where(cond[3])]**(-index5)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4)) * (break4**(-index4+index5))
        if breaknum >= 5:
            model[np.where(cond[4])] = normal * (x[np.where(cond[4])]**(-index6)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4)) * (break4**(-index4+index5)) * (break5**(-index5+index6))

        # Return just model if no data given, return weighted function if data provided.
        return model if data is None else (data - model)/sigma

class ModelFlare(object):

    def gaussian(params, x, data=None, sigma=None):
        """Gaussian flare curve."""

        height = params['height']
        centre = params['centre']
        width  = params['width']

        model = height * np.exp(-((x-centre)**2)/(2*(width**2)))

        return model if data is None else (model - data)/sigma


    def fred(params, x, data=None, sigma=None):
        """Fast-rise exponential decay (FRED) curve."""

        t_peak = params['peak']
        rise   = params['rise']
        decay  = params['decay']
        sharp  = params['sharp']
        amp    = params['amp']

        cond = x >= t_peak

        model                 = amp * np.exp(-((abs(x-t_peak)/rise)**sharp))
        model[np.where(cond)] = amp * np.exp(-((abs(x[np.where(cond)]-t_peak)/decay)**sharp))

        for idx, number in enumerate(model):
            if np.isinf(number):
                raise ValueError('Infinite value calculated in function fred.')
            if np.isnan(number):
                raise ValueError('NaN value calculcated in function fred.')

class FlareObject:
    """Flare class."""

    def __init__(self, start, peak, end):
        self.start = start
        self.peak = peak
        self.decay = end

        self.keep = True

    def __str__(self):
        return f"Flare indices {self.start}/{self.peak}/{self.decay}"

    def returnPar(self):
        return [self.start, self.peak, self.decay]

    def performChecks(self, data):
        self.data = data

        check1 = self.checkFluxIncrease(data)
        check2 = self.checkAverageNoise(data)
        check3 = True

        AllChecks = [check1, check2, check3]
        self.keep = True if all(AllChecks) else False 
        print(self.keep)

        return AllChecks

    def checkFluxIncrease(self, data):
        comp_start = data.iloc[self.start]['flux'] + data.iloc[self.start]['flux_perr']
        comp_peak  = data.iloc[self.peak]['flux'] + data.iloc[self.peak]['flux_nerr']

        Check = False if comp_start > comp_peak else True
        return Check

    def checkAverageNoise(self, data):

        """
        Take the average noise (error) across all poitns in the flare

        If the vertical height (peak flux - start flux) is smaller than the noise
        Then this isn't significant enough
        """
        avgnoise = np.average(data.iloc[self.start:self.decay]['flux_perr']) - np.average(data.iloc[self.start:self.decay]['flux_nerr'])
        fluxchange = data.iloc[self.peak]['flux'] - data.iloc[self.start]['flux']

        Check = False if fluxchange < avgnoise * 2 else True
        return Check