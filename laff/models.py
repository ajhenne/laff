import numpy as np

"""laff.models: models module within the laff package."""

class Models(object):

    def powerlaw(params, x, data=None, sigma=None):
        num_breaks = params['num_breaks']
        normal = params['normal']

        # Initialise all possible parameters such that they exist.
        index1, index2, index3, index4, index5, index6 = 0,0,0,0,0,0
        break1, break2, break3, break4, break5 = 0,0,0,0,0
        if num_breaks >= 0:
            index1 = params['index1']
        if num_breaks >= 1:
            index2 = params['index2']
            break1 = params['break1']
        if num_breaks >= 2:
            index3 = params['index3']
            break2 = params['break2']
        if num_breaks >= 3:
            index4 = params['index4']
            break3 = params['break3']
        if num_breaks >= 4:
            index5 = params['index5']
            break4 = params['break4']
        if num_breaks >= 5:
            index6 = params['index6']
            break5 = params['break5']

        cond = [x > break1, x > break2, x > break3, x > break4, x > break5]

        if num_breaks >= 0:
            model = normal * (x**(-index1))
        if num_breaks >= 1:
            model[np.where(cond[0])] = normal * (x[np.where(cond[0])]**(-index2)) * (break1**(-index1+index2))
        if num_breaks >= 2:
            model[np.where(cond[1])] = normal * (x[np.where(cond[1])]**(-index3)) * (break1**(-index1+index2)) * (break2**(-index2+index3))
        if num_breaks >= 3:
            model[np.where(cond[2])] = normal * (x[np.where(cond[2])]**(-index4)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4))
        if num_breaks >= 4:
            model[np.where(cond[3])] = normal * (x[np.where(cond[3])]**(-index5)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4)) * (break4**(-index4+index5))
        if num_breaks >= 5:
            model[np.where(cond[4])] = normal * (x[np.where(cond[4])]**(-index6)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4)) * (break4**(-index4+index5)) * (break5**(-index5+index6))

        return (data - model)/sigma if data is not None else model

    def flareGaussian(params, x, data=None):
        """Gaussian flare curve, always symmetrical so not always ideal."""
        height = params['height']
        centre = params['centre']
        width  = params['width']

        model = height * np.exp(-((x-centre)**2)/(2*(width**2)))

        return (model - data) if data is not None else model

    def flareFred(params, x, data=None):
        """Fast-rise exponential-decay (FRED) curve."""
        t_max = params['peak']
        rise  = params['rise']
        decay = params['decay']
        sharp = params['sharp']
        amp   = params['amp']

        cond = x >= t_max
        model = amp * np.exp(-((abs(x-t_max)/rise)**sharp))
        model[np.where(cond)] = amp * np.exp(-((abs(x[np.where(cond)]-t_max)/decay)**sharp))

        for idx, number in enumerate(model):
            if np.isinf(number) or np.isnan(number):
                raise ValueError('it appears there is an infinite or nan value, look at code comments.')
                # if np.isinf(number) or np.isnan(number):
                #     model[idx] = model[idx-1]
                # if number < 0 or number > amp:
                #     model[idx] = 0
        return (model-data) if data is not None else model


    def flareFred_archive(params, x, data=None):
        """Old equation for a FRED curve, didn't work so neatly."""
        rise  = params['rise']      # use 1/4 width
        decay = params['decay']     # use 3/4 width
        time  = params['time']     # use centre
        amp   = params['amp'] # use height

        model = amp * np.exp(-(rise/(x-time))-((x-time)/decay))

        for idx, number in enumerate(model):
            if np.isinf(number) or np.isnan(number):
                model[idx] = model[idx-1]
            if number < 0 or number > amp:
                model[idx] = 0
        return (model-data) if data is not None else model
