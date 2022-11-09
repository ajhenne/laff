from lib2to3.pgen2.token import AMPER
import numpy as np
from lmfit import Minimizer, Parameters, report_fit
import math

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
        height = params['height']
        centre = params['centre']
        width  = params['width']

        model = height * np.exp(-((x-centre)**2)/(2*(width**2)))

        return (model - data) if data is not None else model

    def flareFred(params, x, data=None):

        rise  = params['rise'].value      # use 1/4 width
        decay = params['decay'].value     # use 3/4 width
        time  = params['time'].value      # use centre
        amp   = params['amp'].value # use height

        model = amp * np.exp(-(rise/(x-time))-((x-time)/decay))

        for idx, number in enumerate(model):
            if np.isinf(number) or np.isnan(number):
                model[idx] = model[idx-1]
            if number < 0 or number > amp:
                model[idx] = 0

        return (model-data) if data is not None else model
