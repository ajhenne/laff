################################################################################
# [ LAFF.MODELS.CONTINUUM ]
################################################################################
# Models housing the functions for the broken powerlaw continuum for up to 5
# breaks.
################################################################################

import numpy as np

def broken_powerlaw(params, x, data=None, sigma=None):

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