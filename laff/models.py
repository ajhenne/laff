import numpy as np
from lmfit import Minimizer, Parameters, report_fit

"""laff.models: models module within the laff package."""

class Models(object):

    def powerlaw(params, x, data):
        num_breaks = params['num_breaks']

        # Initialise all possible parameters such that they exist.
        index1, index2, index3, index4, index5, index6 = 0,0,0,0,0,0
        break1, break2, break3, break4, break5 = 0,0,0,0,0

        # Assign param
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

        normal = params['normal']

        # NEXT CHANGE THE UPPER AND LOWER LIMITS ON THE VALUES FOR EACH LOOP
        # UPPER AND LOWER LIMIT SHOULD BE BASED ON THE LATEST BREAK VALUES FOUND

        # break1.max = break2.value
        # break2.max = break3.value
        # break3.max = break4.value
        # break4.max = break5.value

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

        return model - data



    def powerlaw_1break(beta, x):
        count = 1
        index1, index2, break1, norm = beta
        funclist = [lambda x: norm * (x**(-index1)), \
                    lambda x: norm * (x**(-index2)) * (break1**(-index1+index2)) ]
        condlist = [x <= break1, \
                    x > break1]
        return np.piecewise(x, condlist, funclist)

    def powerlaw_2break(beta, x):
        count = 2
        index1, index2, index3, break1, break2, norm = beta
        funclist = [lambda x: norm * (x**(-index1)), \
                    lambda x: norm * (x**(-index2)) * (break1**(-index1+index2)), \
                    lambda x: norm * (x**(-index3)) * (break1**(-index1+index2)) * (break2**(-index2+index3))]
        condlist = [x <= break1, \
                    np.logical_and(x > break1, x <= break2), \
                    x > break2]
        return np.piecewise(x, condlist, funclist)

    def powerlaw_3break(beta, x):
        count = 3
        index1, index2, index3, index4, break1, break2, break3, norm = beta
        funclist = [lambda x: norm * (x**(-index1)), \
                    lambda x: norm * (x**(-index2)) * (break1**(-index1+index2)), \
                    lambda x: norm * (x**(-index3)) * (break1**(-index1+index2)) * (break2**(-index2+index3)), \
                    lambda x: norm * (x**(-index4)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4)) ]
        condlist = [x <= break1, \
                    np.logical_and(x > break1, x <=break2), \
                    np.logical_and(x > break2, x <= break3), \
                    x > break3]
        return np.piecewise(x, condlist, funclist)

    def powerlaw_4break(beta, x):
        count = 4
        index1, index2, index3, index4, index5, break1, break2, break3, break4, norm = beta
        funclist = [lambda x: norm * (x**(-index1)), \
                    lambda x: norm * (x**(-index2)) * (break1**(-index1+index2)), \
                    lambda x: norm * (x**(-index3)) * (break1**(-index1+index2)) * (break2**(-index2+index3)), \
                    lambda x: norm * (x**(-index4)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4)), \
                    lambda x: norm * (x**(-index5)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4)) * (break4**(-index4+index5)) ]
        condlist = [x <= break1, \
                    np.logical_and(x > break1, x <= break2), \
                    np.logical_and(x > break2, x <= break3), \
                    np.logical_and(x > break3, x <= break4), \
                    x > break4]

        return np.piecewise(x, condlist, funclist)

    def powerlaw_5break(beta, x):
        count = 5
        index1, index2, index3, index4, index5, index6, break1, break2, break3, break4, break5, norm = beta

        funclist = [lambda x: norm * (x**(-index1)), \
                    lambda x: norm * (x**(-index2)) * (break1**(-index1+index2)), \
                    lambda x: norm * (x**(-index3)) * (break1**(-index1+index2)) * (break2**(-index2+index3)), \
                    lambda x: norm * (x**(-index4)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4)), \
                    lambda x: norm * (x**(-index5)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4)) * (break4**(-index4+index5)),
                    lambda x: norm * (x**(-index6)) * (break1**(-index1+index2)) * (break2**(-index2+index3)) * (break3**(-index3+index4)) * (break4**(-index4+index5)) * (break5**(-index5+index6)) ]
        condlist = [x <= break1, \
                    np.logical_and(x > break1, x <= break2), \
                    np.logical_and(x > break2, x <= break3), \
                    np.logical_and(x > break3, x <= break4), \
                    np.logical_and(x > break4, x <= break5), \
                    x > break5]
        return np.piecewise(x, condlist, funclist)

    def flare_gaussian(beta, x):
        height, centre, width = beta
        return height * np.exp(-((x-centre)**2)/(2*(width**2)))
