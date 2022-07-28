import numpy as np

"""laff.models: models module within the laff package."""

class Models(object):

    def powerlaw_1break(beta, x):
        count = 1
        a1, a2, b1, norm = beta
        funclist = [lambda x: norm * (x**(-a1)), \
                    lambda x: norm * (x**(-a2)) * (b1**(-a1+a2)) ]
        condlist = [x <= b1, \
                    x > b1]
        return np.piecewise(x, condlist, funclist)

    def powerlaw_2break(beta, x):
        count = 2
        a1, a2, a3, b1, b2, norm = beta
        funclist = [lambda x: norm * (x**(-a1)), \
                    lambda x: norm * (x**(-a2)) * (b1**(-a1+a2)), \
                    lambda x: norm * (x**(-a3)) * (b1**(-a1+a2)) * (b2**(-a2+a3))]
        condlist = [x <= b1, \
                    np.logical_and(x > b1, x <= b2), \
                    x > b2]
        return np.piecewise(x, condlist, funclist)

    def powerlaw_3break(beta, x):
        count = 3
        a1, a2, a3, a4, b1, b2, b3, norm = beta
        funclist = [lambda x: norm * (x**(-a1)), \
                    lambda x: norm * (x**(-a2)) * (b1**(-a1+a2)), \
                    lambda x: norm * (x**(-a3)) * (b1**(-a1+a2)) * (b2**(-a2+a3)), \
                    lambda x: norm * (x**(-a4)) * (b1**(-a1+a2)) * (b2**(-a2+a3)) * (b3**(-a3+a4)) ]
        condlist = [x <= b1, \
                    np.logical_and(x > b1, x <=b2), \
                    np.logical_and(x > b2, x <= b3), \
                    x > b3]
        return np.piecewise(x, condlist, funclist)

    def powerlaw_4break(beta, x):
        count = 4
        a1, a2, a3, a4, a5, b1, b2, b3, b4, norm = beta
        funclist = [lambda x: norm * (x**(-a1)), \
                    lambda x: norm * (x**(-a2)) * (b1**(-a1+a2)), \
                    lambda x: norm * (x**(-a3)) * (b1**(-a1+a2)) * (b2**(-a2+a3)), \
                    lambda x: norm * (x**(-a4)) * (b1**(-a1+a2)) * (b2**(-a2+a3)) * (b3**(-a3+a4)), \
                    lambda x: norm * (x**(-a5)) * (b1**(-a1+a2)) * (b2**(-a2+a3)) * (b3**(-a3+a4)) * (b4**(-a4+a5)) ]
        condlist = [x <= b1, \
                    np.logical_and(x > b1, x <= b2), \
                    np.logical_and(x > b2, x <= b3), \
                    np.logical_and(x > b3, x <= b4), \
                    x > b4]

        return np.piecewise(x, condlist, funclist)

    def powerlaw_5break(beta, x):
        count = 5
        a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, norm = beta

        funclist = [lambda x: norm * (x**(-a1)), \
                    lambda x: norm * (x**(-a2)) * (b1**(-a1+a2)), \
                    lambda x: norm * (x**(-a3)) * (b1**(-a1+a2)) * (b2**(-a2+a3)), \
                    lambda x: norm * (x**(-a4)) * (b1**(-a1+a2)) * (b2**(-a2+a3)) * (b3**(-a3+a4)), \
                    lambda x: norm * (x**(-a5)) * (b1**(-a1+a2)) * (b2**(-a2+a3)) * (b3**(-a3+a4)) * (b4**(-a4+a5)),
                    lambda x: norm * (x**(-a6)) * (b1**(-a1+a2)) * (b2**(-a2+a3)) * (b3**(-a3+a4)) * (b4**(-a4+a5)) * (b5**(-a5+a6)) ]
        condlist = [x <= b1, \
                    np.logical_and(x > b1, x <= b2), \
                    np.logical_and(x > b2, x <= b3), \
                    np.logical_and(x > b3, x <= b4), \
                    np.logical_and(x > b4, x <= b5), \
                    x > b5]
        return np.piecewise(x, condlist, funclist)

    def flare_gaussian(beta, x):
        height, centre, width = beta
        return height * np.exp(-((x-centre)**2)/(2*(width**2)))
