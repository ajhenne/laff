import numpy as np
from scipy.odr import ODR, Model, RealData

from ..models import (
    broken_powerlaw,
)

def _continuum_fitter(data):

    data = data[data.flare == False]

    # Guess initial parameters.
    b1, b2, b3, b4, b5 = np.logspace(np.log10(data['time'].iloc[0] ) * 1.1, \
                                     np.log10(data['time'].iloc[-1]) * 0.9, \
                                     num=5)
    a1, a2, a3, a4, a4, a6 = 1, 1, 1, 1, 1, 1
    norm = data['flux'].iloc[int(len(data.index) / 2)]

    for breaknum in range(0, 6, 1):

        pars = []

        _modelFit()

    return

def _modelFit(datapoints, breaknum, inputpar):

    model = Model(broken_powerlaw)
    odr = ODR(datapoints, model, inputpar)
    odr.set_job(fit_type=0)
    output = odr.run()

    if output.info != 1:
        i =1
        while output.info != 1 and i < 100:
            output = odr.restart()
            i += 1
    return output, output.beta, output.sd_beta



# def fitcontinuum(data):
#     data = _excludedata(data)
#     b1, b2, b3, b4, b5 = np.logspace(np.log10(data['time'].iloc[0] ) * 1.1, \
#                                      np.log10(data['time'].iloc[-1]) * 0.9, \
#                                      num=5)
#     a1, a2, a3, a4, a5, a6 = 1, 1, 1, 1, 1, 1
#     norm = 1e-7

#     brk1_fit, brk1_param = _modelfitter(data, laffmodels.powerlaw_1break, \
#         [a1,a2,b3,norm])
#     brk2_fit, brk2_param = _modelfitter(data, laffmodels.powerlaw_2break, \
#         [a1,a2,a3,b2,b4,norm])
#     brk3_fit, brk3_param = _modelfitter(data, laffmodels.powerlaw_3break, \
#         [a1,a2,a3,a4,b2,b3,b4,norm])
#     brk4_fit, brk4_param = _modelfitter(data, laffmodels.powerlaw_4break, \
#         [a1,a2,a3,a4,a5,b1,b2,b4,b5,norm])
#     brk5_fit, brk5_param = _modelfitter(data, laffmodels.powerlaw_5break, \
#         [a1,a2,a3,a4,a5,a6,b1,b2,b3,b4,b5,norm])

#     fits = [laffmodels.powerlaw_1break, laffmodels.powerlaw_2break, \
#             laffmodels.powerlaw_3break, laffmodels.powerlaw_4break, \
#             laffmodels.powerlaw_5break]
#     pars = [brk1_param, brk2_param, brk3_param, brk4_param, brk5_param]