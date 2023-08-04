import numpy as np

def calculate_fit_statistics(data, model, params, reduced=False):

    fitted_model = model(np.array(data.time), params)
    chisq = np.sum(((data.flux - fitted_model) ** 2) / (data.flux_perr ** 2))
    
    n = len(data.time)
    m = len(params)
    dof = n - m
    r_chisq = chisq / dof
        
    return {'chisq': chisq, 'rchisq': r_chisq, 'n': len(data.time), 'npar': len(params), 'dof': dof}