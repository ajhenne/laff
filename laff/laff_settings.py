
def use_flare_model(model):
    flare_models = ['fred', 'gaussian']
    if not model in flare_models:
        raise ValueError(f"'{model}' not a valid flare model.")

    RUNPARAMETERS['flare_model'] = model

def set_par(par, val):

    int_pars = ['rise_par', 'decay_par']

    if not par in int_pars:
        raise ValueError(f"'{par}' is not a valid parameter. For non-integer parameters see documentation for specific functions.")
    if type(val) != int and type(val) != float:
        raise TypeError(f"{val} is not a valid integer/float. This variable must be a number between 0 and 10.")
    if 0 >= val and val >= 10:
        raise ValueError(f"Value must be a number between 0 and 10.")

    RUNPARAMETERS[par] = val

RUNPARAMETERS = {
    'rise_par': 3,
    'decay_par': 3,
    'flare_model': 'fred',
}