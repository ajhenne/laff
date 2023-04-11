
def use_flare_model(model):
    flare_models = ['fred', 'gaussian']

    if not model in flare_models:
        raise ValueError(f"'{model}' not a valid flare model.")

    RUNPARAMETERS['flare_model'] = model

RUNPARAMETERS = {
    'flare_model': 'fred',
}