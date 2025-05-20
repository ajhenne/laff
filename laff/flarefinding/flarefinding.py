import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('laff')

def flare_finding(data, algorithm, **kwargs):
    """_summary_

    Args:
        data (_type_): _description_
        algorithm (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    ## Choose algorithm.
    if algorithm in ('default', 'savgol', ''):
        from .algorithms import flares_savgol
        flares = flares_savgol(data, **kwargs)
    elif algorithm == 'sequential':
        # oudated
        from .algorithms import sequential
        flares = sequential(data)
    else:
        raise ValueError("invalid algorithm used")
    
    ## Clause for no flares.
    if flares is not False:
        logger.info("Found %s flare(s).", len(flares))
    else:
        logger.info('No flares found')
        
    return flares

