import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('laff')

def flare_finding(data, algorithm):
    """Get the intended algorithm and find flares."""
    
    if algorithm in ('default', 'sequential', ''):
        from .algorithms import sequential
        flares = sequential(data)
    
    elif algorithm == 'sequential_smooth':
        from .algorithms import sequential
        flares = sequential(data, smooth=True)
    
    elif algorithm == 'test':
        from .algorithms import apply_filter
        flares = apply_filter(data)
    
    else:
        raise ValueError("invalid algorithm used")
    
    if flares is not False:
        logger.info("Found %s flare(s).", {len(flares)})
    else:
        logger.info('No flares found')

    return flares

