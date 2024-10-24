import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('laff')

def flare_finding(data, algorithm):
    """Get the intended algorithm and find flares."""

    if algorithm in ('default', ''):
        algorithm = 'sequential'

    #####
    
    if algorithm == 'sequential':
        from .algorithms import sequential
        return sequential(data)
    
    elif algorithm == 'sequential_smooth':
        from .algorithms import sequential
        return sequential(data, smooth=True)
    
    elif algorithm == 'test':
        from .algorithms import apply_filter
        return apply_filter(data)

    return

