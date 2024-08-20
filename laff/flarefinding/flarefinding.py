import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('laff')

def flare_finding(data, algorithm):

    # Run flare_finding.
    if algorithm == 'default' or algorithm =='test':
        raise ValueError("No default currently available!")
    
    elif algorithm == 'sequential':
        from .algorithms import sequential_findflares
        return sequential_findflares(data)

    return

