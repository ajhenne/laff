## Main functions.
from .laff import fitGRB, findFlares, fitFlares, fitAfterglow
from .laff import plotGRB, printGRB
from .lightcurve_import import lcimport
from .fit_bat import fitPrompt, plotPrompt, fred_flare

## Auxilliary.
from .laff import set_logging_level
from .utility import STAT_NAMES_CONTINUUM, PAR_NAMES_FLARE, PAR_NAMES_CONTINUUM