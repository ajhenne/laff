################################################################################
# [ LAFF.LCIMPORT ]
################################################################################
# A collection of functions to import GRB lightcurves from various missions and
# file formats.
#
# This module is optional as the required input is simply an array of time and
# flux counts. This module provides a convenient set of functions for some
# common missions and file types.
################################################################################
# Currently supported:
# > swift-xrt.qdp files from Swift Online Archive
################################################################################

from .swift import (
    lc_swift_online_archive as swift_xrt,
)