# -*- coding: utf-8 -*-


"""laff.laff: provides entry point main()."""


__version__ = "0.0.1"


import sys
from .models import Models
from .lcimport import Imports


def main():
    print("Executing LAFF version %s." % __version__)
    print("List of argument strings: %s" % sys.argv[1:])
    print("Stuff and Boo():\n%s\n%s" % (Models, LAFF()))
    print(Imports())

    Models.gaussian()


class LAFF(Models):
    pass

# i think write main functrions in laff class, use def main() to run each function
# i.e. in def main run the flare finder steps, then powerlaw fitter, then printer etc