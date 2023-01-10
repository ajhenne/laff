"""laff.runparamters: options to change the way laff processes lightcurves."""

RiseCondition  = 2            # Condition for finding flare rises. Higher makes it stricter.
DecayCondition = 5              # Condition for finding flare decays. Higher makes it harder to find the end.
ForceBreaks    = -1             # If between 0 and 5, force a fit with this number of breaks. -1 allows the default best fit algorithm.
Mission        = "Swift/QDP"    # Change the mission/data filetype. Currently onyl Swift .qdp files accepted.
FlareShape     = "fred"         # The flare shape used for fitting: "fred" or "gaussian".