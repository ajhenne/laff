class Flare:

    def __init__(self, start, peak, end):
        self.start = start
        self.peak = peak
        self.end = end

    def __str__(self):
        return f"Flare {self.start} {self.peak} {self.end}"

    def returnTimes(self):
        """Return the flare index values of start/peak/decay times."""
        return self.start, self.peak, self.end

    def setParameters(self,params):
        self.params = params

    def returnParameters(self, pretty=False):
        if pretty == True:
            return [round(self.params[item].value,2) for item in self.params]
        else:
            return [self.params[item].value for item in self.params]