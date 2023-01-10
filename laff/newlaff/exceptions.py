import os.path

class ParameterError(Exception):

    def __init__(self, parameter, input):

        self.message = f"{parameter} in runparameters.py has invalid input '{input}'"
        super().__init__(self.message)

class NoFileError(Exception):

    def __init__(self, filepath):

        # Check if file exists at filepath.
        if os.path.exists(filepath):
            self.message = f"File exists but is of the wrong format."
        else:
            self.message = f"File does not exist at this location."

        super().__init__(self.message)