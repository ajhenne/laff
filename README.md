# Lightcurve and Flare Fitter - LAFF

A python package for automatic lightcurve and flare fitting in GRB lightcurves.

## Description

This program looks to provide automatic and consistent fitting of GRB lightcurves, ultimately intended for statistical studies of a large collection of GRBs.
Initially it looks for any 'significant' rises in flux which can be marked as a potential flare. Each potential flare is then refined and cut down until a set
of start, peak and end times for flares are designated. This flare data is temporarily removed in order to fit a broken powerlaw to the continuum data - the best
solution between 0 and up to 5 breaks is used. The flare data can then be readded and the flares fitted with either a simple gaussian, or more appropriately, a
fast-rise slow-decay (FRED) curve.

With a fully fitted model, the program will output useful information, either printed to terminal or into a csv table. Such information includes flare timings,
durations, number of breaks and fluence of the flares and continuum.

To run the program, the user simply needs to point LAFF towards an appropriate lightcurve file (current and planned formats are shown below). See below for full commands and optional instructions.

## Getting Started

### Dependencies

* Python 3
* lmfit (v1.0.3 or newer)
* astropy (v5.1 or newer)
* pandas (v1.4 or newer)
* matplotlib (v3.5 or newer)
* scipy (v1.8 or newer)
* numpy (v1.23 or newer)

All available through standard python package installation methods (e.g. pip). Earlier versions of these may work but have not been explicility tested.

### Installing

To download as pip package:
```
pip install laff
```

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)