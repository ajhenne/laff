# LAFF — Lightcurve and Flare Fitter

A scientific Python package for the automated modelling of Swift-XRT and Swift-BAT gamma-ray burst (GRB) light curves. It was developed as part of my PhD to enable the statistical analysis of the full GRB population, in particular the pulses and flares observed in many bursts. It has two primary functions dealing with Swift-XRT and Swift-BAT data, respectively.

## Features 

- Functions to model Swift-XRT and Swift-BAT light curves
- Fully automated, just provide time vs. flux or time vs. count rate data
- Process typically takes about five seconds, or a few tens of second for the most complex light curves
- Returns a well-structured dictionary for each afterglow, flare and pulse component including model parameters and fitting statistics
- Plotting functions for publication-ready figures

## Usage

### Swift-XRT data

```python
afterglow, flares = laff.fitXRT(data)
```

Flares are identified within the dataset. These are temporarily removed leaving only the underlying afterglow, and the best fit among a set of broken power laws with up to five breaks is found. The removed data can then be fitted, as residuals over the afterglow, with fast-rise exponential-decay (FRED) curves, and finally all components are combined to produce a fully modelled afterglow.

`fitXRT` returns:
- `afterglow`: a dictionary containing model parameter and fit statistics
- `flares`: a list of nested dictionaries, one for each flare, containing model parameters, timings and fit statistics

### Swift-BAT data

```python
pulses = laff.fitBAT(data)
```

The data is iteratively filtered to find the noise level across the data. Residuals significantly above this threshold are then modelled with FRED pulses.

`fitBAT` returns:
- `pulses`: a list of nested dictionaries, one for each pulse, containing model parameters, timings and fit statistics

### Data import function

```python
data = laff.lcimport('/path/to/file.qdp', format='')
```

The importing function is a helper function to take in data from the several common formats Swift data can come in and prepare it for LAFF in a Pandas DataFrame. Available options for `format` are:

- `xrt_repo` - XRT light curve data that is available from the [GRB lightcurve repository](https://www.swift.ac.uk/xrt_curves/) in the .qdp format.
- `xrt_python` - for light curve data obtained from the `swifttools` [Python package](https://www.swift.ac.uk/API/), usually when analysing large batches of data, in a slightly varied .qdp format
- `bat` - BAT .csv format file containing time, count rate and error columns, obtained from manually processing BAT observation data with [Heasoft](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/).

## Installation

```
pip install laff
```

**Dependencies**

This package was built and tested in Python 3.12.4, but should work for most recent versions of Python 3.

The required packages and the specific versions everything is tested and compatible in, but any recent version should not cause conflict.

- pandas 2.2.2
- matplotlib 3.9.0
- numpy 1.26.4
- scipy 1.14.0
- astropy 5.3.4

### Standard usage

For analysing one of, or both, the XRT and BAT data of a burst.

```python
import laff

# Import data into a pandas DataFrame
xrt_data = laff.lcimport('/path/to/lightcurve_xrt.qdp', format='xrt_repo')
bat_data = laff.lcimport('/path/to/lightcurve_bat.csv', format='bat')

# Fit and plot the XRT light curve
afterglow, flares = laff.fitXRT(data)
laff.plotXRT(data, afterglow, flares)

# Fit and plot the BAT light curve
pulses = laff.fitBAT(bat_data)
laff.plotBAT(data, pulses)
```

## Contributing

This project was initially developed as part of my thesis and is now available open source. Contributions are welcome, please open issues or pull requests through GitHub.

LAFF was developed for typical Swift-XRT light curves available from the [Swift XRT lightcurve repository](https://www.swift.ac.uk/xrt_curves/), and Swift-BAT lightcurves obtained though standard Heasoft processing at 64ms. In theory, it should work for any similar binning, or data from other facilities or frequencies, in practice this may not be the case due to the parameters the modelling algorithms were designed around. If you have another mission in mind, I'd be happy to take a look - please raise a GitHub issue and advise where I can find typical lightcurves - this also means I can add a dedicated import helper function.

### Poor fits

Despite the fact I have shown some level of verification to this work through my PhD thesis, there are inevitably some erroneous results spewed out by the code. The random nature of GRBs, noise within the data and things such as observation constraints will cause some strange things to occur in the light curve and my code. The randomness also means it is difficult to fine tune an exact method to consistently catch every single dataset to a perfect standard.

While I have eye-tested a number of bursts, there are well over a thousand (and increasing) now, and I have not gone through every single one. If you notice something odd, I would love to hear, so I can continue to refine this work. You may either raise an issue or GitHub, or find ways to contact me on my GitHub profile.

## Publications

Publications in which the products of this work were used in:
- Hennessy, A. et al. (2023) 'A LOFAR prompt search for radio emission accompanying X-ray flares in GRB 210112A', *MNRAS*, 526(1), pp. 106–117. https://doi.org/10.1093/mnras/stad2670
- Hennessy, A. et al. (2025) 'A LOFAR search for coherent radio emission accompanying prompt engine activity in gamma-ray bursts', *MNRAS*, 544(1), pp. 53-66. https://doi.org/10.1093/mnras/staf1640

