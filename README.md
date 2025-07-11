# LAFF — Lightcurve and Flare Fitter

A scientific Python package for the automated modelling of Swift-XRT and Swift-BAT gamma-ray burst (GRB) light curves. It was developed as part of my PhD to enable the statistical analysis of the full GRB population, in particular the pulses and flares observed in many bursts. It has two primary functions dealing with Swift-XRT and Swift-BAT data, respectively.

## Features 

- Functions to model Swift-XRT and Swift-BAT light curves
- Fully automated, just provide time/flux or time/countrate data
- Process typically takes about five seconds, or a few tens of second for the most complex light curves
- Returns a well-structured dictionary for each afterglow, flare and pulse component including model parameters and fitting statistics
- Plotting functions for publication-ready figures

## Usage

### Swift-XRT data

```python
afterglow, flares = laff.fitXRT(data)
```

Flares are identified within the dataset. These are temporarily removed leaving only the underlying afterglow, and a best fit among a set of broken power laws with up to five breaks is found. The removed data can then be fitted, as residuals over the afterglow, with fast-rise exponential-decay (FRED) curves, and finally all components are combined to produce a fully modelled afterglow.

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
- `bat` - BAT .csv format file containing time, countrate and error columns, obtained from manually processing BAT observation data with Heasoft.

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
xrt_data = laff.lcimport('/path/to/file.qdp', format='xrt_repo')
bat_data = laff.lcimport('/path/to/file.csv', format='bat')

# Fit and plot the XRT light curve
afterglow, flares = laff.fitXRT(data)
laff.plotXRT(data, afterglow, flares)

# Fit and plot the BAT light curve
pulses = laff.fitBAT(bat_data)
laff.plotBAT(data, pulses)
```

## Troubleshooting

Despite the fact I have shown some level of verification to this work through my PhD thesis, there are inevitably some erroneous results spewed out by the code. The random nature of GRBs, noise within the data and things such as observation constraints will cause some strange things to occur in the light curve and my code. The randomness also means it is difficult to fine tune an exact method to consistently catch every single dataset to a perfect standard.

While I have eye-tested a number of bursts, there are well over a thousand (and increasing) now, and I have not gone through every single one. If you notice something odd, I would love to hear so I can continue to develop this code. You may either raise an issue or Github, or find ways to contact me on my Github profile.

## Publications

A full description of the methods is described in Hennessy et al. (2025) (in prep), or my PhD thesis available at (submitted).

Publications in which the products of this work were used in:
- Hennessy, A. et al. (2023) 'A LOFAR prompt search for radio emission accompanying X-ray flares in GRB 210112A', *MNRAS*, 526(1), pp. 106–117. https://doi.org/10.1093/mnras/stad2670
- Hennessy, A. et al. (2025) submitted to *MNRAS*
 
## Contributing

This project was initially developed as part of my thesis and is now available open source. Contributions are welcome, please open issues or pull request through GitHub.

## Acknowledgements

Paper references.
