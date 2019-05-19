[![InstallConda](https://anaconda.org/finsberg/pulse/badges/installer/conda.svg)](https://anaconda.org/finsberg/pulse)
[![CircleCI](https://circleci.com/gh/finsberg/pulse.svg?style=shield)](https://circleci.com/gh/finsberg/pulse)
[![Platform](https://anaconda.org/finsberg/pulse/badges/platforms.svg)](https://anaconda.org/finsberg/pulse)

# pulse


A software for solving problems in cardiac mechanics.
The code in this repo used to be part of [pulse-adjoint](https://bitbucket.org/finsberg/pulse_adjoint), but now works as a standalone mechanics solver without the need for dolfin-adjoint

## Installation instructions

### Install with pip
First install the requrements
```
pip install fenics-pulse
```
or you can install the most recent development version
```
pip install git+https://github.com/finsberg/pulse.git
```

### Install with conda
You can also install the package using `conda`
```
conda install -c finsberg pulse
```
However, note that there are some problems with the 2017 version of FEniCS on conda. 
If you want a working conda environment with FEniCS 2017 check out
[this gist](https://gist.github.com/finsberg/96eeb1d564aab4a73f53a46a1588e6a6)


## Requirements
* FEniCS verision 2017.x

Note that if you install FEniCS using anaconda then you will not get support for paralell HDF5
see e.g [this issue](https://github.com/conda-forge/hdf5-feedstock/issues/51).

## Getting started
Check out the demos in the demo folder.

## Documentation
Documentation can be found at [finsberg.github.io/pulse](https://finsberg.github.io/pulse)


