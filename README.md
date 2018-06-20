# pulse
A software for solving problems in cardiac mechanics.
The code in this repo used to be part of [pulse-adjoint](https://bitbucket.org/finsberg/pulse_adjoint), but now works as a standalone mechanics solver without the need for dolfin-adjoint

## Installation intructions
Install with python
```
python setup.py install --prefix=/path/to/install/directory
```
or install with pip
```
pip install .
```

## Requrirements
* FEniCS verision 2016.x or 2017

Note that if you install FEniCS using anaconda then you will not get support for paralell HDF5
see e.g [this issue](https://github.com/conda-forge/hdf5-feedstock/issues/51).

## Getting started
Check out the demos in the demo folder.


