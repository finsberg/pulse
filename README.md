[![InstallConda](https://anaconda.org/finsberg/pulse/badges/installer/conda.svg)](https://anaconda.org/finsberg/pulse)
[![CircleCI](https://circleci.com/gh/finsberg/pulse.svg?style=shield)](https://circleci.com/gh/finsberg/pulse)
[![Platform](https://anaconda.org/finsberg/pulse/badges/platforms.svg)](https://anaconda.org/finsberg/pulse)
[![Binder](https://binder.pangeo.io/badge_logo.svg)](https://binder.pangeo.io/v2/gh/finsberg/pulse/master?filepath=index.ipynb)
[![status](http://joss.theoj.org/papers/9abee735e6abadabe9252d5fcc84fd40/status.svg)](http://joss.theoj.org/papers/9abee735e6abadabe9252d5fcc84fd40)

# pulse

A software for solving problems in cardiac mechanics.
The code in this repository used to be part of [pulse-adjoint](https://bitbucket.org/finsberg/pulse_adjoint), but now works as a standalone mechanics solver without the need for dolfin-adjoint.

## Overview
`pulse` is a software based on [FEniCS](https://fenicsproject.org) that aims to solve problems in cardiac mechanics (but is easily extended to solve more general problems in continuum mechanics). `pulse` is a results of the author's [PhD thesis](https://www.duo.uio.no/handle/10852/62015), where most of the relevant background for the code can be found.

While FEniCS offers a general framework for solving PDEs, `pulse` specifically targets problems in continuum mechanics. Therefore, most of the code for applying compatible boundary conditions, formulating the governing equations, choosing appropriate spaces for the solutions and applying iterative strategies etc. are already implemented, so that the user can focus on the actual problem he/she wants to solve rather than implementing all the necessary code for formulating and solving the underlying equations. 

## Installation instructions

### Install with pip
`pulse` can be installed directly from [PyPI](https://pypi.org/project/fenics-pulse/)
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
`pulse` is also available on conda-forge
```
conda install -c conda-forge pulse
```

### Docker
It is also possible to use Docker. There is a prebuilt docker image
using FEniCS 2017.2, python3.6 and pulse. You can get it by typing
```
docker pull finsberg/pulse:latest
```

## Requirements
* FEniCS version 2016 or newer

Note that if you install FEniCS using anaconda then you will not get support for parallel HDF5
see e.g [this issue](https://github.com/conda-forge/hdf5-feedstock/issues/51).

## Getting started
Check out the demos in the demo folder.

## Automated test
Test are provided in the folder [`tests`](tests). You can run the test
with `pytest`
```
python -m pytest tests -vv
```


## Documentation
Documentation can be found at [finsberg.github.io/pulse](https://finsberg.github.io/pulse)
You can create documentation yourselves by typing `make html` in the
root directory.

## Citing

If you use `pulse` in your own research, please cite the [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.01539)

```
@article{pulse,
  doi = {10.21105/joss.01539},
  url = {https://doi.org/10.21105/joss.01539},
  year  = {2019},
  month = {sept},
  publisher = {The Open Journal},
  volume = {4},
  number = {41},
  pages = {1539},
  author = {Henrik Finsberg},
  title = {pulse: A python package based on FEniCS for solving problems in cardiac mechanics},
  journal = {The Journal of Open Source Software}
}
```

## Known issues
* If you encounter errors with `h5py` try to uninstall it (`pip uninstall h5py`) and then re-install it without installing any binary packages, i.e
```
pip install h5py --no-binary=h5py
```

