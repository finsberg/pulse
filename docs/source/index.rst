|InstallConda| |CircleCI| |Platform| |Binder| |status| |codecov|

pulse
=====

A software for solving problems in cardiac mechanics.

Overview
--------

``pulse`` is a software based on `FEniCS <https://fenicsproject.org>`__
that aims to solve problems in cardiac mechanics (but is easily extended
to solve more general problems in continuum mechanics). ``pulse`` is a
results of the author’s `PhD
thesis <https://www.duo.uio.no/handle/10852/62015>`__, where most of the
relevant background for the code can be found.

While FEniCS offers a general framework for solving PDEs, ``pulse``
specifically targets problems in continuum mechanics. Therefore, most of
the code for applying compatible boundary conditions, formulating the
governing equations, choosing appropriate spaces for the solutions and
applying iterative strategies etc. are already implemented, so that the
user can focus on the actual problem he/she wants to solve rather than
implementing all the necessary code for formulating and solving the
underlying equations.

Installation instructions
-------------------------

Install with pip
~~~~~~~~~~~~~~~~

``pulse`` can be installed directly from
`PyPI <https://pypi.org/project/fenics-pulse/>`__

::

   python3  -m pip install fenics-pulse

or you can install the most recent development version

::

   python3 -m pip install git+https://github.com/finsberg/pulse.git

Install with conda
~~~~~~~~~~~~~~~~~~

You can also install the package using ``conda``

::

   conda install -c conda-forge pulse

Docker
~~~~~~

It is also possible to use Docker. There is a prebuilt docker image
using FEniCS 2017.2, python3.6 and pulse. You can get it by typing

::

   docker pull finsberg/pulse:latest

Requirements
------------

-  FEniCS version 2016 or newer

Note that if you install FEniCS using anaconda then you will not get
support for parallel HDF5 see e.g `this
issue <https://github.com/conda-forge/hdf5-feedstock/issues/51>`__.

Getting started
---------------

Check out the demos in the demo folder. These demos are currently in
jupyter notebook format. If you want to run them as python files you can
convert the notebooks to ``.py`` files using e.g
`jupytext <https://jupytext.readthedocs.io/en/latest/>`__

Automated test
--------------

Test are provided in the folder ```tests`` <tests>`__. You can run the
test with ``pytest``

::

   python3 -m pytest tests -vv

Documentation
-------------

Documentation can be found at
`finsberg.github.io/pulse <https://finsberg.github.io/pulse>`__ You can
create documentation yourselves by typing ``make html`` in the root
directory.

Citing
------

If you use ``pulse`` in your own research, please cite the `JOSS
paper <https://joss.theoj.org/papers/10.21105/joss.01539>`__

::

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

Known issues
------------

-  If you encounter errors with ``h5py`` try to uninstall it
   (``pip uninstall h5py``) and then re-install it without installing
   any binary packages, i.e

::

   python3 -m pip install h5py --no-binary=h5py

.. |InstallConda| image:: https://anaconda.org/finsberg/pulse/badges/installer/conda.svg
   :target: https://anaconda.org/finsberg/pulse
.. |CircleCI| image:: https://circleci.com/gh/finsberg/pulse.svg?style=shield
   :target: https://circleci.com/gh/finsberg/pulse
.. |Platform| image:: https://anaconda.org/finsberg/pulse/badges/platforms.svg
   :target: https://anaconda.org/finsberg/pulse
.. |Binder| image:: https://binder.pangeo.io/badge_logo.svg
   :target: https://binder.pangeo.io/v2/gh/finsberg/pulse/master?filepath=index.ipynb
.. |status| image:: http://joss.theoj.org/papers/9abee735e6abadabe9252d5fcc84fd40/status.svg
   :target: http://joss.theoj.org/papers/9abee735e6abadabe9252d5fcc84fd40
.. |codecov| image:: https://codecov.io/gh/finsberg/pulse/branch/master/graph/badge.svg?token=cZEkiXSOKm
   :target: https://codecov.io/gh/finsberg/pulse


.. toctree::
   :maxdepth: 2
   :caption: Demos

   demos/demos


.. toctree::
   :maxdepth: 3
   :caption: Programmers reference:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


License
=======
LGPL version 3 or later

Contributors
============
For questions please contact Henrik Finsberg (henriknf@simula.no)
