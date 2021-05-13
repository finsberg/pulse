.. pulse documentation master file, created by
   sphinx-quickstart on Sun Sep 30 12:32:22 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pulse's documentation!
=================================

`pulse` is a python library based on `FEniCS <https://fenicsproject.org>`_
that aims to solve problems in continuum mechanics.
The source code is hosted at Github (`github.com/finsberg/pulse <https://github.com/finsberg/pulse>`_)

The `pulse` package is based on a previous vesion of
`pulse-adjoint <https://bitbucket.org/finsberg/pulse_adjoint>`_ which is
based on both FEniCs and
`dolfin-adjoint <http://www.dolfin-adjoint.org/en/latest/>`_. Now
`pulse` works without dolfin-adjoint, and the dolfin-adjoint specific features in the
old repository has now migrated to a now `pulse-adjoint` repository
at
`github.com/finsberg/pulse_adjoint <https://github.com/finsberg/pulse_adjoint>`_
which works as an extension to this library.

Overview
--------
`pulse` is a software based on `FEniCS <https://fenicsproject.org>`_  that aims to solve problems in cardiac mechanics (but is easily extended to solve more general problems in continuum mechanics). `pulse` is a results of the author's `PhD thesis <https://www.duo.uio.no/handle/10852/62015>`_, where most of the relevant background for the code can be found.
 
While FEniCS offers a general framework for solving PDEs, `pulse` specifically targets problems in continuum mechanics. Therefore, most of the code for applying compatible boundary conditions, formulating the governing equations, choosing appropriate spaces for the solutions and applying iterative strategies etc. are already implemented, so that the user can focus on the actual problem he/she wants to solve rather than implementing all the necessary code for formulating and solving the underlying equations. 

The user can pick any of the built-in meshes or choose a custom user defined mesh. The user also need to provide appropriate markers for the boundaries where the boundary conditions will be applied, as well as microstructural information (i.e information about muscle fiber orientations) if an anisotropic model is to be used. Examples of how to create custom idealized geometries as well as appropriate microstructure can be found in another repository called `ldrb <https://github.com/finsberg/ldrb>`_ which uses the Laplace-Dirichlet Rule-Based (LDRB) algorithm for assigning myocardial fiber orientations.

Installation
------------
In order to install the software you need to have installed `FEniCS <https://fenicsproject.org>`_ verision
2016.x or 2017.x.  The `pulse` package can be installed with `pip`

.. code::

    pip install fenics-pulse

or if you need the most recent version you can install the source

.. code::

    pip install git+https://github.com/finsberg/pulse.git


You can also install the library using conda

.. code::

   conda install -c conda-forge pulse
   conda install -c finsberg pulse

It is also possible to use Docker. There is a prebuilt docker image using FEniCS 2017.2, python3.6 and pulse. You can get it by typing

.. code::

   docker pull finsberg/pulse:latest


Source code
-----------
Source code is available at GitHub https://github.com/finsberg/pulse




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
