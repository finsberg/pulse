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
which works as an extenstion to this library.


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
