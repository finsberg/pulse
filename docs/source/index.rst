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

Installation
------------
In order to install the software you need to have installed `FEniCS <https://fenicsproject.org>`_ verision
2016.x or 2017.x. Next you can install `pulse` package using

.. code::

    pip install git+https://github.com/finsberg/pulse.git

Alternatively, you can clone / download the repository at `<https://github.com/finsberg/pulse>`_
and install the dependencies

.. code::

    pip install -r requirements.txt

and finally you can instll the `pulse` package using either

.. code::

    pip install .

or

.. code::

    python setup.py install


You can also install the library using conda

.. code::

   conda install -c finsberg pulse




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
