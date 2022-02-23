#!/usr/bin/env python
import logging
from typing import Optional

import dolfin
import numpy as np


class MPIFilt(logging.Filter):
    def filter(self, record):

        if dolfin.MPI.rank(mpi_comm_world()) == 0:
            return 1
        else:
            return 0


mpi_filt = MPIFilt()


def getLogger(name):
    import daiquiri

    logger = daiquiri.getLogger(name)
    logger.logger.addFilter(mpi_filt)
    return logger


logger = getLogger(__name__)


def get_dolfin_version():
    if dolfin.__version__.startswith("20"):
        # Year based versioning
        return float(dolfin.__version__.split(".")[0])
    else:
        return float(".".join(dolfin.__version__.split(".")[:2]))


try:
    DOLFIN_VERSION_MAJOR = get_dolfin_version()
except AttributeError:
    # Just assume the lastest one
    DOLFIN_VERSION_MAJOR = 2019.0


class Annotation(object):
    """
    Object holding global annotation for dolfin-adjoint
    """

    def __init__(self):
        self.annotate = False

    @property
    def annotate(self):
        return self._annotate

    @annotate.setter
    def annotate(self, annotate=False):
        """
        Set global annotation for dolfin-adjoint.
        Default False
        """
        if "adjoint" in dolfin.parameters:
            dolfin.parameters["adjoint"]["stop_annotating"] = not annotate
        else:
            try:
                from pyadjoint.tape import _stop_annotating
            except ImportError:
                # Dolfin-adjoint is most likely not installed
                pass
            else:
                _stop_annotating = not annotate  # noqa: F841,F811

        # Update local variable
        self._annotate = annotate


try:
    annotation: Optional[Annotation] = Annotation()
except Exception:
    annotation = None


class Enlisted(tuple):
    pass


def enlist(x, force_enlist=False):
    if isinstance(x, Enlisted):
        return x
    elif isinstance(x, (list, tuple, np.ndarray)):
        if force_enlist:
            return Enlisted([x])
        else:
            return Enlisted(x)
    else:
        return Enlisted([x])


def delist(x):
    if isinstance(x, Enlisted):
        if len(x) == 1:
            return x[0]
        else:
            return x
    else:
        return x


def mpi_comm_world():
    if DOLFIN_VERSION_MAJOR >= 2018:
        return dolfin.MPI.comm_world
    else:
        return dolfin.mpi_comm_world()


def value_size(obj):
    try:
        return obj.value_size()
    except AttributeError:
        value_shape = obj.value_shape()
        if len(value_shape) == 0:
            return 1
        else:
            return [0]


# Dummy object
class Object(object):
    pass


def get_lv_marker(geometry):

    for key in ["ENDO", "ENDO_LV", "endo", "endo_lv"]:
        if key in geometry.markers:
            return geometry.markers[key][0]
    raise KeyError("Geometry is missing marker for LV ENDO")


class UnableToChangePressureExeption(Exception):
    pass
