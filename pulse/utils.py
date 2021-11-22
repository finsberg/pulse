#!/usr/bin/env python
import logging
import os
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


def number_of_passive_controls(params):
    # Number of passive parameters to optimize
    return sum([not v for v in params["Fixed_parameters"].values()])


def get_lv_marker(geometry):

    for key in ["ENDO", "ENDO_LV", "endo", "endo_lv"]:
        if key in geometry.markers:
            return geometry.markers[key][0]
    raise KeyError("Geometry is missing marker for LV ENDO")


def save_logger(params):

    outdir = os.path.dirname(params["sim_file"])
    logfile = "output.log" if outdir == "" else outdir + "/output.log"
    logging.basicConfig(
        filename=logfile,
        filemode="a",
        format="%(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    ffc_logger = logging.getLogger("FFC")
    ffc_logger.setLevel(logging.WARNING)
    ufl_logger = logging.getLogger("UFL")
    ufl_logger.setLevel(logging.WARNING)

    import datetime

    time = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
    logger.info(f"Time: {time}")


class UnableToChangePressureExeption(Exception):
    pass


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def print_head(for_res, display_iter=True):

    targets = for_res["optimization_targets"]
    keys = targets.keys() + ["regularization"]
    n = len(keys)

    head = f"\n{'Iter':<6}\t" if display_iter else "\n" + " " * 7
    head += (
        f"{'Obj':<10}\t" + f"{'||grad||':<10}" + "\t" + (n * "I_{:<10}\t").format(*keys)
    )

    return head


def print_line(for_res, it=None, grad_norm=None, func_value=None):

    func_value = for_res["func_value"] if func_value is None else func_value
    grad_norm = 0.0 if grad_norm is None else grad_norm

    targets = for_res["target_values"]
    reg_func = targets.pop("regularization")
    values = targets.values() + [reg_func]
    targets["regularization"] = reg_func

    n = len(values)
    line = f"{it:<6d}\t" if it is not None else ""
    line += (
        f"{func_value:<10.2e}\t"
        + f"{grad_norm:<10.2e}"
        + "\t"
        + (n * "{:<10.2e}\t").format(*values)
    )

    return line


def rename_attribute(object_, old_attribute_name, new_attribute_name):
    setattr(object_, new_attribute_name, getattr(object_, old_attribute_name))
    delattr(object_, old_attribute_name)


class TablePrint(object):
    """
    Print output in nice table format.

    **Example of use**::

      fldmap = (
         'LVP',  '0.5f',
         'LV_Volume', '0.5f',
         'Target_Volume', '0.5f',
         'I_strain', '0.2e',
         'I_volume', '0.2e',
         'I_reg', '0.2e',
         )

      my_print = TablePrint(fldmap)
      print my_print.print_head()
      print my_print.print_line(LVP=1, LV_Volume=1, Target_Volume=1,
                                I_strain=1, I_volume=1, I_reg=1)

    """

    def __init__(self, fldmap, fancyhead=False):

        if fancyhead:
            q = [int(a.split(".")[0]) for a in fldmap[1::2]]

            fmt = "\t".join(["{:" + f"{f}" + "}" for f in q])

            self.head = fmt.format(*fldmap[0::2])
        else:
            self.head = "\n" + "\t".join(fldmap[0 : len(fldmap) : 2])

        self.fmt = "\t".join(
            [
                "{" + f"{col}:{f}" + "}"
                for col, f in zip(
                    fldmap[0 : len(fldmap) : 2],
                    fldmap[1 : len(fldmap) : 2],
                )
            ],
        )

    def print_head(self):
        return self.head

    def print_line(self, **kwargs):
        return self.fmt.format(**kwargs)


class Text:
    """
    Ansi escape sequences for coloured text output
    """

    _PURPLE = "\033[95m"
    _OKBLUE = "\033[94m"
    _OKGREEN = "\033[92m"
    _YELLOW = "\033[93m"
    _RED = "\033[91m "
    _ENDC = "\033[0m"

    @staticmethod
    def blue(text):
        out = Text._OKBLUE + text + Text._ENDC
        return out

    @staticmethod
    def green(text):
        out = Text._OKGREEN + text + Text._ENDC
        return out

    @staticmethod
    def red(text):
        out = Text._RED + text + Text._ENDC
        return out

    @staticmethod
    def yellow(text):
        out = Text._YELLOW + text + Text._ENDC
        return out

    @staticmethod
    def purple(text):
        out = Text._PURPLE + text + Text._ENDC
        return out

    @staticmethod
    def decolour(text):
        to_remove = [
            Text._ENDC,
            Text._OKBLUE,
            Text._OKGREEN,
            Text._RED,
            Text._YELLOW,
            Text._PURPLE,
        ]

        for chars in to_remove:
            text = text.replace(chars, "")
        return text
