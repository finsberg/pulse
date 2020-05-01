#!/usr/bin/env python

import os
import logging

import dolfin
from . import parameters

if dolfin.__version__.startswith("20"):
    # Year based versioning
    DOLFIN_VERSION_MAJOR = float(dolfin.__version__.split(".")[0])
else:
    DOLFIN_VERSION_MAJOR = float(".".join(dolfin.__version__.split(".")[:2]))


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
                _stop_annotating = not annotate

        # Update local variable
        self._annotate = annotate


annotation = Annotation()


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


def set_default_none(NamedTuple, default=None):
    NamedTuple.__new__.__defaults__ = (default,) * len(NamedTuple._fields)


# Dummy object
class Object(object):
    pass


def make_logger(name, level=parameters["log_level"]):
    def log_if_process0(record):
        if dolfin.MPI.rank(mpi_comm_world()) == 0:
            return 1
        else:
            return 0

    mpi_filt = Object()
    mpi_filt.filter = log_if_process0

    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(0)
    # formatter = logging.Formatter('%(message)s')
    formatter = logging.Formatter(
        ("%(asctime)s - " "%(name)s - " "%(levelname)s - " "%(message)s")
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addFilter(mpi_filt)

    dolfin.set_log_level(logging.WARNING)

    ffc_logger = logging.getLogger("FFC")
    ffc_logger.setLevel(logging.WARNING)
    ffc_logger.addFilter(mpi_filt)

    ufl_logger = logging.getLogger("UFL")
    ufl_logger.setLevel(logging.WARNING)
    ufl_logger.addFilter(mpi_filt)

    return logger


logger = make_logger(__name__)


def number_of_passive_controls(params):
    # Number of passive parameters to optimize
    return sum([not v for v in params["Fixed_parameters"].values()])


def get_lv_marker(geometry):

    if "ENDO" in geometry.markers:
        return geometry.markers["ENDO"][0]
    elif "ENDO_LV" in geometry.markers:
        return geometry.markers["ENDO_LV"][0]
    else:
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
    logger.info("Time: {}".format(time))


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

    head = "\n{:<6}\t".format("Iter") if display_iter else "\n" + " " * 7
    head += (
        "{:<10}\t".format("Obj")
        + "{:<10}".format("||grad||")
        + "\t"
        + (n * "I_{:<10}\t").format(*keys)
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
    line = "{:<6d}\t".format(it) if it is not None else ""
    line += (
        "{:<10.2e}\t".format(func_value)
        + "{:<10.2e}".format(grad_norm)
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

            fmt = "\t".join(["{:" + "{}".format(f) + "}" for f in q])

            self.head = fmt.format(*fldmap[0::2])
        else:
            self.head = "\n" + "\t".join(fldmap[0 : len(fldmap) : 2])

        self.fmt = "\t".join(
            [
                "{" + "{0}:{1}".format(col, f) + "}"
                for col, f in zip(
                    fldmap[0 : len(fldmap) : 2], fldmap[1 : len(fldmap) : 2]
                )
            ]
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
