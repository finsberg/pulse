#!/usr/bin/env python
import logging

import dolfin
import h5py
import numpy as np

from . import numpy_mpi
from .dolfin_utils import get_pressure
from .iterate import iterate
from .iterate import logger as logger_it
from .utils import annotation
from .utils import getLogger
from .utils import mpi_comm_world

logger = getLogger(__name__)


class ResidualCalculator(object):
    def __init__(self, mesh):

        try:
            from scipy import spatial

        except ImportError:
            logger.warning(
                (
                    "Scipy is not install. Residual in the unloading "
                    "algorithm cannot be computed. Please install "
                    'scipy "pip install scipy" if you want to compute '
                    "the residual"
                ),
            )
            self.bbtree = None

        else:

            self.mesh = mesh
            d = self.mesh.topology().dim()
            local_points = [v.point() for v in dolfin.vertices(self.mesh)]
            coords = [(p.x(), p.y(), p.z()) for p in local_points]

            # FIXME
            coords = numpy_mpi.gather_broadcast(np.array(coords).flatten())
            coords.resize(int(len(coords) / d), d)

            self.bbtree = spatial.KDTree(coords)

    def calculate_residual(self, mesh2):
        boundmesh = dolfin.BoundaryMesh(mesh2, "exterior")
        d = max(
            [
                self.bbtree.query(dolfin.Vertex(boundmesh, v_idx).point().array())[0]
                for v_idx in range(boundmesh.num_vertices())
            ],
        )

        return dolfin.MPI.max(mpi_comm_world(), d)


def inflate_to_pressure(target_pressure, problem, ntries=5, n=2, annotate=False):

    logger.debug(f"\nInflate geometry to p = {target_pressure} kPa")
    pressure = get_pressure(problem)
    solve(target_pressure, problem, pressure, ntries, n, annotate)

    return problem.get_displacement(annotate=annotate)


def print_volumes(geometry, logger=logger, txt="original", u=None):

    logger.info(
        ("\nLV Volume of {} geometry = {:.3f} ml" "").format(
            txt,
            geometry.cavity_volume(chamber="lv"),
        ),
    )
    if geometry.is_biv:
        logger.info(
            ("RV Volume of {} geometry = {:.3f} ml" "").format(
                txt,
                geometry.cavity_volume(chamber="rv"),
            ),
        )


def solve(target_pressure, problem, pressure, ntries=5, n=2, annotate=False):

    annotation.annotate = False

    level = logger_it.logger.level
    logger_it.setLevel(logging.WARNING)

    iterate(problem, pressure, target_pressure)

    annotation.annotate = True
    # Only record the last solve, otherwise it becomes too
    # expensive memorywise.
    problem.solve()

    logger_it.setLevel(level)
    w = problem.state.copy(deepcopy=True)
    return w


def load_opt_target(h5name, h5group, key="volume", data="simulated"):

    with h5py.File(h5name) as f:
        vols = [a[:][0] for a in f[h5group]["passive_inflation"][key][data].values()]

    return vols


def continuation_step(params, it_, paramvec):

    # Use data from the two prevoious steps and continuation
    # to get a good next gues
    values = []
    vols = []

    v_target = load_opt_target(params["sim_file"], "0", "volume", "target")
    for it in range(it_):
        p_tmp = dolfin.Function(paramvec.function_space())
        load_material_parameter(params["sim_file"], str(it), p_tmp)

        values.append(numpy_mpi.gather_vector(p_tmp.vector()))

        v = load_opt_target(params["sim_file"], str(it), "volume", "simulated")
        vols.append(v)

    ed_vols = np.array(vols).T[-1]
    # Make continuation approximation
    delta = (v_target[-1] - ed_vols[-2]) / (ed_vols[-1] - ed_vols[-2])
    a_cont = (1 - delta) * values[-2] + delta * values[-1]
    a_prev = values[-1]

    # Make sure next step is not to far away
    if hasattr(a_cont, "__len__"):

        a_next = np.array(
            [
                min(max(a_cont[i], a_prev[i] / 2), a_prev[i] * 2)
                for i in range(len(a_cont))
            ],
        )

        # Just make sure that we are within the given bounds
        a = np.array(
            [
                min(
                    max(a_next[i], params["Optimization_parameters"]["matparams_min"]),
                    params["Optimization_parameters"]["matparams_max"],
                )
                for i in range(len(a_cont))
            ],
        )

    else:

        a_next = min(max(a_cont, a_prev / 2), a_prev * 2)

        # Just make sure that we are within the given bounds
        a = min(
            max(a_next, params["Optimization_parameters"]["matparams_min"]),
            params["Optimization_parameters"]["matparams_max"],
        )

    numpy_mpi.assign_to_vector(paramvec.vector(), a)


def load_material_parameter(h5name, h5group, paramvec):
    logger.info(f"Load {h5name}:{h5group}")
    group = "/".join([h5group, "passive_inflation", "optimal_control"])
    with dolfin.HDF5File(mpi_comm_world(), h5name, "r") as h5file:
        h5file.read(paramvec, group)
