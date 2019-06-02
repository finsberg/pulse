#!/usr/bin/env python
import os
import logging
import numpy as np
import dolfin
from . import numpy_mpi
from .utils import make_logger, mpi_comm_world
from .io_utils import check_and_delete
from .iterate import iterate, logger as logger_it
from .dolfin_utils import get_pressure

from . import parameters

logger = make_logger(__name__, parameters["log_level"])


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
                )
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
            ]
        )

        return dolfin.MPI.max(mpi_comm_world(), d)


def save(obj, h5name, name, h5group=""):
    """
    Save object to and HDF file.

    Parameters
    ----------

    obj : dolfin.Mesh or dolfin.Function
        The object you want to save
    name : str
        Name of the object
    h5group : str
        The folder you want to save the object
        withing the HDF file. Default: ''

    """
    group = "/".join([h5group, name])
    file_mode = "a" if os.path.isfile(h5name) else "w"

    if os.path.isfile(h5name):
        check_and_delete(h5name, group)
        file_mode = "a"
    else:
        file_mode = "w"

    logger.debug("Save {0} to {1}:{2}/{0}".format(name, h5name, h5group))

    if isinstance(obj, dolfin.Function) and obj.ufl_element().family() == "Quadrature":

        quad_to_xdmf(obj, h5name, group, file_mode)

    else:
        with dolfin.HDF5File(mpi_comm_world(), h5name, file_mode) as h5file:
            h5file.write(obj, group)


def quad_to_xdmf(obj, h5name, h5group="", file_mode="w"):

    V = obj.function_space()
    gx, gy, gz = obj.split(deepcopy=True)

    W = V.sub(0).collapse()
    # FIXME
    coords_tmp = numpy_mpi.gather_broadcast(W.tabulate_dof_coordinates())
    coords = coords_tmp.reshape((-1, 3))
    u = numpy_mpi.gather_vector(gx.vector())
    v = numpy_mpi.gather_vector(gy.vector())
    w = numpy_mpi.gather_vector(gz.vector())
    vecs = np.array([u, v, w]).T
    from ..io_utils import open_h5py, parallel_h5py

    with open_h5py(h5name) as h5file:

        if not parallel_h5py:
            if mpi_comm_world().rank == 0:

                h5file.create_dataset("/".join([h5group, "coordinates"]), data=coords)
                h5file.create_dataset("/".join([h5group, "vector"]), data=vecs)
        else:
            h5file.create_dataset("/".join([h5group, "coordinates"]), data=coords)
            h5file.create_dataset("/".join([h5group, "vector"]), data=vecs)


def inflate_to_pressure(target_pressure, problem, ntries=5, n=2, annotate=False):

    logger.debug("\nInflate geometry to p = {} kPa".format(target_pressure))
    pressure = get_pressure(problem)
    solve(target_pressure, problem, pressure, ntries, n, annotate)

    return problem.get_displacement(annotate=annotate)


def print_volumes(geometry, logger=logger, txt="original", u=None):

    logger.info(
        ("\nLV Volume of {} geometry = {:.3f} ml" "").format(
            txt, geometry.cavity_volume(chamber="lv")
        )
    )
    if geometry.is_biv:
        logger.info(
            ("RV Volume of {} geometry = {:.3f} ml" "").format(
                txt, geometry.cavity_volume(chamber="rv")
            )
        )


def solve(target_pressure, problem, pressure, ntries=5, n=2, annotate=False):

    if "adjoint" in dolfin.parameters:
        dolfin.parameters["adjoint"]["stop_annotating"] = True

    level = logger_it.level
    logger_it.setLevel(logging.WARNING)

    try:
        iterate(problem, pressure, target_pressure)

    except:
        from IPython import embed

        embed()
        exit()

    if annotate and "adjoint" in dolfin.parameters:
        # Only record the last solve, otherwise it becomes too
        # expensive memorywise.
        dolfin.parameters["adjoint"]["stop_annotating"] = False
        problem.solve()

    logger_it.setLevel(level)
    w = problem.state.copy(deepcopy=True)
    return w


def update_material_parameters(material_parameters, mesh, merge_control_str=""):

    from ..setup_optimization import RegionalParameter, merge_control

    new_matparams = {}
    for k, v in material_parameters.iteritems():
        if isinstance(v, RegionalParameter):
            geo = lambda: None
            geo.mesh = mesh
            geo.sfun = dolfin.MeshFunction("size_t", mesh, 3, mesh.domains())
            sfun = merge_control(geo, merge_control_str)

            v_new = RegionalParameter(sfun)
            v_arr = numpy_mpi.gather_vector(v.vector())

            numpy_mpi.assign_to_vector(v_new.vector(), v_arr)
            new_matparams[k] = v_new

        elif isinstance(v, dolfin.Function):
            v_new = dolfin.Function(
                dolfin.FunctionSpace(mesh, v.function_space().ufl_element())
            )
            v_arr = numpy_mpi.gather_vector(v.vector())
            numpy_mpi.assign_to_vector(v_new.vector(), v_arr)
            new_matparams[k] = v_new

        else:
            new_matparams[k] = v

    return new_matparams


def load_opt_target(h5name, h5group, key="volume", data="simulated"):

    with h5py.File(h5name) as f:
        vols = [a[:][0] for a in f[h5group]["passive_inflation"][key][data].values()]

    return vols


def save_unloaded_geometry(new_geometry, h5name, h5group, backward_displacement=None):

    fields = ["fiber", "sheet", "sheet_normal"]
    local_basis = ["circumferential", "radial", "longitudinal"]

    new_fields = []
    for fattr in fields:
        if hasattr(new_geometry, fattr) and getattr(new_geometry, fattr) is not None:
            f = getattr(new_geometry, fattr).copy()
            f.rename(fattr, "microstructure")
            new_fields.append(f)

            new_local_basis = []
    for fattr in local_basis:
        if hasattr(new_geometry, fattr) and getattr(new_geometry, fattr) is not None:
            f = getattr(new_geometry, fattr).copy()
            f.rename(fattr, "local_basis_function")
            new_local_basis.append(f)

    logger.debug("Save geometry to {}:{}".format(h5name, h5group))

    if backward_displacement:
        other_functions = {"backward_displacement": backward_displacement}
    else:
        other_functions = {}

    from mesh_generation.mesh_utils import save_geometry_to_h5

    save_geometry_to_h5(
        new_geometry.mesh,
        h5name,
        h5group,
        new_geometry.markers,
        new_fields,
        new_local_basis,
        other_functions=other_functions,
    )


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
            ]
        )

        # Just make sure that we are within the given bounds
        a = np.array(
            [
                min(
                    max(a_next[i], params["Optimization_parameters"]["matparams_min"]),
                    params["Optimization_parameters"]["matparams_max"],
                )
                for i in range(len(a_cont))
            ]
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
    logger.info("Load {}:{}".format(h5name, h5group))
    group = "/".join([h5group, "passive_inflation", "optimal_control"])
    with dolfin.HDF5File(mpi_comm_world(), h5name, "r") as h5file:
        h5file.read(paramvec, group)
