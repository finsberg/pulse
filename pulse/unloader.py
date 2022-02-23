#!/usr/bin/env python
"""
This script contains the code for unloading
geometry
"""

__author__ = "Henrik Finsberg (henriknf@simula.no)"
import os

import dolfin
import numpy as np

from . import Function, interpolate

from . import unloading_utils as utils
from .mechanicsproblem import MechanicsProblem, cardiac_boundary_conditions
from .utils import getLogger, mpi_comm_world

logger = getLogger(__name__)


def setup_unloading_parameters():
    """
    Parameters for coupled unloading/material parameter
    estimation.

    For info about the different parameters,
    see the unloading module.
    """

    params = dolfin.Parameters("Unloading_parameters")

    params.add("method", "fixed_point", ["fixed_point", "raghavan"])
    # Terminate if difference in reference (unloaded) volume
    # is less than tol
    params.add("tol", 0.05)
    # Maximum number of coupled iterations
    params.add("maxiter", 5)
    # Apply conitinuation step
    params.add("continuation", False)
    # Estimate initial guess based on loaded configuration
    params.add("estimate_initial_guess", True)

    unload_options = dolfin.Parameters("unload_options")
    unload_options.add("maxiter", 10)
    unload_options.add("tol", 0.01)
    unload_options.add("ub", 2.0)
    unload_options.add("lb", 0.5)
    unload_options.add("regen_fibers", False)

    params.add(unload_options)

    return params


# try:
#     dolfin.parameters["adjoint"]["stop_annotating"] = True
# except (KeyError, RuntimeError):
#     pass


# def step(
#     geometry,
#     pressure,
#     k,
#     u,
#     residual,
#     big_res,
#     material_parameters=None,
#     params=None,
#     n=2,
#     solve_tries=1,
#     merge_control="",
#     regen_fibers=False,
# ):

#     logger.info("\n\nk = {}".format(k))

#     # Create new reference geometry by moving according to rule
#     U = Function(u.function_space())
#     numpy_mpi.assign_to_vector(U.vector(), k * numpy_mpi.gather_vector(u.vector()))

#     new_geometry = geometry.copy(u=U)

#     matparams = utils.update_material_parameters(
#         material_parameters, new_geometry.mesh, merge_control
#     )

#     if isinstance(matparams["a"], float):
#         logger.info("material parameters = {}".format(matparams["a"]))
#     else:
#         a_arr = numpy_mpi.gather_vector(matparams["a"].vector())
#         logger.info("material parameters = {}".format(a_arr))


#     # Make the solver
#     problem, active_control, passive_control = make_mechanics_problem(
#         params, new_geometry, matparams=matparams
#     )

#     try:
#         # Inflate new geometry to target pressure
#         u0 = utils.inflate_to_pressure(
#             pressure, solver, p_expr, solve_tries, n, annotate=False
#         )
#     except:
#         logger.info("Failed to increase pressure")
#         return big_res

#     # Move mesh to ED
#     mesh = Mesh(new_geometry.mesh)
#     move(mesh, u0, 1.0)

#     # Compute the residual
#     res = residual.calculate_residual(mesh)
#     logger.info("\nResidual:\t{}\n".format(res))
#     return res


class MeshUnloader(object):
    def __init__(
        self,
        problem,
        pressure,
        h5name="test.h5",
        options=None,
        h5group="",
        overwrite=False,
        merge_control="",
    ):

        self.problem = problem
        self.pressure = pressure

        self.U = Function(problem.get_displacement(annotate=True).function_space())

        self.merge_control = merge_control
        self.h5name = h5name
        self.h5group = h5group

        if os.path.isfile(h5name) and overwrite:
            if mpi_comm_world().rank == 0:
                os.remove(h5name)

        self.n = int(np.rint(np.max(np.divide(pressure, 0.4))))

        self.parameters = self.default_parameters()
        if options is not None:
            self.parameters.update(**options)

        msg = (
            "\n\n"
            + " Unloading options ".center(72, "-")
            + "\n\n"
            + f"\tTarget pressure: {pressure}\n"
            + f"\tmaxiter = {self.parameters['maxiter']}\n"
            + f"\ttolerance = {self.parameters['tol']}\n"
            + "\tregenerate_fibers (serial only)= "
            "{}\n\n".format(self.parameters["regen_fibers"]) + "".center(72, "-") + "\n"
        )
        logger.info(msg)

    def default_parameters(self):
        """
        Default parameters.
        """

        return {
            "maxiter": 10,
            "tol": 1e-4,
            "lb": 0.5,
            "ub": 2.0,
            "regen_fibers": False,
            "solve_tries": 20,
        }

    def save(self, obj, name, h5group=""):
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
        group = os.path.join(self.h5group, h5group)
        utils.save(obj, self.h5name, name, group)

    def unload(self, save=False):
        """
        Unload the geometry
        """
        if save:
            self.save(self.problem.geometry.mesh, "original_geometry/mesh", "0")

        logger.info("".center(72, "-"))
        logger.info("Start unloading".center(72, "-"))
        logger.info("".center(72, "-"))

        logger.info(
            (
                "\nLV Volume of original geometry = "
                "{:.3f} ml".format(self.problem.geometry.cavity_volume(chamber="lv"))
            ),
        )
        if self.problem.geometry.is_biv:
            logger.info(
                (
                    "RV Volume of original geometry = "
                    "{:.3f} ml".format(
                        self.problem.geometry.cavity_volume(chamber="rv"),
                    )
                ),
            )

        residual = utils.ResidualCalculator(self.problem.geometry.mesh)

        u = self.initial_solve(True)
        self.U = Function(u.function_space())

        self.unload_step(u, residual, save=save)

        logger.info("".center(72, "#") + "\nUnloading suceeding")

    @property
    def backward_displacement(self):
        """
        Return the current backward displacement as a
        function on the original geometry.
        """
        W = dolfin.VectorFunctionSpace(self.U.function_space().mesh(), "CG", 1)
        u_int = interpolate(self.U, W)
        u = Function(W)
        u.vector()[:] = -1 * u_int.vector()
        return u

    def initial_solve(self, save=False):
        """
        Inflate the original geometry to the target pressure and return
        the displacement field.
        """

        # Do an initial solve
        logger.info("\nDo an intial solve")
        self.problem.solve()

        u = utils.inflate_to_pressure(
            self.pressure,
            self.problem,
            self.parameters["solve_tries"],
            self.n,
            annotate=True,
        )
        return u

    @property
    def unloaded_geometry(self):
        return self.problem.geometry.copy(u=self.U, factor=-1.0)


class RaghavanUnloader(MeshUnloader):
    """
    Assumes that the given geometry is loaded
    with the given pressure.

    Find the reference configuration corresponding
    to zero pressure by first inflating the original
    geometry to the target pressure and subtacting
    the displacement times some factor `k`.
    Use a 1D minimization algorithm from scipy to find `k`.
    The method is described in [1]_
    This method assumes that material properties are given.
    Make sure to change this in utils.py

    This also runs in parallel.

    Parameters
    ----------

    geometry : object
        An obeject with attributes `mesh`, `fiber`
        and `markers`.
    pressure : float or tuple
        The target pressure. If LV only provide a
        float, if BiV proble a tuple of the form (p_LV, p_RV).
    material_parameters : dict, optional
        A dictionary with material parameters
    h5name : str, optional
        Path to where you want to save the output
        Default: test.h5
    options : dict, optional, optional
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform.
            tol : float
                Tolerance of termination
            solve_tries : int
                Number of attemtps the solver should use to
                increase the pressure (after pressure reduction)
    h5group : str
        Subfolder within the HDF file where you save the results
    remove_old : bool
        If the file with the same h5name exist, delete it before starting
    params : dict
        These are parameters that is parsed to pulse_adjoint that
        make the parameters for the solver.
        See pulse_adjoint.setup_parametere.setup_adjoint_contraction_parameters



    References
    ----------
    .. [1] Raghavan, M. L., Baoshun Ma, and Mark F. Fillinger.
       "Non-invasive determination of zero-pressure geometry of
       arterial aneurysms." Annals of biomedical engineering 34.9
       (2006): 1414-1419.

    """

    def unload_step(self, u, residual, save=True):

        raise NotImplementedError("RahavanUnloader is not yet implemented")

        # big_res = 100.0
        # residuals = {}

        # def iterate(k):

        #     res = step(
        #         self.geometry,
        #         self.pressure,
        #         k,
        #         u,
        #         residual,
        #         big_res,
        #         self.material_parameters,
        #         self.params,
        #         self.n,
        #         self.parameters["solve_tries"],
        #         self.merge_control,
        #         self.parameters["regen_fibers"],
        #     )
        #     residuals[k] = res
        #     return res

        # logger.info("\nStart iterating....")
        # res = minimize_scalar(
        #     iterate,
        #     method="bounded",
        #     bounds=(self.parameters["lb"], self.parameters["ub"]),
        #     options={
        #         "xatol": self.parameters["tol"],
        #         "maxiter": self.parameters["maxiter"],
        #     },
        # )

        # logger.info("Minimzation terminated sucessfully".center(72, "-"))
        # logger.info("Found:\n\tk={:.6f}\n\tResidual={:.3e}\n".format(res.x, res.fun))
        # logger.info("Save new reference geometry")

        # numpy_mpi.assign_to_vector(
        #     self.U.vector(), res.x * numpy_mpi.gather_vector(u.vector())
        # )
        # new_geometry = utils.update_geometry(
        #     self.geometry, self.U, self.parameters["regen_fibers"]
        # )

        # if save:
        #     self.save(new_geometry.mesh, "reference_geometry/mesh", "")


class FixedPointUnloader(MeshUnloader):
    """
    Assumes that the given geometry is loaded
    with the given pressure.

    Find the reference configuration corresponding
    to zero pressure using a backward displacement method,
    described in [2]_.
    This method assumes that material properties are given.
    Make sure to change this in utils.py

    This also runs in parallel.


    References
    ----------
    .. [2] Bols, Joris, et al. "A computational method to assess the in
        vivo stresses and unloaded configuration of patient-specific blood
        vessels." Journal of computational and Applied mathematics 246 (2013): 10-17.

    """

    def unload_step(self, u, residual, save=False, it=0):
        """
        Unload step
        """

        res = np.inf
        while it < self.parameters["maxiter"] and res > self.parameters["tol"]:

            logger.info(f"\nIteration: {it}")
            self.U.vector()[:] = u.vector()

            # The displacent field that we will move the mesh according to
            if save:
                self.save(self.U, "displacement", str(it))

            # Create new reference geomtry
            logger.debug("Create new reference geometry")
            new_geometry = self.problem.geometry.copy(u=self.U, factor=-1.0)
            utils.print_volumes(new_geometry, txt="original")
            if save:
                self.save(new_geometry.mesh, "reference_geometry/mesh", str(it))

            material = self.problem.material.copy(geometry=new_geometry)
            # TDOO: Make a copy function for bcs as well
            # use cardiac_boundary_conditions function for mechanicsproblem
            # in make_solver_paramerters
            bcs = cardiac_boundary_conditions(
                geometry=new_geometry, **self.problem.bcs_parameters
            )
            problem = MechanicsProblem(
                geometry=new_geometry,
                material=material,
                bcs=bcs,
            )

            # Solve
            u = utils.inflate_to_pressure(
                self.pressure,
                problem,
                self.parameters["solve_tries"],
                self.n,
                annotate=True,
            )

            utils.print_volumes(new_geometry, txt="inflated", u=u)

            # Move the mesh accoring to the new displacement
            ed_geometry = problem.geometry.copy(u=u, factor=1.0)
            utils.print_volumes(ed_geometry, txt="new reference")

            if save:
                self.save(ed_geometry.mesh, "ed_geometry", str(it))

            # Compute the residual
            if residual is not None:
                res = residual.calculate_residual(ed_geometry.mesh)
                logger.info(f"\nResidual:\t{res}\n")
            else:
                res = 0.0

            it += 1
