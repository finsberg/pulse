#!/usr/bin/env python
# c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED
# Authors: Henrik Finsberg
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS:
# post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
"""
This script contains the code for unloading
geometry
"""

__author__ = "Henrik Finsberg (henriknf@simula.no)"
import os
import numpy as np
try:
    from scipy.optimize import minimize_scalar
    has_scipy = True
except ImportError:
    has_scipy = False

import dolfin


from . import unloading_utils as utils
from . import numpy_mpi

from .utils import make_logger
from .dolfin_utils import get_cavity_volume, get_pressure_dict

from . import parameters
logger = make_logger(__name__, parameters['log_level'])

try:
    dolfin.parameters["adjoint"]["stop_annotating"] = True
except KeyError:
    pass


def step(geometry, pressure, k, u, residual, big_res,
         material_parameters=None, params=None,
         n=2, solve_tries=1, approx="project", merge_control="",
         regen_fibers=False):

    logger.info("\n\nk = {}".format(k))

    # Create new reference geometry by moving according to rule
    U = dolfin.Function(u.function_space())
    numpy_mpi.assign_to_vector(U.vector(),
                               k * numpy_mpi.
                               gather_broadcast(u.vector().array()))

    new_geometry = geometry.copy(u=U)

    matparams = utils.update_material_parameters(material_parameters,
                                                 new_geometry.mesh,
                                                 merge_control)

    if isinstance(matparams["a"], float):
        logger.info("material parameters = {}".format(matparams["a"]))
    else:
        a_arr = numpy_mpi.gather_broadcast(matparams["a"].vector().array())
        logger.info("material parameters = {}".format(a_arr))


    from IPython import embed; embed()
    exit()
    # Make the solver
    problem, active_control, passive_control \
        = make_mechanics_problem(params, new_geometry,
                                 matparams=matparams)

    try:
        # Inflate new geometry to target pressure
        u0 = utils.inflate_to_pressure(pressure, solver, p_expr,
                                       solve_tries, n, annotate = False)
    except:
        logger.info("Failed to increase pressure")
        return big_res
    
    # Move mesh to ED
    mesh = dolfin.Mesh(new_geometry.mesh)
    move(mesh, u0, 1.0, approx)
    
    # Compute the residual
    res = residual.calculate_residual(mesh)
    logger.info("\nResidual:\t{}\n".format(res))
    return res
    

class MeshUnloader(object):
    def __init__(self, geometry, pressure,
                 material_parameters=None,
                 h5name = "test.h5",
                 options = {"maxiter": 50, "regen_fibers":False,
                            "tol": 1e-4, "solve_tries" : 20},
                 h5group = "", remove_old = False,
                 params = {}, approx = "project", merge_control=""):

        
        self.geometry = geometry
        self.pressure = pressure

        self.approx = approx
        self.merge_control = merge_control
        self.h5name = h5name
        self.h5group = h5group

        if os.path.isfile(h5name) and remove_old:
            if mpi_comm_world().rank == 0: os.remove(h5name)

        self.n = int(np.rint(np.max(np.divide(pressure, 0.4))))

        self.is_biv = isinstance(pressure, tuple) and len(pressure) == 2
        self.parameters = self.default_parameters()
        self.parameters.update(**options)
                
        self.params = setup_adjoint_contraction_parameters()
        self.params.update(params)
        self.params["phase"] = "unloading"

        
        self.material_parameters = material_parameters if material_parameters \
                                   is not None else self.params["Material_parameters"]

        msg = ("\n\n"+" Unloading options ".center(72, "-") + "\n\n" +
               "\tTarget pressure: {}\n".format(pressure) + \
               "\tBiV: {}, LV only: {}\n".format(self.is_biv, not self.is_biv) + \
               "\tApproximation: {}\n".format(approx) +\
               "\tmaxiter = {}\n".format(self.parameters["maxiter"]) + \
               "\ttolerance = {}\n".format(self.parameters["tol"]) + \
               "\tregenerate_fibers (serial only)= {}\n\n".format(self.parameters["regen_fibers"]) + \
               "".center(72, "-") + "\n")
        logger.info(msg)

        

    def default_parameters(self):
        """
        Default parameters.
        """

        return {"maxiter": 10,
                "tol": 1e-4 ,
                "lb": 0.5, 
                "ub": 2.0,
                "regen_fibers":False,
                "solve_tries":20 }

    def save(self, obj, name, h5group = ""):
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

            


    def unload(self, save =True):
        """
        Unload the geometry
        """


        if save:
            self.save(self.geometry.mesh, "original_geometry/mesh", "0")
            #self.save(self.geometry.fiber, "original_geometry/fiber", "0")

        logger.info("".center(72,"-"))
        logger.info("Start unloading".center(72,"-"))
        logger.info("".center(72,"-"))
        
        logger.info(("\nLV Volume of original geometry = "\
                     "{:.3f} ml".format(get_volume(self.geometry))))
        if self.is_biv:
            logger.info(("RV Volume of original geometry = "\
                         "{:.3f} ml".format(get_volume(self.geometry, chamber="rv"))))

        residual = utils.ResidualCalculator(self.geometry.mesh)
        u = self.initial_solve(True)
        self.U = dolfin.Function(u.function_space())
        
        self.unload_step(u, residual, True)

        logger.info("".center(72, "#")+"\nUnloading suceeding")

    def get_backward_displacement(self):
        """
        Return the current backward displacement as a
        function on the original geometry.
        """
        W = dolfin.VectorFunctionSpace(self.U.function_space().mesh(), "CG", 1)

        if 0:
            # Ideally we would do this 
            u_int = dolfin.interpolate(self.U, W)
        else:
            # This only works with dolfin-adjoint
            u_int = dolfin.project(self.U, W)
            
        u = dolfin.Function(W, name = "backward_displacement")
        u.vector()[:] = -1*u_int.vector()
        return u

    def initial_solve(self, save=True):
        """
        Inflate the original geometry to the target pressure and return
        the displacement field.
        """
        
        # Do an initial solve
        logger.info("\nDo an intial solve")

        problem, active_control, passive_control \
            = make_mechanics_problem(self.params, self.geometry,
                                     matparams=self.material_parameters)
        problem.solve()
        pressure_dict = get_pressure_dict(problem)

        u = utils.inflate_to_pressure(self.pressure, problem,
                                      pressure_dict,
                                      self.parameters["solve_tries"],
                                      self.n, annotate=False)
        return u

    def get_unloaded_geometry(self):

        return utils.update_geometry(self.geometry, self.U,
                                     self.parameters["regen_fibers"])


class RaghavanUnloader(MeshUnloader):
    """
    Assumes that the given geometry is loaded
    with the given pressure.

    Find the reference configuration corresponding
    to zero pressure by first inflating the original
    geometry to the target pressure and subtacting
    the displacement times some factor `k`.
    Use a 1D minimization algorithm from scipy to find `k`.
    The method is described in [1]
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



    Reference
    ---------
    .. [1] Raghavan, M. L., Baoshun Ma, and Mark F. Fillinger.
       "Non-invasive determination of zero-pressure geometry of
       arterial aneurysms." Annals of biomedical engineering 34.9
       (2006): 1414-1419.

    """
    def unload_step(self, u, residual, save=True):

        big_res = 100.0
        residuals = {}

        def iterate(k):

            res = step(self.geometry, self.pressure, k, u, residual,
                       big_res, self.material_parameters,
                       self.params, self.n,
                       self.parameters["solve_tries"],
                       self.approx, self.merge_control,
                       self.parameters["regen_fibers"])
            residuals[k] = res
            return res

        logger.info("\nStart iterating....")
        res = minimize_scalar(iterate, method="bounded",
                              bounds=(self.parameters["lb"],
                                      self.parameters["ub"]),
                              options={"xatol": self.parameters["tol"],
                                       "maxiter": self.parameters["maxiter"]})

        logger.info("Minimzation terminated sucessfully".center(72, "-"))
        logger.info("Found:\n\tk={:.6f}\n\tResidual={:.3e}\n".format(res.x,
                                                                     res.fun))
        logger.info("Save new reference geometry")

        numpy_mpi.assign_to_vector(self.U.vector(),
                                   res.x *
                                   numpy_mpi.
                                   gather_broadcast(u.vector().array()))
        new_geometry = utils.update_geometry(self.geometry, self.U, self.parameters["regen_fibers"])

        if save:
            self.save(new_geometry.mesh, "reference_geometry/mesh", "")


class HybridUnloader(MeshUnloader):
    """
    Assumes that the given geometry is loaded
    with the given pressure.

    Find the reference configuration corresponding
    to zero pressure using first a backward displacement method,
    described in [1] until non-convergence which is known to happen
    for some biventricular geometries. If non-convergence is
    reached switch to the algorithm algorithm outlined in [2
    using the last converging displacement from the  backward
    displacement method, otherwise continue using the backward
    displacement method until maximum number of iterations, or given
    tolerance is reached.


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

    Reference
    ---------
    .. [1] Bols, Joris, et al. "A computational method to assess the in
        vivo stresses and unloaded configuration of patient-specific blood
        vessels." Journal of computational and Applied mathematics 246 (2013):
        10-17.
    .. [2] Raghavan, M. L., Baoshun Ma, and Mark F. Fillinger.
       "Non-invasive determination of zero-pressure geometry of
       arterial aneurysms." Annals of biomedical engineering 34.9
       (2006): 1414-1419.

    """
    def unload_step(self, u, residual, save=True):

        big_res = 100.0
        residuals = {}

        U_prev = dolfin.Function(u.function_space())

        fixed_point_unloader = FixedPoint(self.geometry, self.pressure,
                                          self.material_parameters,
                                          self.h5name,
                                          self.parameters,
                                          self.h5group, False,
                                          self.params)
        fixed_point_unloader.U = self.U

        res = np.inf
        iter = 0
        done = False
        while not done and iter < self.parameters["maxiter"] and res > self.parameters["tol"]:

            u_arr = numpy_mpi.gather_broadcast(u.vector().array())
            numpy_mpi.assign_to_vector(self.U.vector(), u_arr)
            try:
                u, res = fixed_point_unloader.unload_step(u, residual,
                                                          True, True, iter)
            except RuntimeError as ex:
                logger.info(ex)
                logger.info("Fixed-point method failed".center(72, "-"))
                logger.info("Swith to the raghavan method")
                
                def iterate(k):

                    res = step(self.geometry, self.pressure, k, u, residual,
                               big_res, self.material_parameters,
                               self.params, self.n,
                               self.parameters["solve_tries"], self.approx)
                    residuals[k] = res
                    return res


                logger.info("\nStart iterating....")
                res = minimize_scalar(iterate, method="bounded",
                                      bounds=(0.5, 2.0),
                                      options={"xatol": self.parameters["tol"],
                                               "maxiter": self.parameters["maxiter"]})
                
                logger.info("Minimzation terminated sucessfully".center(72, "-"))
                logger.info("Found:\n\tk={:.6f}\n\tResidual={:.3e}\n".format(res.x,
                                                                             res.fun))
                logger.info("Save new reference geometry")
                u_arr = np.multiply(res.x,
                                    numpy_mpi.gather_broadcast(u.vector().array()))
                numpy_mpi.assign_to_vector(self.U.vector(), u_arr)
                new_geometry = utils.update_geometry(self.geometry, self.U,
                                                     self.parameters["regen_fibers"])

                if save:
                    self.save(new_geometry.mesh, "reference_geometry/mesh", "")

                done = True
            else:
                U_arr = numpy_mpi.gather_broadcast(self.U.vector().array())
                numpy_mpi.assign_to_vector(U_prev.vector(), U_arr)

                iter += 1

class FixedPointUnloader(MeshUnloader):
    """
    Assumes that the given geometry is loaded 
    with the given pressure.

    Find the reference configuration corresponding
    to zero pressure using a backward displacement method, 
    described in [1].
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

    Reference
    ---------
    .. [1] Bols, Joris, et al. "A computational method to assess the in 
        vivo stresses and unloaded configuration of patient-specific blood 
        vessels." Journal of computational and Applied mathematics 246 (2013): 10-17.

    """
    def unload_step(self, u, residual, save = True, return_u = False, iter = 0):
        """
        Unload step
        """

        res = np.inf      
        while iter < self.parameters["maxiter"] \
              and  res > self.parameters["tol"] :
            
            logger.info("\nIteration: {}".format(iter))

            u_arr = gather_broadcast(u.vector().array())
            assign_to_vector(self.U.vector(), u_arr)

            # The displacent field that we will move the mesh according to
            if save:
                self.save(self.U, "displacement", str(iter))

            
            # Create new reference geomtry
            logger.debug("Create new reference geometry")
            new_geometry = self.get_unloaded_geometry()

            # Compute volume of new reference geometry
            logger.info(("LV Volume of new reference geometry = "\
                         "{:.3f} ml".format(get_volume(new_geometry))))
            if self.is_biv:
                logger.info(("RV Volume of new reference geometry = "\
                             "{:.3f} ml".format(get_volume(new_geometry, chamber="rv"))))
        

            if save:
                self.save(new_geometry.mesh, "reference_geometry/mesh", str(iter))
                #self.save(new_geometry.fiber, "reference_geometry/fiber", str(iter))
            

            matparams = update_material_parameters(self.material_parameters,
                                                   new_geometry.mesh, self.merge_control)
    
            # Make the solver
            from ..setup_optimization import (make_solver_parameters,
                                                          check_patient_attributes)
            check_patient_attributes(new_geometry)
            params, p_expr = make_solver_parameters(self.params, new_geometry, matparams)

            from ..lvsolver import LVSolver as Solver
            solver =  Solver(params)
            logger.info("Initial solve")
            solver.solve()
        
            # Solve
            u = inflate_to_pressure(self.pressure, solver, p_expr,
                                    self.parameters["solve_tries"],
                                    self.n, annotate = False)

            logger.debug(("LV Volume of new inflated geometry +u= "\
                         "{:.3f} ml".format(get_volume(new_geometry, u))))
            # Move the mesh accoring to the new displacement
            mesh = dolfin.Mesh(new_geometry.mesh)
            move(mesh, u, 1.0, self.approx)

        
            #Compute the volume of the ned ED geometry
            ed_geometry = copy_geometry(mesh, self.geometry)
            logger.info(("LV Volume of new inflated geometry = "\
                         "{:.3f} ml".format(get_volume(ed_geometry))))
            if self.is_biv:
                logger.info(("RV Volume of new inflated geometry = "\
                             "{:.3f} ml".format(get_volume(ed_geometry, chamber="rv"))))
                
                
            if save:
                self.save(mesh, "ed_geometry", str(iter))


            # Copmute the residual
            if residual is not None:
                res = residual.calculate_residual(mesh)
                logger.info("\nResidual:\t{}\n".format(res))
            else:
                res = 0.0
            
            iter += 1

            if return_u:
                return u, res

            




if __name__ == "__main__":
    from mesh_generation.mesh_utils import load_geometry_from_h5
    geo = load_geometry_from_h5("DS.h5")
    p_ED = 0.5
    
    unloader = FixedPoint(geo, p_ED, options = {"maxiter":1})
    unloader.unload()
    
