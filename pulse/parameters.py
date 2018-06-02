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
# from .dolfinimport import *
# from .adjoint_contraction_args import *
import dolfin
import logging

# from . import config


def check_parameters(params):
    """Check that parameters are consistent.
    If not change the parameters and print out
    a warning

    :param params: Application parameters

    """
    pass

def setup_adjoint_contraction_parameters(material_model="holzapfel_ogden"):

    params = setup_application_parameters(material_model)

    # Patient parameters
    patient_parameters = setup_patient_parameters()
    params.add(patient_parameters)

    # Optimization parameters
    opt_parameters = setup_optimization_parameters()
    params.add(opt_parameters)

    # Weigths for each optimization target
    optweigths_parameters = setup_optimization_weigths()
    params.add(optweigths_parameters)

    # Parameter for the unloading
    unload_params = setup_unloading_parameters()
    params.add(unload_params)
        
    check_parameters(params)
    
    return params


def setup_general_parameters():
    """
    Parameters to speed up the compiler
    """

    # Parameter for the compiler
    flags = ["-O3", "-ffast-math", "-march=native"]
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
    dolfin.parameters["form_compiler"]["representation"] = "uflacs"
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    # dolfin.parameters["adjoint"]["test_derivative"] = True
    # dolfin.parameters["std_out_all_processes"] = False
    # dolfin.parameters["num_threads"] = 8
    
    dolfin.set_log_active(False)
    dolfin.set_log_level(logging.INFO)


def setup_patient_parameters():
    """
    Have a look at :py:class:`patient_data.FullPatient`
    for options

    Defaults are

    +------------------+-----------------+---------------+
    | key              | Default Value   | Description   |
    +==================+=================+===============+
    | weight_rule      | equal           |               |
    +------------------+-----------------+---------------+
    | weight_direction | all             |               |
    +------------------+-----------------+---------------+
    | mesh_path        |                 |               |
    +------------------+-----------------+---------------+
    | data_path        |                 |               |
    +------------------+-----------------+---------------+
    | echo_path        |                 |               |
    +------------------+-----------------+---------------+

    """
    params = dolfin.Parameters("Patient_parameters")

    params.add("weight_rule",
               config.DEFAULT_WEIGHT_RULE,
               config.WEIGHT_RULES)
    params.add("weight_direction",
               config.DEFAULT_WEIGHT_DIRECTION,
               config.WEIGHT_DIRECTIONS)

    params.add("data_path", "")
    params.add("mesh_path", "")
    params.add("echo_path", "")
    params.add("mesh_group", "")

    # Which index does the geometry correspond to
    # "0" means first measurement numer
    # "-1" means end-diastole
    # Any number between 0 and passive filling duration
    # also work (passive filling duration would correspond)
    # to end-diastole. 
    params.add("geometry_index", "0")

    return params


def setup_optimization_weigths():
    """
    Set which targets to use
    Default solver parameters are:

    +----------------------+-----------------------+
    |Key                   | Default value         |
    +======================+=======================+
    | volume               | 0.0                   |
    +----------------------+-----------------------+
    | rv_volume            | 0.0                   |
    +----------------------+-----------------------+
    | regional_strain      | 0.0                   |
    +----------------------+-----------------------+
    | full_strain          | 0.0                   |
    +----------------------+-----------------------+
    | displacement         | 0.0                   |
    +----------------------+-----------------------+
    """

    params = dolfin.Parameters("Optimization_weigths")
    params.add('volume', 0.0)
    params.add('rv_volume', 0.0)
    params.add('regional_strain', 0.0)
    params.add('full_strain', 0.0)
    params.add('displacement', 0.0)
    params.add('regularization', 0.0)
    return params


def setup_application_parameters(material_model="holzapfel_ogden"):
    """
    Setup the main parameters for the pipeline

    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | key                                 | Default Value                                        | Description                        |
    +=====================================+======================================================+====================================+
    | base_bc                             | 'fix_x'                                              | Boudary condition at the base.     |
    |                                     |                                                      | ['fix_x', 'fixed', 'from_seg_base] |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | matparams_space                     | 'R_0'                                                | Space for material parameters.     |
    |                                     |                                                      | 'R_0', 'regional' or 'CG_1'        |         
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | use_deintegrated_strains            | False                                                | Use full strain field              |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | nonzero_initial_guess               | True                                                 | If true, use gamma = 0 as initial  |
    |                                     |                                                      | guess for all iterations           |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | active_model                        | 'active_strain'                                      | 'active_strain', 'active stress'   |
    |                                     |                                                      | or 'active_strain_rossi'           |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | base_spring_k                       | 1.0                                                  | Basal spring constant              |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | sim_file                            | 'result.h5'                                          | Path to result file                |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | Material_parameters                 | {'a': 2.28, 'a_f': 1.685, 'b': 9.726, 'b_f': 15.779} | Material parameters                |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | phase                               | passive_inflation                                    | 'passive_inflation'                |
    |                                     |                                                      | 'active_contraction' or 'all'      |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | optimize_matparams                  | True                                                 | Optimiza materal parameter or use  |
    |                                     |                                                      | default values                     |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | state_space                         | 'P_2:P_1'                                            | Taylor-hood finite elemet          |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | gamma_space                         | 'CG_1'                                               | Space for gammma.                  |
    |                                     |                                                      | 'R_0', 'regional' or 'CG_1'        |         
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | incomp_penalty                      | 0.0                                                  | Penalty for compresssible model    |
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | compressibility                     | 'incompressible'                                     | Model for compressibility          |
    |                                     |                                                      | see compressibility.py             |         
    +-------------------------------------+------------------------------------------------------+------------------------------------+
    | active_contraction_iteration_number | 0                                                    | Iteration in the active phase      |
    +-------------------------------------+------------------------------------------------------+------------------------------------+


    """
    params = dolfin.Parameters("Application_parmeteres")

    # Output #

    # Location of output
    params.add("sim_file", "result.h5")
    # Store the results in the file within a folder
    params.add("h5group", "")

    # Parameters #

    # Spring constant at base (Note: works one for base_bc = fix_x)
    params.add("base_spring_k", 1.0)

    # Spring constatnt at pericardium (if zero - divergence free)
    params.add("pericardium_spring", 0.0)

    # Space for material parameter(s)
    # If optimization of multiple material parameters are selected,
    # then R_0 is currently the only applicable space
    params.add("matparams_space", "R_0", ["CG_1", "R_0", "regional"])

    # Models ##

    # Active model
    params.add("active_model", "active_stress", ["active_strain",
                                                 "active_stress"])

    # Material model
    params.add("material_model", material_model,
               ["holzapfel_ogden",
                "neo_hookean",
                "guccione"])

    # Material parameters
    material_parameters = setup_material_parameters(material_model)
    params.add(material_parameters)

    fixed_matparams = setup_fixed_material_parameters(material_model)
    params.add(fixed_matparams)

    # Boundary condition at base
    params.add("base_bc", "fix_x", ["from_seg_base",
                                    "fix_x",
                                    "fixed"])

    # Iterators ##

    # Active of passive phase
    params.add("phase", config.PHASES[0])

    # Iteration for active phase
    params.add("active_contraction_iteration_number", 0)

    # Additional setup ##

    # Do you want to find the unloaded geometry and use that?
    params.add("unload", False)

    # For passive optimization, include all passive points ('all')
    # or only the final point ('-1'), or specific point ('point')
    params.add("passive_weights", "all")

    # Update weights so that the initial value of the functional is 0.1
    params.add("adaptive_weights", True)

    # Space for active parameter
    params.add("gamma_space", "CG_1", ["CG_1", "R_0", "regional"])

    # If you want to optimize passive parameters
    params.add("optimize_matparams", True)

    # Normalization factor for active contraction
    # For default values see material module
    params.add("T_ref", 0.0)

    # Decouple deviatoric and isochoric strains
    params.add("dev_iso_split", True)

    # Fraction of transverse active tesion for active stress formulation.
    # 0 = active only along fiber, 1 = equal force in all directions
    # (default=0.0).
    params.add("eta", 0.0)

    # If you want to use a zero initial guess for gamma (False),
    # or use gamma from previous iteration as initial guess (True)
    params.add("initial_guess", "previous", ["previous", "zero", "smooth"])

    # Log level
    params.add("log_level", logging.INFO)
    # If False turn of logging of the forward model during
    # functional evaluation
    params.add("verbose", False)

    # If you optimize against strain which reference geometry should be used
    # to compute the strains.  "0" is the starting geometry, "ED" is the
    # end-diastolic geometry, while if you are using unloading, you can
    # also use that geometry as referece.
    params.add("strain_reference", "0", ["0", "ED", "unloaded"])

    # Relaxation parameters. If smaller than one, the step size
    # in the direction will be smaller, and perhaps avoid the solver
    # to crash.
    params.add("passive_relax", 1.0)
    params.add("active_relax", 1.0)

    # When computing the volume/strain, do you want to the project or
    # interpolate the diplacement onto a CG 1 space, or do you want to
    # keep the original displacement (default CG2)
    params.add("volume_approx", "project",
               ["project", "interpolate", "original"])
    params.add("strain_approx", "original",
               ["project", "interpolate", "original"])

    params.add("strain_tensor", "gradu", ["E", "gradu"])
    params.add("map_strain", False)

    # e.g merge region 1,2 into one region -> "1,2"
    # e.g merge region 1,2 into one region and
    # region 3,4 into one region -> "1,2:3,4"
    # this will yeild two regions (region 1 and 3)
    # Note: only applicable for regional parameters
    params.add("merge_passive_control", "")
    params.add("merge_active_control", "")

    return params


def setup_material_parameters(material_model):

    material_parameters = dolfin.Parameters("Material_parameters")

    if material_model == "guccione":
        material_parameters.add("C", 2.0)
        material_parameters.add("bf", 8.0)
        material_parameters.add("bt", 2.0)
        material_parameters.add("bfs", 4.0)

    elif material_model == "neo_hookean":
        
        material_parameters.add("mu", 0.385)
        
    else:
        # material_model == "holzapfel_ogden":
        
        material_parameters.add("a", 2.28)
        material_parameters.add("a_f", 1.685)
        material_parameters.add("b", 9.726)
        material_parameters.add("b_f", 15.779)

    return material_parameters


def setup_optimization_parameters():
    """
    Parameters for the optimization.
    Default parameters are

    +-----------------+-----------------+---------------+
    | key             | Default Value   | Description   |
    +=================+=================+===============+
    | disp            | False           |               |
    +-----------------+-----------------+---------------+
    | active_maxiter  | 100             |               |
    +-----------------+-----------------+---------------+
    | scale           | 1.0             |               |
    +-----------------+-----------------+---------------+
    | passive_maxiter | 30              |               |
    +-----------------+-----------------+---------------+
    | matparams_max   | 50.0            |               |
    +-----------------+-----------------+---------------+
    | fix_a           | False           |               |
    +-----------------+-----------------+---------------+
    | fix_a_f         | True            |               |
    +-----------------+-----------------+---------------+
    | fix_b           | True            |               |
    +-----------------+-----------------+---------------+
    | fix_b_f         | True            |               |
    +-----------------+-----------------+---------------+
    | gamma_max       | 0.9             |               |
    +-----------------+-----------------+---------------+
    | matparams_min   | 0.1             |               |
    +-----------------+-----------------+---------------+
    | passive_opt_tol | 1e-06           |               |
    +-----------------+-----------------+---------------+
    | active_opt_tol  | 1e-06           |               |
    +-----------------+-----------------+---------------+
    | method_1d       | brent           |               |
    +-----------------+-----------------+---------------+
    | method          | slsqp           |               |
    +-----------------+-----------------+---------------+
    

    """
    # Parameters for the Optimization
    params = dolfin.Parameters("Optimization_parameters")
    params.add("opt_type", "scipy_slsqp")
    params.add("method_1d", "bounded")
    params.add("active_opt_tol", 1e-10)
    params.add("active_maxiter", 100)
    params.add("passive_opt_tol", 1e-10)
    params.add("passive_maxiter", 30)
    params.add("scale", 1.0)
    
    params.add("gamma_min", 0.0)
    params.add("gamma_max", 1.0)
    
    params.add("matparams_min", 1.0)
    params.add("matparams_max", 50.0)

    params.add("soft_tol", 1e-6)
    params.add("soft_tol_rel", 0.1)

    params.add("adapt_scale", True)
    params.add("disp", False)

    # Add indices seprated with comma,
    # e.g fix first and third control "1,3"
    params.add("fixed_matparams", "")
    # Add values seprated with comma,
    # e.g fix first and third control "3.11,2.14"
    params.add("fixed_matparams_values", "")

    return params


def setup_fixed_material_parameters(material_model):

    fixed_matparams = dolfin.Parameters("Fixed_parameters")
    
    if material_model == "holzapfel_ogden":
        fixed_matparams.add("a", False)
        fixed_matparams.add("a_f", True)
        fixed_matparams.add("b", True)
        fixed_matparams.add("b_f", True)
    elif material_model == "neo_hookean":
        fixed_matparams.add("mu", False)
    elif material_model == "guccione":
        fixed_matparams.add("C", False)
        fixed_matparams.add("bf", True)
        fixed_matparams.add("bt", True)
        fixed_matparams.add("bfs", True)

    return fixed_matparams


def setup_unloading_parameters():
    """
    Parameters for coupled unloading/material parameter
    estimation.

    For info about the different parameters,
    see the unloading module.
    """

    params = dolfin.Parameters("Unloading_parameters")

    params.add("method", "hybrid",
               ["hybrid", "fixed_point", "raghavan"])
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
