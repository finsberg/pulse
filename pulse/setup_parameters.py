#!/usr/bin/env python
import dolfin
import logging

parameters = dolfin.Parameters("Pulse_parameters")
parameters.add("log_level", dolfin.get_log_level())


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

    # dolfin.set_log_active(False)
    dolfin.set_log_level(logging.INFO)


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
