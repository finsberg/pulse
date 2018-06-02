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
import numpy as np
import operator as op
import logging

import dolfin

try:
    from dolfin_adjoint import Function
except ImportError:
    from dolfin import Function

from .utils import logger
from . import numpy_mpi
from .mechanicsproblem import SolverDidNotConverge


MAX_GAMMA_STEP = 0.05
MAX_PRESSURE_STEP = 0.2
MAX_PRESSURE_STEP_BIV = 0.05
MAX_CRASH = 20
MAX_ITERS = 40
    

def get_diff(current, target, control):

    if control == "gamma":
        diff = target.vector() - current.vector()
        
    elif control == "pressure":
        diff = np.subtract(target, current)

    else:
        raise ValueError("Unknown control mode {}".format(control))

    return diff


def get_current_control_value(problem, expr, control):

    if control == "gamma":
        return expr

    elif control == "pressure":
        if "p_rv" in expr:

            return (float(expr["p_lv"]),
                    float(expr["p_rv"]))
        else:
            return float(expr["p_lv"])


def assign_new_control(control, control_mode, new_control):

    if control_mode == "gamma":
        control.assign(new_control)
        
    elif control_mode == "pressure":
        if "p_rv" in control:
            control["p_lv"].assign(new_control[0])
            control["p_rv"].assign(new_control[1])
        else:
            control["p_lv"].assign(new_control)

    else:
        raise ValueError("Unknown control mode {}".format(control_mode))


def check_target_reached(problem, expr, control, target):

    current = get_current_control_value(problem, expr,  control)
    diff = get_diff(current, target, control)

    if control == "gamma":
        diff.abs()
        max_diff = diff.max()

    elif control == "pressure":
        max_diff = np.max(abs(diff))

    reached = max_diff < 1e-6
    if reached:
        logger.info("Check target reached: YES!")
    else:
        logger.info("Check target reached: NO")
        logger.info("Maximum difference: {:.3e}".format(max_diff))

    return reached


def get_initial_step(problem, expr, control, target):

    current = get_current_control_value(problem, expr, control)
    diff = get_diff(current, target, control)

    if control == "gamma":
        max_diff = dolfin.norm(diff, 'linf')
        nsteps = int(np.ceil(float(max_diff)/MAX_GAMMA_STEP) + 1)
        step = Function(expr.function_space(), name="step")
        step.vector().axpy(1.0/float(nsteps), diff)

    elif control == "pressure":
        max_diff = abs(np.max(diff))
        if hasattr(diff, "__len__") and len(diff) == 2:
            MAX_STEP = MAX_PRESSURE_STEP_BIV
        else:
            MAX_STEP = MAX_PRESSURE_STEP

        nsteps = int(np.ceil(float(max_diff) / MAX_STEP)) + 1
        step = diff/float(nsteps)

    logger.debug("Intial number of steps: {}".format(nsteps))

    if control == "gamma":
        return step, nsteps

    return step


def step_too_large(current, target, step, control):

    if control == "gamma":
        diff_before = current.vector()[:] - target.vector()[:]
        diff_before_arr = numpy_mpi.gather_broadcast(diff_before.array())

        diff_after = current.vector()[:] + \
            step.vector()[:] - target.vector()[:]
        diff_after_arr = numpy_mpi.gather_broadcast(diff_after.array())

        # diff_after.axpy(-1.0, target.vector())
        if dolfin.norm(diff_after, 'linf') < dolfin.DOLFIN_EPS:
            # We will reach the target in next iteration
            return False

        return not all(np.sign(diff_before_arr) ==
                       np.sign(diff_after_arr))

    elif control == "pressure":

        if isinstance(target, (float, int)):
            comp = op.gt if current < target else op.lt
            return comp(current + step, target)
        else:
            assert hasattr(target, "__len__")

            too_large = []
            for (c, t, s) in zip(current, target, step):
                comp = op.gt if c < t else op.lt
                too_large.append(comp(c+s, t))

            return any(too_large)


def change_step_size(step, factor, control):

    if control == "gamma":
        new_step = dolfin.Function(step.function_space())
        new_step.vector()[:] = factor*step.vector()[:]
        # new_step.assign(factor*step)

    elif control == "pressure":
        new_step = np.multiply(factor, step)

    return new_step


def print_control(control):

    def print_arr(arr):

        if len(arr) == 2:
            # This has to be (LV, RV)
            logger.info("\t{:>6}\t{:>6}".format("LV", "RV"))
            logger.info("\t{:>6.2f}\t{:>6.2f}".format(arr[0],
                                                      arr[1]))
            
        elif len(arr) == 3:
            # This has to be (LV, Septum, RV)
            logger.info("\t{:>6}\t{:>6}\t{:>6}".format("LV", "SEPT", "RV"))
            logger.info("\t{:>6.2f}\t{:>6.2f}\t{:>6.2f}".format(arr[0],
                                                                arr[1],
                                                                arr[2]))
        else:
            # Print min, mean and max
            logger.info("\t{:>6}\t{:>6}\t{:>6}".format("Min", "Mean", "Max"))
            logger.info("\t{:>6.2f}\t{:>6.2f}\t{:>6.2f}".format(np.min(arr),
                                                                np.mean(arr),
                                                                np.max(arr)))
    if isinstance(control, (float, int)):
        logger.info("\t{:>6.3f}".format(control))

    elif isinstance(control, (dolfin.Function, Function)):
        arr = numpy_mpi.gather_broadcast(control.vector().array())
        logger.info("\t{:>6}\t{:>6}\t{:>6}".format("Min", "Mean", "Max"))
        logger.info("\t{:>6.2f}\t{:>6.2f}\t{:>6.2f}".format(np.min(arr),
                                                            np.mean(arr),
                                                            np.max(arr)))
    elif isinstance(control, (dolfin.GenericVector, dolfin.Vector)):
        arr = numpy_mpi.gather_broadcast(control.array())
        print_arr(arr)

    elif isinstance(control, (tuple, np.ndarray, list)):
        print_arr(control)


def get_delta(new_control, c0, c1):

    if isinstance(new_control, (int, float)):
        return (new_control - c0) / float(c1 - c0)

    elif isinstance(new_control, (tuple, np.ndarray, list)):
        return (new_control[0] - c0[0]) / float(c1[0] - c0[0])

    elif isinstance(new_control, (dolfin.GenericVector, dolfin.Vector)):
        new_control_arr = numpy_mpi.gather_broadcast(new_control.array())
        c0_arr = numpy_mpi.gather_broadcast(c0.array())
        c1_arr = numpy_mpi.gather_broadcast(c1.array())
        return (new_control_arr[0] - c0_arr[0]) / float(c1_arr[0] - c0_arr[0])

    elif isinstance(new_control, (dolfin.Function, Function)):
        new_control_arr = numpy_mpi.\
                          gather_broadcast(new_control.vector().array())
        c0_arr = numpy_mpi.gather_broadcast(c0.vector().array())
        c1_arr = numpy_mpi.gather_broadcast(c1.vector().array())
        return (new_control_arr[0] - c0_arr[0]) / float(c1_arr[0] - c0_arr[0])


def iterate_pressure(problem, target, p_expr,
                     continuation=True, max_adapt_iter=8, adapt_step=True,
                     max_nr_crash=MAX_CRASH, max_iters=MAX_ITERS):
    """
    Using the given problem, iterate control to given target.

    *Parameters*

    problem (LVProblem)
        The problem
    target (dolfin.Function or tuple or float)
        The target value. Typically a float if target is LVP, a tuple
        if target is (LVP, RVP) and a function if target is gamma.
    p_expr (dict)
        A dictionary with expression for the pressure and keys
        'p_lv' (and 'p_rv' if BiV)
    continuation (bool)
        Apply continuation for better guess for newton problem
        Note: Replay test seems to fail when continuation is True,
        but taylor test passes
    max_adapt_iter (int)
        If number of iterations is less than this number and adapt_step=True,
        then adapt control step
    adapt_step (bool)
        Adapt / increase step size when sucessful iterations are achevied.

    """
    assert p_expr is not None, "provide the pressure"
    assert isinstance(p_expr, dict), "p_expr should be a dictionray"
    assert "p_lv" in p_expr, "p_expr do not have the key 'p_lv'"

    target_reached = check_target_reached(problem, p_expr, "pressure", target)
    logger.info("\nIterate Control: pressure")

    step = get_initial_step(problem, p_expr, "pressure", target)
    new_control = get_current_control_value(problem, p_expr, "pressure")

    logger.info("Current value")
    print_control(new_control)
    control_values = [new_control]
    prev_states = [problem.state.copy(deepcopy=True)]

    ncrashes = 0
    niters = 0

    while not target_reached:

        niters += 1
        control_value_old = control_values[-1]
        state_old = prev_states[-1]
        
        if ncrashes > MAX_CRASH or niters > 2*MAX_ITERS:

            # Go to last converged state
            assign_new_control(p_expr, "pressure", control_value_old)
            problem.reinit(state_old)
            raise SolverDidNotConverge

        first_step = len(prev_states) < 2

        # Check if we are close
        if step_too_large(control_value_old, target, step, "pressure"):
            logger.info("Change step size for final iteration")
            # Change step size so that target is reached in the next iteration
            step = target-control_value_old

        new_control = get_current_control_value(problem, p_expr, "pressure")
        new_control += step

        assign_new_control(p_expr, "pressure", new_control)
        logger.info("\nTry new pressure")
        print_control(new_control)

        # Prediction step (Make a better guess for newtons method)
        # Assuming state depends continuously on the control
        if not first_step and continuation:
            c0, c1 = control_values[-2:]
            s0, s1 = prev_states

            delta = get_delta(new_control, c0, c1)

            if not dolfin.parameters["adjoint"]["stop_annotating"]:
                w = dolfin.Function(problem.state.function_space())

                w.vector().zero()
                w.vector().axpy(1.0-delta, s0.vector())
                w.vector().axpy(delta, s1.vector())
                problem.reinit(w, annotate=True)
            else:
                problem.state.vector().zero()
                problem.state.vector().axpy(1.0-delta, s0.vector())
                problem.state.vector().axpy(delta, s1.vector())

        try:
            nliter, nlconv = problem.solve()
            if not nlconv:
                raise SolverDidNotConverge("Problem did not converge")

        except SolverDidNotConverge as ex:
            logger.debug(ex)
            logger.info("\nNOT CONVERGING")
            logger.info("Reduce control step")
            ncrashes += 1

            new_control -= step

            # Assign old state
            logger.debug("Assign old state")
            # problem.reinit(state_old)
            problem.state.vector().zero()
            problem.reinit(state_old)

            # Assign old control value
            logger.debug("Assign old control")
            assign_new_control(p_expr, "pressure", new_control)
            # Reduce step size
            step = change_step_size(step, 0.5, "pressure")

            continue

        else:
            ncrashes = 0
            logger.info("\nSUCCESFULL STEP:")

            target_reached = check_target_reached(problem, p_expr,
                                                  "pressure", target)

            if not target_reached:

                if nliter < max_adapt_iter and adapt_step:
                    logger.info("Adapt step size. New step size:")
                    step = change_step_size(step, 1.5, "pressure")
                    print_control(step)

                control_values.append(new_control)

                if first_step:
                    prev_states.append(problem.state.copy(deepcopy=True))
                else:

                    # Switch place of the state vectors
                    prev_states = [prev_states[-1], prev_states[0]]

                    # Inplace update of last state values
                    prev_states[-1].vector().zero()
                    prev_states[-1].vector().axpy(1.0, problem.state.vector())

    return control_values, prev_states


def iterate_expression(problem, expr, attr, target, continuation=True,
                       max_adapt_iter=8, adapt_step=True,
                       max_nr_crash=MAX_CRASH, max_iters=MAX_ITERS,
                       initial_number_of_steps=3, log_level=logging.INFO):
    """
    Iterate expression with attribute attr to target.
    Increment until expr.attr = target

    """

    old_level = logger.level
    logger.setLevel(log_level)
    logger.info("\nIterate Control")

    assert isinstance(expr, dolfin.Expression)
    assert isinstance(target, (float, int))
    if isinstance(target, int):
        target = float(target)

    val = getattr(expr, attr)
    step = abs(target - val) / float(initial_number_of_steps)

    logger.info("Current value: {}".format(val))
    control_values = [float(val)]
    prev_states = [problem.state.copy(deepcopy=True)]

    ncrashes = 0
    niters = 0

    target_reached = (val == target)

    while not target_reached:
        niters += 1
        if ncrashes > MAX_CRASH or niters > 2*MAX_ITERS:
            raise SolverDidNotConverge

        control_value_old = control_values[-1]
        state_old = prev_states[-1]

        first_step = len(prev_states) < 2

        # Check if we are close
        if step_too_large(control_value_old, target, step, "pressure"):
            logger.info("Change step size for final iteration")
            # Change step size so that target is reached in the next iteration
            step = target-control_value_old

        val = getattr(expr, attr)
        val += step
        setattr(expr, attr, val)
        logger.info("\nTry new control: {}".format(val))

        # Prediction step (Make a better guess for newtons method)
        # Assuming state depends continuously on the control
        if not first_step and continuation:
            logger.debug("\nContinuation step")
            c0, c1 = control_values[-2:]
            s0, s1 = prev_states[-2:]

            delta = get_delta(val, c0, c1)
            if not dolfin.parameters["adjoint"]["stop_annotating"]:
                w = dolfin.Function(problem.state.function_space())
                w.vector().zero()
                w.vector().axpy(1.0-delta, s0.vector())
                w.vector().axpy(delta, s1.vector())
                problem.reinit(w, annotate=True)
            else:
                problem.state.vector().zero()
                problem.state.vector().axpy(1.0-delta, s0.vector())
                problem.state.vector().axpy(delta, s1.vector())

        try:
            nliter, nlconv = problem.solve()
            if not nlconv:
                raise SolverDidNotConverge("Problem did not converge")
        except SolverDidNotConverge as ex:
            logger.debug(ex)
            logger.info("\nNOT CONVERGING")
            logger.info("Reduce control step")
            ncrashes += 1

            val = getattr(expr, attr)
            val -= step
            setattr(expr, attr, val)

            # Assign old state
            logger.debug("Assign old state")
            # problem.reinit(state_old)
            problem.state.vector().zero()
            problem.reinit(state_old)

            # Assign old control value
            logger.debug("Assign old control")

            # Reduce step size
            step /= 2.0

            continue

        else:
            ncrashes = 0
            logger.info("\nSUCCESFULL STEP:")

            val = getattr(expr, attr)
            target_reached = (val == target)

            if not target_reached:

                if nliter < max_adapt_iter and adapt_step:
                    step *= 1.5
                    logger.info(("Adapt step size. "
                                "New step size: {}").format(step))

                control_values.append(float(val))

                prev_states.append(problem.state.copy(deepcopy=True))

    logger.setLevel(old_level)
    return control_values, prev_states


def get_mean(f):
    return numpy_mpi.gather_broadcast(f.vector().array()).mean()


def get_max(f):
    return numpy_mpi.gather_broadcast(f.vector().array()).max()


def get_max_diff(f1, f2):
    diff = f1.vector() - f2.vector()
    diff.abs()
    return diff.max()


def iterate_gamma(problem, target, gamma,
                  continuation=True, max_adapt_iter=8,
                  adapt_step=True, old_states=None, old_gammas=None,
                  max_nr_crash=MAX_CRASH, max_iters=MAX_ITERS,
                  initial_number_of_steps=None):
    """
    Using the given problem, iterate control to given target.

    *Parameters*

    problem (LVProblem)
        The problem
    target (dolfin.Function or tuple or float)
        The target value. Typically a float if target is LVP, a tuple
        if target is (LVP, RVP) and a function if target is gamma.
    control (str)
        Control mode, so far either 'pressure' or 'gamma'
    p_expr (dict)
        A dictionary with expression for the pressure and keys
        'p_lv' (and 'p_rv' if BiV)
    continuation (bool)
        Apply continuation for better guess for newton problem
        Note: Replay test seems to fail when continuation is True,
        but taylor test passes
    max_adapt_iter (int)
        If number of iterations is less than this number and adapt_step=True,
        then adapt control step
    adapt_step (bool)
        Adapt / increase step size when sucessful iterations are achevied.
    """
    old_gammas = [] if old_gammas is None else old_gammas
    old_states = [] if old_states is None else old_states

    if isinstance(target, (float, int)):
        target_ = Function(gamma.function_space())
        target_.assign(dolfin.Constant(target))
        target = target_

    elif isinstance(target, (list, np.ndarray)):
        target_ = dolfin.Function(gamma.function_space())
        numpy_mpi.assign_to_vector(target_.vector(), np.array(target))
        target = target_

    target_reached = check_target_reached(problem, gamma, "gamma", target)

    if initial_number_of_steps is None:
        step, nr_steps = get_initial_step(problem, gamma, "gamma", target)
    else:
        nr_steps = initial_number_of_steps
        diff = get_diff(gamma, target, "gamma")
        step = Function(gamma.function_space(),
                        name="gamma_step")
        step.vector().axpy(1.0/float(nr_steps), diff)

    logger.debug("\tGamma:    Mean    Max   ")
    logger.debug("\tPrevious  {:.3f}  {:.3f}  ".format(get_mean(gamma),
                                                       get_max(gamma)))
    logger.debug("\tNext      {:.3f}  {:.3f} ".format(get_mean(target),
                                                      get_max(target)))

    g_previous = gamma.copy(deepcopy=True)
    g_next = gamma.copy(deepcopy=True)

    control_values = [gamma.copy(deepcopy=True)]
    prev_states = [problem.state.copy(deepcopy=True)]

    annotate = not dolfin.parameters["adjoint"]["stop_annotating"]

    ncrashes = 0
    niters = 0

    logger.info("\n\tIncrement gamma...")
    logger.info("\tMean \tMax")

    while not target_reached:

        niters += 1
        if ncrashes > max_nr_crash or niters > max_iters:

            problem.reinit(prev_states[0])
            gamma.assign(control_values[0])

            raise SolverDidNotConverge

        state_old = prev_states[-1]

        # Check if we are close
        if step_too_large(g_previous, target, step, "gamma"):
            logger.info("Change step size for final iteration")
            # Change step size so that target is reached in the next iteration
            step = Function(target.function_space(),
                            name='final step')
            step.vector().axpy(1.0, target.vector())
            step.vector().axpy(-1.0, g_previous.vector())

        # Increment gamma
        g_next.vector()[:] += step.vector()[:]
        assign_new_control(gamma, "gamma", g_next)

        # Prediction step
        # Hopefully a better guess for the newton problem
        if continuation and old_states:

            old_diffs = [dolfin.norm(gamma.vector() - g.vector(), "linf")
                         for g in old_gammas]

            cur_diff = dolfin.norm(step.vector(), "linf")

            if any([old_diff < cur_diff for old_diff in old_diffs]):

                logger.info("Assign an old state")
                idx = np.argmin(old_diffs)
                state_old = old_states[idx]

                problem.reinit(state_old, annotate=annotate)
                prev_states.append(state_old)
                control_values.append(old_gammas[idx])

        # Try to solve
        logger.info("\nTry new gamma")
        logger.info("\t{:.3f} \t{:.3f}".format(get_mean(gamma),
                                               get_max(gamma)))
        try:
            nliter, nlconv = problem.solve()

        except SolverDidNotConverge as ex:
            logger.debug(ex)
            logger.info("\nNOT CONVERGING")
            logger.info("Reduce control step")
            ncrashes += 1

            assign_new_control(gamma, "gamma", g_previous)

            # Assign old state
            logger.debug("Assign old state")
            problem.state.vector().zero()
            problem.reinit(state_old)

            step = change_step_size(step, 0.5, "gamma")

        else:
            ncrashes = 0
            logger.info("\nSUCCESFULL STEP:")
            g_next = gamma.copy(deepcopy=True)
            g_previous = gamma.copy(deepcopy=True)

            target_reached = check_target_reached(problem, gamma,
                                                  "gamma", target)
            if not target_reached:

                if nliter < max_adapt_iter and adapt_step:
                    logger.info("Adapt step size. New step size:")
                    step = change_step_size(step, 1.5, "gamma")
                    print_control(step)

                control_values.append(gamma.copy(deepcopy=True))
                prev_states.append(problem.state.copy(deepcopy=True))

    return control_values, prev_states


def iterate(control, *args, **kwargs):

    if control == "pressure":
        return iterate_pressure(*args, **kwargs)

    if control == "gamma":
        return iterate_gamma(*args, **kwargs)

    if control == "expression":
        return iterate_expression(*args, **kwargs)

