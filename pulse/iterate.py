#!/usr/bin/env python
import numpy as np
import operator as op
import logging

import dolfin

try:
    from dolfin_adjoint import Function, Constant
    has_dolfin_adjoint = True
except ImportError:
    from dolfin import Function, Constant
    has_dolfin_adjoint = False

from .utils import logger
from .dolfin_utils import get_constant
from . import numpy_mpi
from .mechanicsproblem import SolverDidNotConverge


MAX_GAMMA_STEP = 0.05
MAX_PRESSURE_STEP = 0.2
MAX_PRESSURE_STEP_BIV = 0.05
MAX_CRASH = 20
MAX_ITERS = 40
    

def get_diff(current, target):

    msg = ('Expected target and current to be of same type. '
           'Got type(current) = {}, type(target) = {}'
           ).format(type(current), type(target))
    assert type(current) == type(target), msg

    if isinstance(target, (Function, dolfin.Function)):
        diff = target.vector() - current.vector()

    elif isinstance(target, (Constant, dolfin.Constant)):
        diff = np.subtract(float(target), float(current))
    elif isinstance(target, (tuple, list)):
        diff = np.subtract([float(t) for t in target],
                           [float(c) for c in target])
    else:
        try:
            diff = np.subtract(target, current)
        except Exception as ex:
            logger.error(ex)
            raise ValueError(("Unable to compute diff with type {}"
                              "").format(type(current)))

    return diff


def get_current_control_value(control, value=None):

    if has_dolfin_adjoint and value is not None:
        return value

    # else:
    #     return control
    if isinstance(control, (Function, dolfin.Function)):
        return control

    elif isinstance(control, (Constant, dolfin.Constant)):
        return float(control)
    
    elif isinstance(control, (tuple, list)):
        return tuple(float(c) for c in control)
        
    else:
        raise ValueError("Unknown control type {}".format(type(control)))


def assign_new_control(control, new_control):

    msg = ('Expected old and new control to be of same type. '
           'Got type(control) = {}, type(new_control) = {}'
           ).format(type(control), type(new_control))
    assert type(control) == type(new_control), msg

    if isinstance(control, (tuple, list)):
        for c, n in zip(control, new_control):
            c.assign(n)
    else:
        try:
            control.assign(new_control)
        except Exception as ex:
            logger.error(ex)
            raise ValueError(("Unable to assign control of type {}"
                             "").format(type(control)))


def check_target_reached(problem, current, target):

    diff = get_diff(current, target)

    if isinstance(diff, dolfin.GenericVector):
        diff.abs()
        max_diff = diff.max()

    else:
        max_diff = np.max(abs(diff))

    reached = max_diff < 1e-6
    if reached:
        logger.info("Check target reached: YES!")
    else:
        logger.info("Check target reached: NO")
        logger.info("Maximum difference: {:.3e}".format(max_diff))

    return reached


def copy(f, deepcopy=True):

    if isinstance(f, (dolfin.Function, Function)):
        return f.copy(deepcopy=deepcopy)
    elif isinstance(f, dolfin.Constant):
        return dolfin.Constant(f)
    elif isinstance(f, Constant):
        return Constant(f)
    elif isinstance(f, (float, int)):
        return f
    elif isinstance(f, (list, tuple)):
        l = []
        for fi in f:
            l.apply(copy(fi))
        return tuple(l)
    else:
        return f    


def get_initial_step(problem, current, target, nsteps=None):

    diff = get_diff(current, target)

    if isinstance(diff, dolfin.GenericVector):
        max_diff = dolfin.norm(diff, 'linf')
        if nsteps is None:
            nsteps = int(np.ceil(float(max_diff)/MAX_GAMMA_STEP) + 1)
        step = Function(current.function_space())
        step.vector().axpy(1.0/float(nsteps), diff)

    else:
        max_diff = abs(np.max(diff))
        if hasattr(diff, "__len__") and len(diff) == 2:
            MAX_STEP = MAX_PRESSURE_STEP_BIV
        else:
            MAX_STEP = MAX_PRESSURE_STEP

        if nsteps is None:
            nsteps = int(np.ceil(float(max_diff) / MAX_STEP)) + 1
        step = diff/float(nsteps)

    logger.debug("Intial number of steps: {}".format(nsteps))

    # if control == "gamma":
        # return step, nsteps

    return step


def constant2float(const):

    try:
        c = float(const)
    except TypeError:
        c = np.zeros(len(const))
        const.eval(c, c)

    return c


def step_too_large(current, target, step):

    if isinstance(target, (Constant, dolfin.Constant)):

        target = constant2float(target)
        current = constant2float(current)
        step = constant2float(step)

    if isinstance(target, (dolfin.Function, Function)):
        diff_before = current.vector()[:] - target.vector()[:]
        diff_before_arr = numpy_mpi.gather_broadcast(diff_before.get_local())

        diff_after = current.vector()[:] + \
            step.vector()[:] - target.vector()[:]
        diff_after_arr = numpy_mpi.gather_broadcast(diff_after.get_local())

        if dolfin.norm(diff_after, 'linf') < dolfin.DOLFIN_EPS:
            # We will reach the target in next iteration
            return False

        return not all(np.sign(diff_before_arr) ==
                       np.sign(diff_after_arr))

    elif isinstance(target, (float, int)):
        comp = op.gt if current < target else op.lt
        return comp(current + step, target)
    else:
        assert hasattr(target, "__len__")

        too_large = []
        for (c, t, s) in zip(current, target, step):
            comp = op.gt if c < t else op.lt
            too_large.append(comp(c+s, t))

        return any(too_large)


def change_step_size(step, factor):

    if isinstance(step, (dolfin.Function, Function)):
        new_step = dolfin.Function(step.function_space())
        new_step.vector()[:] = factor*step.vector()[:]

    else:
        new_step = np.multiply(factor, step)

    return new_step


def print_control(control):

    if isinstance(control, (Constant, dolfin.Constant)):
        control = constant2float(control)

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
        arr = numpy_mpi.gather_broadcast(control.vector().get_local())
        logger.info("\t{:>6}\t{:>6}\t{:>6}".format("Min", "Mean", "Max"))
        logger.info("\t{:>6.2f}\t{:>6.2f}\t{:>6.2f}".format(np.min(arr),
                                                            np.mean(arr),
                                                            np.max(arr)))
    elif isinstance(control, (dolfin.GenericVector, dolfin.Vector)):
        arr = numpy_mpi.gather_broadcast(control.get_local())
        print_arr(arr)

    elif isinstance(control, (tuple, np.ndarray, list)):
        print_arr(control)


def get_delta(new_control, c0, c1):

    if isinstance(c0, (Constant, dolfin.Constant)):
        c0 = constant2float(c0)
        c1 = constant2float(c1)
        new_control = constant2float(new_control)

    if isinstance(new_control, (int, float)):
        return (new_control - c0) / float(c1 - c0)

    elif isinstance(new_control, (tuple, np.ndarray, list)):
        return (new_control[0] - c0[0]) / float(c1[0] - c0[0])

    elif isinstance(new_control, (dolfin.GenericVector, dolfin.Vector)):
        new_control_arr = numpy_mpi.gather_broadcast(new_control.get_local())
        c0_arr = numpy_mpi.gather_broadcast(c0.get_local())
        c1_arr = numpy_mpi.gather_broadcast(c1.get_local())
        return (new_control_arr[0] - c0_arr[0]) / float(c1_arr[0] - c0_arr[0])

    elif isinstance(new_control, (dolfin.Function, Function)):
        new_control_arr = numpy_mpi.\
                          gather_broadcast(new_control.vector().get_local())
        c0_arr = numpy_mpi.gather_broadcast(c0.vector().get_local())
        c1_arr = numpy_mpi.gather_broadcast(c1.vector().get_local())
        return (new_control_arr[0] - c0_arr[0]) / float(c1_arr[0] - c0_arr[0])


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
            if has_dolfin_adjoint and \
               not dolfin.parameters["adjoint"]["stop_annotating"]:
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
    return numpy_mpi.gather_broadcast(f.vector().get_local()).mean()


def get_max(f):
    return numpy_mpi.gather_broadcast(f.vector().get_local()).max()


def iterate(problem, control, target,
            continuation=True, max_adapt_iter=8,
            adapt_step=True, old_states=None, old_controls=None,
            max_nr_crash=MAX_CRASH, max_iters=MAX_ITERS,
            initial_number_of_steps=None):

    """
    Using the given problem, iterate control to given target.

    *Parameters*

    problem (LVProblem)
        The problem
    control (dolfin.Function or dolfin.Constant)
        The control
    target (dolfin.Function, dolfin.Constant, tuple or float)
        The target value. Typically a float if target is LVP, a tuple
        if target is (LVP, RVP) and a function if target is gamma.
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
    old_controls = [] if old_controls is None else old_controls
    old_states = [] if old_states is None else old_states

    if isinstance(target, (float, int, list, np.ndarray)):
        value_size = 1 if isinstance(target, (float, int)) else len(target)
        target = get_constant(value_size=value_size, value_rank=0,
                              val=target, constant=Constant)

    else:
        msg = "Unknown targt type {}".format(type(target))
        assert isinstance(target, (dolfin.Constant, Constant,
                                   dolfin.Function, Function)), msg

    target_reached = check_target_reached(problem, control, target)

    step = get_initial_step(problem, control,
                            target, initial_number_of_steps)

    # logger.debug("\tGamma:    Mean    Max   ")
    # logger.debug("\tPrevious  {:.3f}  {:.3f}  ".format(get_mean(gamma),
    #                                                    get_max(gamma)))
    # logger.debug("\tNext      {:.3f}  {:.3f} ".format(get_mean(target),
    #                                                   get_max(target)))

    control_prev = None
    control_next = None
    if has_dolfin_adjoint:
        try:
            control_prev = copy(control, deepcopy=True)
            control_next = copy(control, deepcopy=True)
        except Exception as ex:
            pass

    control_values = [copy(control)]
    prev_states = [copy(problem.state)]

    ncrashes = 0
    niters = 0

    while not target_reached:

        niters += 1
        if ncrashes > max_nr_crash or niters > max_iters:

            problem.reinit(prev_states[0])
            assign_new_control(control, control_values[0])

            raise SolverDidNotConverge

        state_old = prev_states[-1]
        control_old = control_prev or control_values[-1]

        # Check if we are close
        if step_too_large(control_old, target, step):
            logger.info("Change step size for final iteration")
            
            # Change step size so that target is reached in the next iteration
            if isinstance(step, (dolfin.Function, Function)):
                step = Function(target.function_space())
                step.vector().axpy(1.0, target.vector())
                step.vector().axpy(-1.0, control_old.vector())
            else:
                step = target - control_old

        # Increment gamma
        current_control = get_current_control_value(control, control_next)
        if isinstance(current_control, (dolfin.Function, Function)):
            current_control.vector()[:] += step.vector()[:]
        else:
            current_control += step
            current_control = get_constant(current_control)
        assign_new_control(control, current_control)

        first_step = len(prev_states) < 2
        # Prediction step
        # Hopefully a better guess for the newton problem
        if not first_step and continuation:

            c0, c1 = control_values[-2:]
            s0, s1 = prev_states

            delta = get_delta(current_control, c0, c1)

            if has_dolfin_adjoint and \
               not dolfin.parameters["adjoint"]["stop_annotating"]:
                w = dolfin.Function(problem.state.function_space())

                w.vector().zero()
                w.vector().axpy(1.0-delta, s0.vector())
                w.vector().axpy(delta, s1.vector())
                problem.reinit(w, annotate=True)
            else:
                problem.state.vector().zero()
                problem.state.vector().axpy(1.0-delta, s0.vector())
                problem.state.vector().axpy(delta, s1.vector())

        logger.info("Try new control")
        print_control(control)
        try:
            nliter, nlconv = problem.solve()

        except SolverDidNotConverge as ex:
            logger.debug(ex)
            logger.info("\nNOT CONVERGING")
            logger.info("Reduce control step")
            ncrashes += 1

            assign_new_control(control, control_old)

            # Assign old state
            logger.debug("Assign old state")
            problem.state.vector().zero()
            problem.reinit(state_old)

            step = change_step_size(step, 0.5)

        else:
            ncrashes = 0
            logger.info("\nSUCCESFULL STEP:")
            if has_dolfin_adjoint:
                try:
                    control_prev = copy(control, deepcopy=True)
                    control_next = copy(control, deepcopy=True)
                except Exception as ex:
                    pass

            target_reached = check_target_reached(problem, control, target)
            if not target_reached:

                if nliter < max_adapt_iter and adapt_step:
                    logger.info("Adapt step size. New step size:")
                    step = change_step_size(step, 1.5)
                    print_control(step)

                control_values.append(copy(control, deepcopy=True))

                if first_step:
                    prev_states.append(problem.state.copy(deepcopy=True))
                else:

                    # Switch place of the state vectors
                    prev_states = [prev_states[-1], prev_states[0]]

                    # Inplace update of last state values
                    prev_states[-1].vector().zero()
                    prev_states[-1].vector().axpy(1.0, problem.state.vector())

