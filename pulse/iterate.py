#!/usr/bin/env python
import operator as op

import dolfin
import numpy as np

from . import Constant
from . import Constants
from . import Function
from . import Functions
from . import has_dolfin_adjoint
from . import numpy_mpi
from .dolfin_utils import get_constant
from .mechanicsproblem import SolverDidNotConverge
from .utils import delist
from .utils import enlist
from .utils import Enlisted
from .utils import getLogger
from .utils import value_size

logger = getLogger(__name__)


def copy(f, deepcopy=True, name="copied_function"):
    """
    Copy a function. This is to ease the integration
    with dolfin adjoint where copied fuctions are annotated.
    """

    if isinstance(f, Functions):
        if has_dolfin_adjoint:
            try:
                return f.copy(deepcopy=deepcopy, name=name)
            except TypeError:
                return f.copy(deepcopy=deepcopy)
        else:
            return f.copy(deepcopy=deepcopy)
    elif isinstance(f, dolfin.Constant):
        return dolfin.Constant(f, name=name)
    elif isinstance(f, Constant):
        return Constant(f, name=name)
    elif isinstance(f, (float, int)):
        return f
    elif isinstance(f, Enlisted):
        lst = []
        for fi in f:
            lst.append(copy(fi))
        return enlist(lst)

    elif isinstance(f, (list, tuple)):
        lst = []
        for fi in f:
            lst.append(copy(fi))
        return tuple(lst)
    else:
        return f


def constant2float(const):
    """
    Convert a :class:`dolfin.Constant` to float
    """
    try:
        c = float(const)
    except Exception:
        const = get_constant(const)
        try:
            c = float(const)
        except TypeError:
            try:
                c = np.zeros(len(const))
                const.eval(c, c)
            except Exception as ex:
                logger.warning(ex, exc_info=True)
                return const
    return c


def get_delta(new_control, c0, c1):
    """
    Get extrapolation parameter used in the continuation step.
    """
    if isinstance(c0, Constants):
        c0 = constant2float(c0)
        c1 = constant2float(c1)
        new_control = constant2float(new_control)

    if isinstance(new_control, (int, float)):
        delta = (new_control - c0) / float(c1 - c0)

    elif isinstance(new_control, (tuple, np.ndarray, list)):
        c0 = [constant2float(c) for c in c0]
        c1 = [constant2float(c) for c in c1]
        new_control = [constant2float(c) for c in new_control]
        delta = (new_control[0] - c0[0]) / float(c1[0] - c0[0])

    elif isinstance(new_control, (dolfin.GenericVector, dolfin.Vector)):
        new_control_arr = numpy_mpi.gather_vector(new_control)
        c0_arr = numpy_mpi.gather_vector(c0)
        c1_arr = numpy_mpi.gather_vector(c1)
        delta = (new_control_arr[0] - c0_arr[0]) / float(c1_arr[0] - c0_arr[0])

    elif isinstance(new_control, (dolfin.Function, Function)):
        new_control_arr = numpy_mpi.gather_vector(
            new_control.vector(),
            new_control.function_space().dim(),
        )
        c0_arr = numpy_mpi.gather_vector(c0.vector(), c0.function_space().dim())
        c1_arr = numpy_mpi.gather_vector(c1.vector(), c1.function_space().dim())
        delta = (new_control_arr[0] - c0_arr[0]) / float(c1_arr[0] - c0_arr[0])

    else:
        msg = ("Unexpected type for new_crontrol in get_delta" "Got {}").format(
            type(new_control),
        )
        raise TypeError(msg)

    return squeeze(delta)


def np2str(c_, fmt="{:.3f}"):
    c = delist(c_)
    if isinstance(c, (np.ndarray, list, tuple)):
        return ", ".join([f"{ci:.3f}" for ci in c])
    return f"{c:.3f}"


def print_control(cs, msg):
    controls = [constant2float(c) for c in cs]

    if len(controls) > 3:
        msg += ("\n\tMin:{:.2f}\tMean:{:.2f}\tMax:{:.2f}" "").format(
            np.min(controls),
            np.mean(controls),
            np.max(controls),
        )
    else:
        cs = []
        for c in controls:
            if hasattr(c, "__len__"):
                msg += print_control(c, msg)
            else:
                cs.append(c)
        if cs:
            msg += ",".join([np2str(c) for c in controls])
    return msg


def get_diff(current, target):
    """
    Get difference between current and target value
    """
    if isinstance(target, (Function, dolfin.Function)):
        diff = target.vector() - current.vector()

    elif isinstance(target, Constants):
        diff = np.subtract(constant2float(target), constant2float(current))
    elif isinstance(target, (tuple, list)):
        diff = np.subtract(
            [constant2float(t) for t in target],
            [constant2float(c) for c in current],
        )
    else:
        try:
            diff = np.subtract(target, current)
        except Exception as ex:
            logger.error(ex)
            raise ValueError(
                ("Unable to compute diff with type {}" "").format(type(current)),
            )

    return squeeze(diff)


def squeeze(x):

    try:
        y = np.squeeze(x)
    except Exception:
        return x
    else:
        try:
            shape = np.shape(y)
        except Exception:
            return y
        else:
            if len(shape) == 0:
                return float(y)
            else:
                return y


def get_initial_step(current, target, nsteps=5):
    """
    Estimate the step size needed to step from current to target
    in `nsteps`.
    """

    diff = get_diff(current, target)
    if isinstance(diff, dolfin.GenericVector):
        step = Function(current.function_space())
        step.vector().axpy(1.0 / float(nsteps), diff)

    else:
        step = diff / float(nsteps)

    logger.debug(
        ("Intial number of steps: {} with step size {}" "").format(nsteps, step),
    )
    return step


def step_too_large(current, target, step):
    """
    Check if `current + step` exceeds `target`
    """
    if isinstance(target, (dolfin.Function, Function)):
        target = numpy_mpi.gather_vector(target.vector(), target.function_space().dim())
    elif isinstance(target, Constants):
        target = constant2float(target)
    target = squeeze(target)

    if isinstance(current, (dolfin.Function, Function)):
        current = numpy_mpi.gather_vector(
            current.vector(),
            current.function_space().dim(),
        )
    elif isinstance(current, Constants):
        current = constant2float(current)
    current = squeeze(current)

    if isinstance(step, (dolfin.Function, Function)):
        step = numpy_mpi.gather_vector(step.vector(), step.function_space().dim())
    elif isinstance(step, Constants):
        step = constant2float(step)
    step = squeeze(step)

    if not hasattr(target, "__len__"):
        cond = current < target
        if hasattr(cond, "__len__"):
            cond = np.any(cond)

        comp = op.gt if cond else op.lt
        return np.any(comp(current + step, target))
    else:
        too_large = []
        for (c, t, s) in zip(current, target, step):
            too_large.append(step_too_large(c, t, s))

    return np.any(too_large)


def iterate(
    problem,
    control,
    target,
    continuation=True,
    max_adapt_iter=8,
    adapt_step=True,
    old_states=None,
    old_controls=None,
    max_nr_crash=20,
    max_iters=40,
    initial_number_of_steps=5,
    reinit_each_step=False,
):

    """
    Using the given problem, iterate control to given target.

    Arguments
    ---------
    problem : pulse.MechanicsProblem
        The problem
    control : dolfin.Function or dolfin.Constant
        The control
    target: dolfin.Function, dolfin.Constant, tuple or float
        The target value. Typically a float if target is LVP, a tuple
        if target is (LVP, RVP) and a function if target is gamma.
    continuation: bool
        Apply continuation for better guess for newton problem
        Note: Replay test seems to fail when continuation is True,
        but taylor test passes
    max_adapt_iter: int
        If number of iterations is less than this number and adapt_step=True,
        then adapt control step. Default: 8
    adapt_step: bool
        Adapt / increase step size when sucessful iterations are achevied.
    old_states: list
        List of old controls to help speed in the continuation
    reinit_each_step : bool
        If True reinitialize form at each step.
    """

    with Iterator(
        problem=problem,
        control=control,
        target=target,
        continuation=continuation,
        max_adapt_iter=max_adapt_iter,
        adapt_step=adapt_step,
        old_states=old_states,
        old_controls=old_controls,
        max_nr_crash=max_nr_crash,
        max_iters=max_iters,
        initial_number_of_steps=initial_number_of_steps,
        reinit_each_step=reinit_each_step,
    ) as iterator:
        res = iterator.solve()

    return res


class Iterator(object):
    """
    Iterator
    """

    _control_types = (Function, dolfin.Function, Constant, dolfin.Constant)

    def __init__(
        self, problem, control, target, old_states=None, old_controls=None, **params
    ):
        self.parameters = Iterator.default_parameters()
        self.parameters.update(params)

        self.old_controls = () if old_controls is None else old_controls
        self.old_states = () if old_states is None else old_states
        self.problem = problem

        self._check_control(control)
        self._check_target(target)

        self.control_values = [
            copy(delist(self.control), deepcopy=True, name="previous control"),
        ]
        self.prev_states = [
            copy(self.problem.state, deepcopy=True, name="previous state"),
        ]

        self.step = get_initial_step(
            self.control,
            self.target,
            self.parameters["initial_number_of_steps"],
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    @staticmethod
    def default_parameters():
        return dict(
            continuation=True,
            max_adapt_iter=8,
            adapt_step=True,
            max_nr_crash=20,
            max_iters=40,
            initial_number_of_steps=5,
            reinit_each_step=False,
        )

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        s = squeeze(step)
        if hasattr(s, "__len__") and len(s) != len(self.control):
            if len(self.control) == 1:
                # Then probably the control is a function with
                # dimension higher than one and we have squeezed to much
                s = enlist(s, force_enlist=True)
            else:
                msg = (
                    "Step is of lenght {} while the lenght " "of the control is {}"
                ).format(len(s), len(self.control))
                raise ValueError(msg)
        else:
            s = enlist(s)

        self._step = s

    def solve(self):

        self.ncrashes = 0
        self.niters = 0
        # print(type(self.control))
        control_name = self.control[0].name()
        logger.info(f"Iterating to target control ({control_name})...")
        msg = f"Current control: {control_name} = "
        logger.info(print_control(self.control, msg))
        msg = "Target: "
        logger.info(print_control(self.target, msg))
        while not self.target_reached():

            self.niters += 1
            if (
                self.ncrashes > self.parameters["max_nr_crash"]
                or self.niters > self.parameters["max_iters"]
            ):

                self.problem.reinit(self.prev_states[0])
                self.assign_control(enlist(self.control_values[0]))

                raise SolverDidNotConverge

            prev_state = self.prev_states[-1]
            prev_control = enlist(self.control_values[-1])

            # Check if we are close
            if step_too_large(
                delist(prev_control),
                delist(self.target),
                delist(self.step),
            ):
                self.change_step_for_final_iteration(delist(prev_control))

            self.increment_control()

            if self.parameters["continuation"]:
                self.continuation_step()

            logger.debug("Try new control")
            self.print_control()

            try:
                nliter, nlconv = self.problem.solve()

            except SolverDidNotConverge as ex:
                logger.debug(ex)
                logger.debug("\nNOT CONVERGING")
                logger.debug("Reduce control step")
                self.ncrashes += 1
                self.assign_control(prev_control)
                # Assign old state
                logger.debug("Assign old state")
                self.problem.state.vector().zero()
                self.problem.reinit(prev_state)

                self.change_step_size(0.5)

            else:
                self.ncrashes = 0
                logger.debug("\nSUCCESFULL STEP:")

                if not self.target_reached():
                    if (
                        nliter < self.parameters["max_adapt_iter"]
                        and self.parameters["adapt_step"]
                    ):
                        self.change_step_size(1.5)
                        msg = f"Adapt step size. New step size: {np2str(self.step)}"
                        # print_control(enlist(self.step), msg)
                        logger.debug(msg)

                self.control_values.append(
                    delist(copy(self.control, deepcopy=True, name="Previous control")),
                )

                self.prev_states.append(
                    copy(self.problem.state, deepcopy=True, name="Previous state"),
                )
        return self.prev_states, self.control_values

    def change_step_size(self, factor):
        step = delist(self.step)
        if isinstance(step, (list, tuple)):
            self.step = [factor * s for s in step]
        else:
            self.step = factor * step

    def print_control(self):
        msg = "Current control: "
        msg = print_control(self.control, msg)
        logger.debug(msg)

    def continuation_step(self):

        first_step = len(self.prev_states) < 2
        if first_step:
            return

        c0, c1 = self.control_values[-2:]
        s0, s1 = self.prev_states[-2:]

        delta = get_delta(delist(self.control), c0, c1)

        if has_dolfin_adjoint:
            w = dolfin.Function(self.problem.state.function_space())

            w.vector().zero()
            w.vector().axpy(1.0 - delta, s0.vector())
            w.vector().axpy(delta, s1.vector())
            self.problem.reinit(w, annotate=True)
        else:
            self.problem.state.vector().zero()
            self.problem.state.vector().axpy(1.0 - delta, s0.vector())
            self.problem.state.vector().axpy(delta, s1.vector())

    def increment_control(self):

        for c, s in zip(self.control, self.step):
            # if isinstance(s, (dolfin.Function, Function))
            if isinstance(c, (dolfin.Function, Function)):
                c_arr = numpy_mpi.gather_vector(c.vector(), c.function_space().dim())
                c_tmp = Function(c.function_space())
                c_tmp.vector().set_local(np.array(c_arr + s))
                c_tmp.vector().apply("")
                c.assign(c_tmp)
            else:
                c_arr = c
                c.assign(Constant(constant2float(c) + s))

        if self.parameters["reinit_each_step"]:
            self.problem._init_forms()

    def assign_control(self, new_control):
        """
        Assign a new value to the control
        """
        for c, n in zip(self.control, new_control):
            try:
                c.assign(n)
            except TypeError:
                c.assign(Constant(n))

    def change_step_for_final_iteration(self, prev_control):
        """Change step size so that target is
        reached in the next iteration
        """
        logger.debug("Change step size for final iteration")

        target = delist(self.target)
        prev_control = delist(prev_control)

        if isinstance(target, (dolfin.Function, Function)):
            step = Function(target.function_space())
            step.vector().axpy(1.0, target.vector())
            step.vector().axpy(-1.0, prev_control.vector())
        elif isinstance(target, (list, np.ndarray, tuple)):
            if isinstance(prev_control, (dolfin.Function, Function)):
                prev = numpy_mpi.gather_vector(
                    prev_control.vector(),
                    prev_control.function_space().dim(),
                )
            else:
                prev = prev_control

            step = np.array(
                [constant2float(t) - constant2float(c) for (t, c) in zip(target, prev)],
            )
        elif isinstance(target, (dolfin.Constant, Constant)):
            step = constant2float(target) - constant2float(prev_control)
        else:
            step = target - prev_control

        self.step = step

    def _check_target(self, target):

        target = enlist(target)

        targets = []
        for tar in target:

            try:
                t = get_constant(tar)
            except TypeError:
                msg = ("Unable to convert target for type {} " "to a constant").format(
                    type(target),
                )
                raise TypeError(msg)
            targets.append(t)

        self.target = Enlisted(targets)
        t0 = self.target[0]
        msg = (
            "Unsuppoert shape of control and target. "
            "Can only handle single arrays or multiple scalars."
        )
        for t in self.target[1:]:
            if value_size(t0) > 1 or value_size(t) > 1:
                raise (ValueError(msg))
        logger.debug(f"Target: {[constant2float(t) for t in self.target]}")

    def _check_control(self, control):

        control = enlist(control)
        # Control has to be either a function or
        # a constant
        for c in control:
            msg = ("Expected control parameters to be of type {}, " "got {}").format(
                self._control_types,
                type(c),
            )
            assert isinstance(c, self._control_types), msg

        self.control = control
        logger.debug(f"Control: {[constant2float(c) for c in self.control]}")

    @property
    def ncontrols(self):
        """Number of controls"""
        return len(self.control)

    def target_reached(self):
        """Check if control and target are the same"""
        diff = get_diff(self.control, self.target)

        if isinstance(diff, dolfin.GenericVector):
            diff.abs()
            max_diff = diff.max()

        else:

            max_diff = np.max(abs(diff))

        reached = max_diff < 1e-6
        if reached:
            logger.debug("Check target reached: YES!")
        else:
            logger.debug("Check target reached: NO")
            logger.debug(f"Maximum difference: {max_diff:.3e}")

        return reached
