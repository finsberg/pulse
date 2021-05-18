import itertools

import numpy as np
import pytest

from pulse.iterate import iterate

try:
    import dolfin_adjoint

    has_dolfin_adjoint = True
except ImportError:
    has_dolfin_adjoint = False

import dolfin
from utils import make_lv_mechanics_problem

has_dolfin_adjoint = False
if has_dolfin_adjoint:
    # Run tests with and without annotation
    annotates = (False, True)
else:
    annotates = (False, False)


cases = itertools.product((False, True), annotates)


@pytest.fixture
def problem():
    problem = make_lv_mechanics_problem("R_0")
    return problem


def test_iterate_pressure(problem):

    target_pressure = 0.1
    plv = [p.traction for p in problem.bcs.neumann if p.name == "lv"]
    pressure = plv[0]

    if has_dolfin_adjoint:
        dolfin.parameters["adjoint"]["stop_annotating"] = False
        dolfin_adjoint.adj_reset()

    iterate(problem, pressure, target_pressure)

    if has_dolfin_adjoint:
        # Check the recording
        assert dolfin_adjoint.replay_dolfin(tol=1e-12)

    # Check that the pressure is correct
    assert float(plv[0]) == target_pressure
    # Check that the state is nonzero
    assert dolfin.norm(problem.state.vector()) > 0


def test_iterate_gamma(problem):

    target_gamma = 0.001
    gamma = problem.material.activation

    if has_dolfin_adjoint:
        dolfin.parameters["adjoint"]["stop_annotating"] = False
        dolfin_adjoint.adj_reset()

    iterate(problem, gamma, target_gamma)

    assert all(gamma.vector().get_local() == target_gamma)
    assert dolfin.norm(problem.state.vector()) > 0

    if has_dolfin_adjoint:
        assert dolfin_adjoint.replay_dolfin(tol=1e-12)


def test_iterate_gamma_regional():
    problem = make_lv_mechanics_problem("regional")
    target_gamma = problem.material.activation.vector().get_local()
    for i in range(len(target_gamma)):
        target_gamma[i] = 0.01 - i * 0.001
    print(target_gamma)

    gamma = problem.material.activation

    if has_dolfin_adjoint:
        dolfin.parameters["adjoint"]["stop_annotating"] = False
        dolfin_adjoint.adj_reset()

    iterate(problem, gamma, target_gamma)

    print(gamma.vector().get_local())
    assert np.all(gamma.vector().get_local() - target_gamma < 1e-12)
    assert dolfin.norm(problem.state.vector()) > 0

    if has_dolfin_adjoint:
        assert dolfin_adjoint.replay_dolfin(tol=1e-12)


@pytest.mark.parametrize("continuation", [True, False])
def test_iterate_gamma_cg1(continuation):

    problem = make_lv_mechanics_problem("CG_1")
    V = problem.material.activation.function_space()
    target_gamma = dolfin.interpolate(dolfin.Expression("0.01 * x[0]", degree=1), V)
    gamma = problem.material.activation

    if has_dolfin_adjoint:
        dolfin.parameters["adjoint"]["stop_annotating"] = False
        dolfin_adjoint.adj_reset()

    iterate(problem, gamma, target_gamma, continuation=continuation)

    assert np.all(
        gamma.vector().get_local() - target_gamma.vector().get_local() < 1e-12
    )
    assert dolfin.norm(problem.state.vector()) > 0

    if has_dolfin_adjoint:
        assert dolfin_adjoint.replay_dolfin(tol=1e-12)


def test_iterate_gamma_pressure(problem):
    target_pressure = 1.0
    plv = [p.traction for p in problem.bcs.neumann if p.name == "lv"]
    pressure = plv[0]

    target_gamma = 0.01
    gamma = problem.material.activation

    if has_dolfin_adjoint:
        dolfin.parameters["adjoint"]["stop_annotating"] = False
        dolfin_adjoint.adj_reset()

    iterate(problem, (pressure, gamma), (target_pressure, target_gamma))

    assert all(gamma.vector().get_local() == target_gamma)
    assert float(plv[0]) == target_pressure
    assert dolfin.norm(problem.state.vector()) > 0

    if has_dolfin_adjoint:
        assert dolfin_adjoint.replay_dolfin(tol=1e-12)


def test_iterate_regional_gamma_pressure():

    problem = make_lv_mechanics_problem("regional")
    target_gamma = problem.material.activation.vector().get_local()
    for i in range(len(target_gamma)):
        target_gamma[i] = 0.01 - i * 0.001
    gamma = problem.material.activation

    target_pressure = 0.1
    plv = [p.traction for p in problem.bcs.neumann if p.name == "lv"]
    pressure = plv[0]

    with pytest.raises(ValueError):
        iterate(problem, (pressure, gamma), (target_pressure, target_gamma))


if __name__ == "__main__":

    prob = problem()
    # test_iterate_pressure(prob)
    test_iterate_gamma(prob)
    # test_iterate_gamma_regional()
    # test_iterate_gamma_cg1(True)
    # test_iterate_gamma_pressure(prob)
    # test_iterate_regional_gamma_pressure()
    exit()

    for c, a in cases:
        print("Continuation = {}, annotate = {}".format(c, a))
        prob = problem()
        test_iterate_pressure(prob, continuation=c, annotate=a)
        if has_dolfin_adjoint:
            dolfin_adjoint.adj_reset()

        prob = problem()
        test_iterate_gamma(prob, continuation=c, annotate=a)
        if has_dolfin_adjoint:
            dolfin_adjoint.adj_reset()
