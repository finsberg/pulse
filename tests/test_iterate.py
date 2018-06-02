import pytest
import itertools

from pulse.example_meshes import mesh_paths
from pulse.geometry import Geometry
from pulse.parameters import setup_general_parameters
from pulse.iterate import iterate

try:
    import dolfin_adjoint
    has_dolfin_adjoint = True
except ImportError:
    has_dolfin_adjoint = False

import dolfin
setup_general_parameters()

has_dolfin_adjoint = False
if has_dolfin_adjoint:
    # Run tests with and without annotation
    annotates = (False, True)
else:
    annotates = (False, False)

cases = itertools.product((False, True), annotates)


@pytest.fixture
def problem():

    geometry = Geometry.from_file(mesh_paths['simple_ellipsoid'])
    from utils import make_mechanics_problem
    problem = make_mechanics_problem(geometry)

    return problem


@pytest.mark.parametrize('continuation, annotate', cases)
def test_iterate_pressure(problem, continuation, annotate):

    target_pressure = 1.0
    plv = [p.traction for p in problem.bcs.neumann if p.name == 'lv']
    pressure = {'p_lv': plv[0]}

    if has_dolfin_adjoint:
        dolfin.parameters["adjoint"]["stop_annotating"] = not annotate
        dolfin_adjoint.adj_reset()

    iterate("pressure", problem,
            target_pressure, pressure,
            continuation=continuation)

    if annotate:
        # Check the recording
        assert dolfin_adjoint.replay_dolfin(tol=1e-12)

    # Check that the pressure is correct
    assert float(plv[0]) == target_pressure
    # Check that the state is nonzero
    assert dolfin.norm(problem.state.vector()) > 0


@pytest.mark.parametrize('continuation, annotate', cases)
def test_iterate_gamma(problem, continuation, annotate):

    target_gamma = 0.1
    gamma = problem.material.activation

    if has_dolfin_adjoint:
        dolfin.parameters["adjoint"]["stop_annotating"] = not annotate
        dolfin_adjoint.adj_reset()

    iterate("gamma", problem,
            target_gamma, gamma,
            continuation=continuation)

    assert all(gamma.vector().array() == target_gamma)
    assert dolfin.norm(problem.state.vector()) > 0

    if annotate:
        # dolfin_adjoint.adj_html("active_forward.html", "forward")
        # dolfin_adjoint.adj_html("active_adjoint.html", "adjoint")
        # Check the recording
        assert dolfin_adjoint.replay_dolfin(tol=1e-12)


if __name__ == "__main__":

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

