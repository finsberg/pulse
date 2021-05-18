from unittest import mock

import pytest
from utils import make_mechanics_problem

from pulse import FixedPointUnloader, HeartGeometry, mesh_paths


@pytest.fixture
def problem():
    geo = HeartGeometry.from_file(mesh_paths["simple_ellipsoid"])
    return make_mechanics_problem(geo)


def test_fixedpointunloader(problem):

    unloader = FixedPointUnloader(
        problem=problem, pressure=0.1, options=dict(maxiter=2)
    )

    with mock.patch("pulse.mechanicsproblem.MechanicsProblem.solve") as solve_mock:
        solve_mock.return_value = (1, True)  # (niter, nconv)
        unloader.unload()


if __name__ == "__main__":
    prob = problem()
    test_fixedpointunloader(prob)
