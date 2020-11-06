import pytest
from utils import make_mechanics_problem

from pulse import FixedPointUnloader, HeartGeometry, mesh_paths, parameters

parameters["log_level"] = 20


@pytest.fixture
def problem():
    geo = HeartGeometry.from_file(mesh_paths["simple_ellipsoid"])
    return make_mechanics_problem(geo)


def test_fixedpointunloader(problem):

    unloader = FixedPointUnloader(problem=problem, pressure=1.0)

    unloader.unload()


if __name__ == "__main__":
    prob = problem()
    test_fixedpointunloader(prob)
