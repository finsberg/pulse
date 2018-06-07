import pytest
# import dolfin


from pulse import (FixedPointUnloader, mesh_paths,
                   HeartGeometry, HolzapfelOgden, parameters)

from utils import make_mechanics_problem

parameters['log_level'] = 20

@pytest.fixture
def problem():
    geo = HeartGeometry.from_file(mesh_paths['simple_ellipsoid'])
    return make_mechanics_problem(geo)


@pytest.fixture
def material():
    return HolzapfelOgden()
    

def test_fixedpointunloader(problem):
    
    unloader = FixedPointUnloader(problem=problem,
                                  pressure=1.0)

    unloader.unload()
    
    
    

if __name__ == "__main__":
    prob = problem()
    test_fixedpointunloader(prob)
