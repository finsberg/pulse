import dolfin

try:
    from dolfin_adjoint import Constant, Function
except ImportError:
    from dolfin import Constant, Function
    
from pulse.material import HolzapfelOgden, NeoHookean
from pulse.dolfin_utils import RegionalParameter, get_constant
from pulse.utils import get_lv_marker
from pulse.mechanicsproblem import (MechanicsProblem,
                                    cardiac_boundary_conditions)


def make_mechanics_problem(geometry):

    # Material = NeoHookean
    Material = HolzapfelOgden

    # activation = Constant(0.0)
    activation = Function(dolfin.FunctionSpace(geometry.mesh, "R", 0))
    matparams = Material.default_parameters()
    # mu = RegionalParameter(geometry.cfun)
    # mu_val = get_constant(mu.value_size(), value_rank=0, val=matparams["mu"])
    # mu.assign(mu_val)
    # matparams["mu"] = mu

    material = Material(activation=activation,
                        parameters=matparams,
                        f0=geometry.f0,
                        s0=geometry.s0,
                        n0=geometry.n0)

    #bcs = cardiac_boundary_conditions(geometry, base_spring=1.0)
    bcs_parameters = MechanicsProblem.default_bcs_parameters()
    bcs_parameters['base_spring'] = 1.0
    bcs_parameters['base_bc'] = 'fix_x'
    

    problem = MechanicsProblem(geometry, material,
                               bcs_parameters=bcs_parameters)

    return problem
