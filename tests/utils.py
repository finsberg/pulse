from functools import partial
import dolfin

try:
    from dolfin_adjoint import Constant, Function
except ImportError:
    from dolfin import Constant, Function
    
from pulse.material import HolzapfelOgden, NeoHookean
from pulse.dolfin_utils import RegionalParameter, get_constant
from pulse.utils import get_lv_marker
from pulse.mechanicsproblem import (dirichlet_fix_base,
                                    dirichlet_fix_base_directional,
                                    MechanicsProblem,
                                    BoundaryConditions,
                                    NeumannBC, RobinBC)


def get_boundary_conditions(geometry, pericardium_spring=0.0,
                            base_spring=0.0, base_bc='fix_x'):

    # Neumann BC
    lv_marker = get_lv_marker(geometry)
    lv_pressure = NeumannBC(traction=Constant(0.0, name="lv_pressure"),
                            marker=lv_marker, name='lv')
    neumann_bc = [lv_pressure]

    if 'ENDO_RV' in geometry.markers:

        rv_pressure = NeumannBC(traction=Constant(0.0, name='lv_pressure'),
                                marker=geometry.markers['ENDO_RV'][0],
                                name='rv')

        neumann_bc += [rv_pressure]

    # Robin BC
    if pericardium_spring > 0.0:

        robin_bc = [RobinBC(value=dolfin.Constant(pericardium_spring),
                            marker=geometry.markers["EPI"][0])]

    else:
        robin_bc = []

    # Apply a linear sprint robin type BC to limit motion
    if base_spring > 0.0:
        robin_bc += [RobinBC(value=dolfin.Constant(base_spring),
                             marker=geometry.markers["BASE"][0])]

    # Dirichlet BC
    if base_bc == "fixed":

        dirichlet_bc = [partial(dirichlet_fix_base,
                                ffun=geometry.ffun,
                                marker=geometry.markers["BASE"][0])]

    elif base_bc == 'fix_x':

        dirichlet_bc = [partial(dirichlet_fix_base_directional,
                                ffun=geometry.ffun,
                                marker=geometry.markers["BASE"][0])]
    else:
        raise ValueError("Unknown base bc {}".format(base_bc))

    boundary_conditions = BoundaryConditions(dirichlet=dirichlet_bc,
                                             neumann=neumann_bc,
                                             robin=robin_bc)

    return boundary_conditions


def make_mechanics_problem(geometry):

    # Material = NeoHookean
    Material = HolzapfelOgden
    
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

    bcs = get_boundary_conditions(geometry, base_spring=1.0)

    problem = MechanicsProblem(geometry, material, bcs)

    return problem
