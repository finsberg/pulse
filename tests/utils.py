import dolfin

try:
    from dolfin_adjoint import Function
except ImportError:
    from dolfin import Function

from pulse.dolfin_utils import RegionalParameter
from pulse.example_meshes import mesh_paths
from pulse.geometry import HeartGeometry
from pulse.material import NeoHookean
from pulse.mechanicsproblem import MechanicsProblem


def make_lv_mechanics_problem(space="R_0"):

    geometry = HeartGeometry.from_file(mesh_paths["simple_ellipsoid"])
    return make_mechanics_problem(geometry, space)


def make_mechanics_problem(geometry, space="R_0"):

    Material = NeoHookean
    # Material = HolzapfelOgden

    if space == "regional":
        activation = RegionalParameter(geometry.cfun)
    else:
        family, degree = space.split("_")
        activation = Function(dolfin.FunctionSpace(geometry.mesh, family, int(degree)))

    matparams = Material.default_parameters()
    # mu = RegionalParameter(geometry.cfun)
    # mu_val = get_constant(mu.value_size(), value_rank=0, val=matparams["mu"])
    # mu.assign(mu_val)
    # matparams["mu"] = mu

    material = Material(
        activation=activation,
        parameters=matparams,
        f0=geometry.f0,
        s0=geometry.s0,
        n0=geometry.n0,
    )

    # bcs = cardiac_boundary_conditions(geometry, base_spring=1.0)
    bcs_parameters = MechanicsProblem.default_bcs_parameters()
    bcs_parameters["base_spring"] = 1.0
    bcs_parameters["base_bc"] = "fix_x"

    problem = MechanicsProblem(geometry, material, bcs_parameters=bcs_parameters)

    return problem
