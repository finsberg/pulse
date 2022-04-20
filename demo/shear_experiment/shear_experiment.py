# # Shear experiment
# Attempt to reproduce Figure 7 in [1].
#
#
# > [1] Holzapfel, Gerhard A., and Ray W. Ogden.
#     "Constitutive modelling of passive myocardium:
#     a structurally based framework for material characterization.
#     "Philosophical Transactions of the Royal Society of London A:
#     Mathematical, Physical and Engineering Sciences 367.1902 (2009): 3445-3475.
#

from __future__ import annotations
from pathlib import Path
import dolfin
import matplotlib.pyplot as plt
import numpy as np
import ufl
import typing

import pulse
from pulse.mechanicsproblem import MechanicsProblem

try:
    from dolfin_adjoint import (
        Constant,
        DirichletBC,
        Expression,
        UnitCubeMesh,
        interpolate,
        Function,
    )
except ImportError:
    from dolfin import (
        Constant,
        DirichletBC,
        interpolate,
        Expression,
        UnitCubeMesh,
        Function,
    )


class Experiment(typing.NamedTuple):
    problem: MechanicsProblem
    increment: typing.Callable[[float], typing.Tuple[float, float, float]]
    shear_component: typing.Callable[[ufl.tensors.ComponentTensor], float]


# Create mesh
N = 2
mesh = UnitCubeMesh(N, N, N)


# Create a facet fuction in order to mark the subdomains

ffun = dolfin.MeshFunction("size_t", mesh, 2)
ffun.set_all(0)

# Mark subdomains

xlow = dolfin.CompiledSubDomain("near(x[0], 0) && on_boundary")
xlow_marker = 1
xlow.mark(ffun, xlow_marker)

xhigh = dolfin.CompiledSubDomain("near(x[0], 1.0) && on_boundary")
xhigh_marker = 2
xhigh.mark(ffun, xhigh_marker)

ylow = dolfin.CompiledSubDomain("near(x[1], 0) && on_boundary")
ylow_marker = 3
ylow.mark(ffun, ylow_marker)

yhigh = dolfin.CompiledSubDomain("near(x[1], 1) && on_boundary")
yhigh_marker = 4
yhigh.mark(ffun, yhigh_marker)

zlow = dolfin.CompiledSubDomain("near(x[2], 0) && on_boundary")
zlow_marker = 5
zlow.mark(ffun, zlow_marker)

zhigh = dolfin.CompiledSubDomain("near(x[2], 1) && on_boundary")
zhigh_marker = 6
zhigh.mark(ffun, zhigh_marker)


# Collect the functions containing the markers

marker_functions = pulse.MarkerFunctions(ffun=ffun)


# Create mictrotructure

V_f = dolfin.VectorFunctionSpace(mesh, "CG", 1)

# Fibers, sheets and fiber-sheet normal

f0 = interpolate(Expression(("1.0", "0.0", "0.0"), degree=1), V_f)
s0 = interpolate(Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
n0 = interpolate(Expression(("0.0", "0.0", "1.0"), degree=1), V_f)

# Collect the mictrotructure

microstructure = pulse.Microstructure(f0=f0, s0=s0, n0=n0)

# Create the geometry

geometry = pulse.Geometry(
    mesh=mesh,
    marker_functions=marker_functions,
    microstructure=microstructure,
)

# Use the default material parameters from the paper

material_parameters = {
    "a": 0.059,
    "b": 8.023,
    "a_f": 18.472,
    "b_f": 16.026,
    "a_s": 2.481,
    "b_s": 11.120,
    "a_fs": 0.216,
    "b_fs": 11.436,
}

# Create material

material = pulse.HolzapfelOgden(parameters=material_parameters)

# Make a space for controling the amount of shear displacement

X_space = dolfin.VectorFunctionSpace(mesh, "R", 0)
x = Function(X_space)
zero = Constant((0.0, 0.0, 0.0))


# Make a method that will return


def create_experiment(case):  # noqa: C901

    if case == "fs":

        def dirichlet_bc(W):
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            return [
                DirichletBC(V, zero, xlow),
                DirichletBC(V, x, xhigh),
            ]

        def increment(xi):
            return (0, xi, 0)

        def shear_component(T):
            return dolfin.assemble(T[0, 1] * dolfin.dx)

    elif case == "fn":

        def dirichlet_bc(W):
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            return [
                DirichletBC(V, zero, xlow),
                DirichletBC(V, x, xhigh),
            ]

        def increment(xi):
            return (0, 0, xi)

        def shear_component(T):
            return dolfin.assemble(T[0, 2] * dolfin.dx)

    elif case == "sf":

        def dirichlet_bc(W):
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            return [
                DirichletBC(V, zero, ylow),
                DirichletBC(V, x, yhigh),
            ]

        def increment(xi):
            return (xi, 0, 0)

        def shear_component(T):
            return dolfin.assemble(T[1, 0] * dolfin.dx)

    elif case == "sn":

        def dirichlet_bc(W):
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            return [
                DirichletBC(V, zero, ylow),
                DirichletBC(V, x, yhigh),
            ]

        def increment(xi):
            return (0, 0, xi)

        def shear_component(T):
            return dolfin.assemble(T[1, 2] * dolfin.dx)

    elif case == "nf":

        def dirichlet_bc(W):
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            return [
                DirichletBC(V, zero, zlow),
                DirichletBC(V, x, zhigh),
            ]

        def increment(xi):
            return (xi, 0, 0)

        def shear_component(T):
            return dolfin.assemble(T[2, 0] * dolfin.dx)

    elif case == "ns":

        def dirichlet_bc(W):
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            return [
                DirichletBC(V, zero, zlow),
                DirichletBC(V, x, zhigh),
            ]

        def increment(xi):
            return (0, xi, 0)

        def shear_component(T):
            return dolfin.assemble(T[2, 1] * dolfin.dx)

    else:
        raise ValueError(f"Unknown case {case}")

    # Collect Boundary Conditions
    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,))

    # Create problem
    return Experiment(
        problem=pulse.MechanicsProblem(geometry, material, bcs),
        increment=increment,
        shear_component=shear_component,
    )


# Loop over all modes and collect the stress values

modes = ["fs", "fn", "sf", "sn", "nf", "ns"]
stress: dict[str, dict[str, list[float]]] = {
    "cauchy": {m: [] for m in modes},
    "pk1": {m: [] for m in modes},
}
shear_values = np.linspace(0, 0.6, 10)
recompute = True
# Solve problem
results_file = Path("result.npy")
if recompute or not results_file.is_file():
    for mode in modes:
        x.assign(zero)
        experiment = create_experiment(mode)

        for shear in shear_values:
            pulse.iterate.iterate(
                experiment.problem,
                x,
                experiment.increment(shear),
                reinit_each_step=True,
            )
            stress["cauchy"][mode].append(
                experiment.shear_component(experiment.problem.ChachyStress()),
            )
            stress["pk1"][mode].append(
                experiment.shear_component(experiment.problem.FirstPiolaStress()),
            )

    np.save(results_file, stress)

# Plot results

stress = np.load(results_file, allow_pickle=True).item()
fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
for mode, values in stress["cauchy"].items():
    ax[0].plot(shear_values, stress["cauchy"][mode], label=mode)
ax[0].set_title("Cauchy Stress")
for mode, values in stress["pk1"].items():
    ax[1].plot(shear_values, stress["pk1"][mode], label=mode)
ax[1].set_title("First Piola Kirchhoff Stress")
ax[0].set_ylabel("Shear stress (kPa)")
for axi in ax:
    axi.set_xlabel("Amount of shear")
    axi.set_ylim((0, 16))
    axi.grid()
    axi.legend()
plt.show()
