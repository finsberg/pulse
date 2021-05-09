"""
Contracting cube
================

In this demo we simulate a unit cube that is fixed
at :math:`x = 0` and free at :math:`x = 1`. We use
a transversally isotropic material with fiber oriented
in the :math:`x`-direction.

The cube is contracting a
"""

import dolfin

try:
    from dolfin_adjoint import (
        Constant,
        DirichletBC,
        Expression,
        UnitCubeMesh,
        interpolate,
    )
except ImportError:
    from dolfin import (
        Constant,
        DirichletBC,
        interpolate,
        Expression,
        UnitCubeMesh,
    )

import pulse

# Create mesh
N = 6
mesh = UnitCubeMesh(N, N, N)


# Create subdomains
class Free(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > (1.0 - dolfin.DOLFIN_EPS) and on_boundary


class Fixed(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < dolfin.DOLFIN_EPS and on_boundary


# Create a facet fuction in order to mark the subdomains
ffun = dolfin.MeshFunction("size_t", mesh, 2)
ffun.set_all(0)

# Mark the first subdomain with value 1
fixed = Fixed()
fixed_marker = 1
fixed.mark(ffun, fixed_marker)

# Mark the second subdomain with value 2
free = Free()
free_marker = 2
free.mark(ffun, free_marker)

# Create a cell function (but we are not using it)
cfun = dolfin.MeshFunction("size_t", mesh, 3)
cfun.set_all(0)


# Collect the functions containing the markers
marker_functions = pulse.MarkerFunctions(ffun=ffun, cfun=cfun)

# Create mictrotructure
V_f = pulse.QuadratureSpace(mesh, 4)

# Fibers
f0 = interpolate(Expression(("1.0", "0.0", "0.0"), degree=1), V_f)
# Sheets
s0 = interpolate(Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
# Fiber-sheet normal
n0 = interpolate(Expression(("0.0", "0.0", "1.0"), degree=1), V_f)

# Collect the mictrotructure
microstructure = pulse.Microstructure(f0=f0, s0=s0, n0=n0)

# Create the geometry
geometry = pulse.Geometry(
    mesh=mesh,
    marker_functions=marker_functions,
    microstructure=microstructure,
)

# Use the default material parameters
material_parameters = pulse.HolzapfelOgden.default_parameters()

# Select model for active contraction
active_model = "active_strain"
# active_model = "active_stress"

# Set the activation
activation = Constant(0.1)

# Create material
material = pulse.HolzapfelOgden(
    active_model=active_model, parameters=material_parameters, activation=activation
)


# Make Dirichlet boundary conditions
def dirichlet_bc(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    return DirichletBC(V, Constant((0.0, 0.0, 0.0)), fixed)


# Make Neumann boundary conditions
neumann_bc = pulse.NeumannBC(traction=Constant(0.0), marker=free_marker)

# Collect Boundary Conditions
bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann_bc,))

# Create problem
problem = pulse.MechanicsProblem(geometry, material, bcs)

# Solve problem
problem.solve()

# Get displacement and hydrostatic pressure
u, p = problem.state.split(deepcopy=True)

# Plot
