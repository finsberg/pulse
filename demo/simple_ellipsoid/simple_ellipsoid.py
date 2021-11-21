# # Simple ellipsoid
import dolfin
from fenics_plotly import plot

import pulse


try:
    from dolfin_adjoint import Constant, DirichletBC, Function, Mesh, interpolate
except ImportError:
    from dolfin import Function, Constant, DirichletBC, Mesh, interpolate

gamma_space = "R_0"

geometry = pulse.geometries.prolate_ellipsoid_geometry(mesh_size_factor=3.0)

if gamma_space == "regional":
    activation = pulse.RegionalParameter(geometry.cfun)
    target_activation = pulse.dolfin_utils.get_constant(0.2, len(activation))
else:
    activation = Function(dolfin.FunctionSpace(geometry.mesh, "R", 0))
    target_activation = Constant(0.2)

matparams = pulse.HolzapfelOgden.default_parameters()
material = pulse.HolzapfelOgden(
    activation=activation,
    parameters=matparams,
    f0=geometry.f0,
    s0=geometry.s0,
    n0=geometry.n0,
)

# LV Pressure
lvp = Constant(0.0)
lv_marker = geometry.markers["ENDO"][0]
lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
neumann_bc = [lv_pressure]

# Add spring term at the epicardium of stiffness 1.0 kPa/cm^2 to represent pericardium
base_spring = 1.0
robin_bc = [
    pulse.RobinBC(value=Constant(base_spring), marker=geometry.markers["EPI"][0]),
]


# Fix the basal plane in the longitudinal direction
# 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
def fix_basal_plane(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = DirichletBC(
        V.sub(0),
        Constant(0.0),
        geometry.ffun,
        geometry.markers["BASE"][0],
    )
    return bc


dirichlet_bc = (fix_basal_plane,)


# Collect boundary conditions
bcs = pulse.BoundaryConditions(
    dirichlet=dirichlet_bc,
    neumann=neumann_bc,
    robin=robin_bc,
)

# Create the problem
problem = pulse.MechanicsProblem(geometry, material, bcs)

# Solve the problem
pulse.iterate.iterate(problem, (lvp, activation), (1.0, target_activation))

# Get the solution
u, p = problem.state.split(deepcopy=True)

# Move mesh accoring to displacement
u_int = interpolate(u, dolfin.VectorFunctionSpace(geometry.mesh, "CG", 1))
mesh = Mesh(geometry.mesh)
dolfin.ALE.move(mesh, u_int)

fig = plot(geometry.mesh, opacity=0.0, show=False)
fig.add_plot(plot(mesh, color="red", show=False))
fig.show()
