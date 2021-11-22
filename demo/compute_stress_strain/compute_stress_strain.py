# # Compute stress and strain
#
# In this demo we show how to compute stress and strain
import dolfin

try:
    from dolfin_adjoint import Constant, DirichletBC, Function, project
except ImportError:
    from dolfin import DirichletBC, Constant, project, Function

import pulse
from fenics_plotly import plot

gamma_space = "R_0"
geometry = pulse.geometries.prolate_ellipsoid_geometry(mesh_size_factor=3.0)
geometry.mesh.coordinates()[:] *= 0.1
# breakpoint()
activation = Function(dolfin.FunctionSpace(geometry.mesh, "R", 0))

matparams = pulse.HolzapfelOgden.default_parameters()
material = pulse.HolzapfelOgden(
    active_model=pulse.ActiveModel.active_stress,
    T_ref=1.0,  # Total active stress should be activation * T_ref
    eta=0.2,  # Fraction of transverse stress
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

# Add spring term at the base with stiffness 1.0 kPa/cm^2
base_spring = 1.0
robin_bc = [
    pulse.RobinBC(value=Constant(base_spring), marker=geometry.markers["BASE"][0]),
]


def fix_basal_plane(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = DirichletBC(
        V.sub(0),
        Constant(0.0),
        geometry.ffun,
        geometry.markers["BASE"][0],
    )
    return bc


dirichlet_bc = [fix_basal_plane]

# Collect boundary conditions
bcs = pulse.BoundaryConditions(
    dirichlet=dirichlet_bc,
    neumann=neumann_bc,
    robin=robin_bc,
)

# Create the problem
problem = pulse.MechanicsProblem(geometry, material, bcs)

# Solve the problem
pulse.iterate.iterate(problem, lvp, 15.0, max_nr_crash=100, max_iters=100)
pulse.iterate.iterate(problem, activation, 60.0)

V = dolfin.VectorFunctionSpace(geometry.mesh, "CG", 1)
u, p = problem.state.split(deepcopy=True)
# u_int = interpolate(u, V)
# mesh = Mesh(geometry.mesh)
# dolfin.ALE.move(mesh, u_int)

# +
# Get the solution
# u, p = problem.state.split(deepcopy=True)
# dolfin.File("u.pvd") << u
# -

W = dolfin.FunctionSpace(geometry.mesh, "CG", 1)
F = pulse.kinematics.DeformationGradient(u)
E = pulse.kinematics.GreenLagrangeStrain(F)
# Green strain normal to fiber direction
Ef = project(dolfin.inner(E * geometry.f0, geometry.f0), W)
# Save to file for visualization in paraview
# dolfin.File("Ef.pvd") << Ef
plot(Ef)

P = material.FirstPiolaStress(F, p)
# First piola stress normal to fiber direction
Pf = project(dolfin.inner(P * geometry.f0, geometry.f0), W)
# Save to file for visualization in paraview
# dolfin.File("Pf.pvd") << Pf
plot(Pf, colorscale="viridis")

T = material.CauchyStress(F, p)
f = F * geometry.f0
# Cauchy fiber stress
Tf = dolfin.project(dolfin.inner(T * f, f), W)
# Save to file for visualization in paraview
# dolfin.File("Tf.pvd") << Tf
plot(Tf, colorscale="jet")
