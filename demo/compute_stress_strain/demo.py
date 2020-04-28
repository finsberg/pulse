import dolfin
import pulse
import sys


gamma_space = "R_0"

geometry = pulse.Geometry.from_file(pulse.mesh_paths["simple_ellipsoid"])
activation = dolfin.Function(dolfin.FunctionSpace(geometry.mesh, "R", 0))

matparams = pulse.HolzapfelOgden.default_parameters()
material = pulse.HolzapfelOgden(
    active_model="active_stress",
    T_ref=1.0,  # Total active stress should be activation * T_ref
    eta=0.2,  # Fraction of transverse stress
    activation=activation,
    parameters=matparams,
    f0=geometry.f0,
    s0=geometry.s0,
    n0=geometry.n0,
)

# LV Pressure
lvp = dolfin.Constant(0.0)
lv_marker = geometry.markers["ENDO"][0]
lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
neumann_bc = [lv_pressure]

# Add spring term at the base with stiffness 1.0 kPa/cm^2
base_spring = 1.0
robin_bc = [
    pulse.RobinBC(
        value=dolfin.Constant(base_spring), marker=geometry.markers["BASE"][0]
    )
]


def fix_basal_plane(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = dolfin.DirichletBC(
        V.sub(0), dolfin.Constant(0.0), geometry.ffun, geometry.markers["BASE"][0]
    )
    return bc


dirichlet_bc = [fix_basal_plane]

# Collect boundary conditions
bcs = pulse.BoundaryConditions(
    dirichlet=dirichlet_bc, neumann=neumann_bc, robin=robin_bc
)

# Create the problem
problem = pulse.MechanicsProblem(geometry, material, bcs)

# Solve the problem
pulse.iterate.iterate(problem, lvp, 15.0)
pulse.iterate.iterate(problem, activation, 60.0)

# Get the solution
u, p = problem.state.split(deepcopy=True)
dolfin.File("u.pvd") << u

F = pulse.kinematics.DeformationGradient(u)
E = pulse.kinematics.GreenLagrangeStrain(F)
# Green strain normal to fiber direction
Ef = dolfin.project(
    dolfin.inner(E * geometry.f0, geometry.f0),
    dolfin.FunctionSpace(geometry.mesh, "CG", 1),
    # solver_type="gmres",
)
# Save to file for visualization in paraview
dolfin.File("Ef.pvd") << Ef

P = material.FirstPiolaStress(F, p)
# First piola stress normal to fiber direction
Pf = dolfin.project(
    dolfin.inner(P * geometry.f0, geometry.f0),
    dolfin.FunctionSpace(geometry.mesh, "CG", 1),
)
# Save to file for visualization in paraview
dolfin.File("Pf.pvd") << Pf

T = material.CauchyStress(F, p)
f = F * geometry.f0
# Cauchy fiber stress
Tf = dolfin.project(
    dolfin.inner(T * f, f), dolfin.FunctionSpace(geometry.mesh, "CG", 1)
)
# Save to file for visualization in paraview
dolfin.File("Tf.pvd") << Tf
