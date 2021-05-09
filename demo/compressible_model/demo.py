r"""
Compressible model
==================

In this demo we show how to make a custom model e.g a compressible
model. The default model in `pulse` is an incompressible model
implemented using a two-field variational approach with Taylor-Hood
finite elements. In this demo we use a pentaly-based compressible
model where the term

.. math::

   \kappa (J \mathrm{ln}J - J + 1)

is added as a penalty to the strain energy denisty function, and we
use :math:`\mathbb{P}1` elements for the displacement

"""
import dolfin

import pulse

# Make sure to use dolfin-adjoint version of object if using dolfin_adjoint
try:
    from dolfin_adjoint import Constant, DirichletBC, Function
except ImportError:
    from dolfin import Function, Constant, DirichletBC

from problem import CompressibleProblem

geometry = pulse.Geometry.from_file(pulse.mesh_paths["simple_ellipsoid"])

activation = Function(dolfin.FunctionSpace(geometry.mesh, "R", 0))
activation.assign(Constant(0.2))
matparams = pulse.HolzapfelOgden.default_parameters()
material = pulse.HolzapfelOgden(
    activation=activation,
    parameters=matparams,
    f0=geometry.f0,
    s0=geometry.s0,
    n0=geometry.n0,
)

# LV Pressure
lvp = Constant(1.0)
lv_marker = geometry.markers["ENDO"][0]
lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
neumann_bc = [lv_pressure]

# Add spring term at the base with stiffness 1.0 kPa/cm^2
base_spring = 1.0
robin_bc = [
    pulse.RobinBC(value=Constant(base_spring), marker=geometry.markers["BASE"][0])
]


# Fix the basal plane in the longitudinal direction
# 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
def fix_basal_plane(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = DirichletBC(
        V.sub(0), Constant(0.0), geometry.ffun, geometry.markers["BASE"][0]
    )
    return bc


dirichlet_bc = [fix_basal_plane]
# You can also use a built in function for this
# from functools import partial
# dirichlet_bc = partial(pulse.mechanicsproblem.dirichlet_fix_base_directional,
#                        ffun=geometry.ffun,
#                        marker=geometry.markers["BASE"][0])

# Collect boundary conditions
bcs = pulse.BoundaryConditions(
    dirichlet=dirichlet_bc, neumann=neumann_bc, robin=robin_bc
)

# Create the problem
problem = CompressibleProblem(geometry, material, bcs)

# Solve the problem
problem.solve()

# Get the solution
u = problem.state
# Dump file that can be viewed in paraview
dolfin.File("displacement.pvd") << u
