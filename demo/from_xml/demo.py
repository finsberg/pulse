import dolfin

try:
    from dolfin_adjoint import Constant, DirichletBC, Function, Mesh, interpolate
except ImportError:
    from dolfin import Mesh, Function, Constant, DirichletBC, interpolate

import json

import pulse

# Mesh
mesh = Mesh(pulse.utils.mpi_comm_world(), "data/mesh.xml")

# Marker functions
facet_function = dolfin.MeshFunction("size_t", mesh, "data/facet_function.xml")
marker_functions = pulse.MarkerFunctions(ffun=facet_function)

# Markers
with open("data/markers.json", "r") as f:
    markers = json.load(f)


# Fiber
fiber_element = dolfin.VectorElement(
    family="Quadrature", cell=mesh.ufl_cell(), degree=4, quad_scheme="default"
)
fiber_space = dolfin.FunctionSpace(mesh, fiber_element)
fiber = Function(fiber_space, "data/fiber.xml")

microstructure = pulse.Microstructure(f0=fiber)


# Create the geometry
geometry = pulse.HeartGeometry(
    mesh=mesh,
    markers=markers,
    marker_functions=marker_functions,
    microstructure=microstructure,
)

activation = Function(dolfin.FunctionSpace(geometry.mesh, "R", 0))
activation.assign(Constant(0.2))
matparams = pulse.HolzapfelOgden.default_parameters()
material = pulse.HolzapfelOgden(
    activation=activation, parameters=matparams, f0=geometry.f0
)

# LV Pressure
lvp = Constant(1.0)
lv_marker = markers["ENDO"][0]
lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
neumann_bc = [lv_pressure]

# Add spring term at the base with stiffness 1.0 kPa/cm^2
base_spring = 1.0
robin_bc = [pulse.RobinBC(value=Constant(base_spring), marker=markers["BASE"][0])]


# Fix the basal plane in the longitudinal direction
# 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
def fix_basal_plane(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = DirichletBC(V.sub(0), Constant(0.0), geometry.ffun, markers["BASE"][0])
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
problem = pulse.MechanicsProblem(geometry, material, bcs)

# Solve the problem
problem.solve()

# Get the solution
u, p = problem.state.split(deepcopy=True)

volume = geometry.cavity_volume(u=u)
print(f"Cavity volume = {volume}")

# Move mesh accoring to displacement
u_int = interpolate(u, dolfin.VectorFunctionSpace(geometry.mesh, "CG", 1))
mesh = Mesh(geometry.mesh)
dolfin.ALE.move(mesh, u_int)

# Plot the result on to of the original
# dolfin.plot(geometry.mesh, alpha=0.1, edgecolor="k", color="w")
# dolfin.plot(mesh, color="r")

# ax = plt.gca()
# ax.view_init(elev=-67, azim=-179)
# ax.set_axis_off()
# # plt.show()
# plt.savefig("from_xml.png")
