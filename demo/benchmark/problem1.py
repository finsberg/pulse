# # Cardiac mechanics Benchmark - Problem 1
# This code implements problem 1 in the Cardiac Mechanic Benchmark paper
#
# > Land S, Gurev V, Arens S, Augustin CM, Baron L, Blake R, Bradley C, Castro S,
# Crozier A, Favino M, Fastl TE. Verification of cardiac mechanics software:
# benchmark problems and solutions for testing active and passive material
# behaviour. Proc. R. Soc. A. 2015 Dec 8;471(2184):20150641.
#
# +
import dolfin
import numpy as np

try:
    from dolfin_adjoint import (
        BoxMesh,
        Constant,
        DirichletBC,
        Expression,
        Mesh,
        interpolate,
    )
except ImportError:
    from dolfin import BoxMesh, Expression, DirichletBC, Constant, interpolate, Mesh

import pulse
from fenics_plotly import plot


# Create the Beam geometry

# Length
L = 10
# Width
W = 1

# Create mesh
mesh = BoxMesh(dolfin.Point(0, 0, 0), dolfin.Point(L, W, W), 30, 3, 3)

# Mark boundary subdomians
left = dolfin.CompiledSubDomain("near(x[0], side) && on_boundary", side=0)
bottom = dolfin.CompiledSubDomain("near(x[2], side) && on_boundary", side=0)

boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)

left_marker = 1
left.mark(boundary_markers, 1)
bottom_marker = 2
bottom.mark(boundary_markers, 2)

marker_functions = pulse.MarkerFunctions(ffun=boundary_markers)

# Create mictrotructure
f0 = Expression(("1.0", "0.0", "0.0"), degree=1, cell=mesh.ufl_cell())
s0 = Expression(("0.0", "1.0", "0.0"), degree=1, cell=mesh.ufl_cell())
n0 = Expression(("0.0", "0.0", "1.0"), degree=1, cell=mesh.ufl_cell())

# Collect the mictrotructure
microstructure = pulse.Microstructure(f0=f0, s0=s0, n0=n0)

# Create the geometry
geometry = pulse.Geometry(
    mesh=mesh,
    marker_functions=marker_functions,
    microstructure=microstructure,
)

# Create the material
material_parameters = pulse.Guccione.default_parameters()
material_parameters["CC"] = 2.0
material_parameters["bf"] = 8.0
material_parameters["bfs"] = 4.0
material_parameters["bt"] = 2.0

material = pulse.Guccione(parameters=material_parameters)


# Define Dirichlet boundary. Fix at the left boundary
def dirichlet_bc(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    return DirichletBC(V, Constant((0.0, 0.0, 0.0)), left)


# Traction at the bottom of the beam
p_bottom = Constant(0.004)
neumann_bc = pulse.NeumannBC(traction=p_bottom, marker=bottom_marker)

# Collect Boundary Conditions
bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann_bc,))

# Create problem
problem = pulse.MechanicsProblem(geometry, material, bcs)

# Solve problem
problem.solve()

# Get displacement and hydrostatic pressure
u, p = problem.state.split(deepcopy=True)

point = np.array([10.0, 0.5, 1.0])
disp = np.zeros(3)
u.eval(disp, point)

print(
    ("Get z-position of point ({}): {:.4f} mm" "").format(
        ", ".join(["{:.1f}".format(p) for p in point]),
        point[2] + disp[2],
    ),
)

V = dolfin.VectorFunctionSpace(geometry.mesh, "CG", 1)
u_int = interpolate(u, V)
mesh = Mesh(geometry.mesh)
dolfin.ALE.move(mesh, u_int)

fig = plot(geometry.mesh, show=False)
fig.add_plot(plot(mesh, color="red", show=False))
fig.show()
