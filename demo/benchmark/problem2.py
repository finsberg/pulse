# # Cardiac mechanics Benchmark - Problem 2
# This code implements problem 2 in the Cardiac Mechanic Benchmark paper
#
# > Land S, Gurev V, Arens S, Augustin CM, Baron L, Blake R, Bradley C, Castro S,
# Crozier A, Favino M, Fastl TE. Verification of cardiac mechanics software:
# benchmark problems and solutions for testing active and passive material
# behaviour. Proc. R. Soc. A. 2015 Dec 8;471(2184):20150641.
#

from pathlib import Path

import dolfin

try:
    from dolfin_adjoint import Constant, DirichletBC, Mesh, interpolate
except ImportError:
    from dolfin import DirichletBC, Constant, interpolate, Mesh

import pulse
import cardiac_geometries as cg
from fenics_plotly import plot


geo_path = Path("geometry")
if not geo_path.is_dir():
    cg.create_benchmark_geometry_land15(outdir=geo_path)
geo = cg.geometry.Geometry.from_folder(geo_path)
geometry = pulse.HeartGeometry(
    mesh=geo.mesh,
    markers=geo.markers,
    marker_functions=pulse.MarkerFunctions(vfun=geo.vfun, ffun=geo.ffun),
    microstructure=pulse.Microstructure(f0=geo.f0, s0=geo.s0, n0=geo.n0),
)

# Create the material
material_parameters = pulse.Guccione.default_parameters()
material_parameters["C"] = 10.0
material_parameters["bf"] = 1.0
material_parameters["bfs"] = 1.0
material_parameters["bt"] = 1.0

material = pulse.Guccione(parameters=material_parameters)


# Define Dirichlet boundary. Fix the base_spring
def dirichlet_bc(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    return DirichletBC(
        V,
        Constant((0.0, 0.0, 0.0)),
        geometry.ffun,
        geometry.markers["BASE"][0],
    )


# Traction at the bottom of the beam
lvp = Constant(0.0)
neumann_bc = pulse.NeumannBC(traction=lvp, marker=geometry.markers["ENDO"][0])

# Collect Boundary Conditions
bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann_bc,))

# Create problem
problem = pulse.MechanicsProblem(geometry, material, bcs)

# Solve problem
pulse.iterate.iterate(problem, lvp, 10.0, initial_number_of_steps=200)

# Get displacement and hydrostatic pressure
u, p = problem.state.split(deepcopy=True)

endo_apex_marker = geometry.markers["ENDOPT"][0]
endo_apex_idx = geometry.vfun.array().tolist().index(endo_apex_marker)
endo_apex = geometry.mesh.coordinates()[endo_apex_idx, :]
endo_apex_pos = endo_apex + u(endo_apex)

print(
    ("\n\nGet longitudinal position of endocardial apex: {:.4f} mm" "").format(
        endo_apex_pos[0],
    ),
)


epi_apex_marker = geometry.markers["EPIPT"][0]
epi_apex_idx = geometry.vfun.array().tolist().index(epi_apex_marker)
epi_apex = geometry.mesh.coordinates()[epi_apex_idx, :]
epi_apex_pos = epi_apex + u(epi_apex)

print(
    ("\n\nGet longitudinal position of epicardial apex: {:.4f} mm" "").format(
        epi_apex_pos[0],
    ),
)

V = dolfin.VectorFunctionSpace(geometry.mesh, "CG", 1)
u_int = interpolate(u, V)
mesh = Mesh(geometry.mesh)
dolfin.ALE.move(mesh, u_int)


fig = plot(geometry.mesh, color="red", show=False)
fig.add_plot(plot(mesh, opacity=0.3, show=False))
fig.show()
