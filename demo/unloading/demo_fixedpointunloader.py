# # Finding the unloaded geometry
#

from fenics_plotly import plot

import pulse


geometry = pulse.HeartGeometry.from_file(pulse.mesh_paths["simple_ellipsoid"])
# geometry = pulse.geometries.prolate_ellipsoid_geometry(mesh_size_factor=3.0)
material = pulse.NeoHookean()
# material = pulse.Guccione()

# Parameter for the cardiac boundary conditions
bcs_parameters = pulse.MechanicsProblem.default_bcs_parameters()
bcs_parameters["base_spring"] = 1.0
bcs_parameters["base_bc"] = "fix_x"

# Create the problem
problem = pulse.MechanicsProblem(geometry, material, bcs_parameters=bcs_parameters)

# Suppose geometry is loaded with a pressure of 1 kPa
# and create the unloader
unloader = pulse.FixedPointUnloader(problem=problem, pressure=3.0)

# Unload the geometry
unloader.unload()

# Get the unloaded geometry
unloaded_geometry = unloader.unloaded_geometry

fig = plot(geometry.mesh, opacity=0.0, show=False)
fig.add_plot(plot(unloaded_geometry.mesh, color="red", show=False))
fig.show()
