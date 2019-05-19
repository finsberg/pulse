"""
This demo demonstrates how to unload a geometry using
the backward displament method which is an iterative
fixed point method.
"""
import matplotlib.pyplot as plt
import dolfin
import pulse


geometry = pulse.HeartGeometry.from_file(pulse.mesh_paths['simple_ellipsoid'])

material = pulse.HolzapfelOgden()
# material = pulse.Guccione()

# Parameter for the cardiac boundary conditions
bcs_parameters = pulse.MechanicsProblem.default_bcs_parameters()
bcs_parameters['base_spring'] = 1.0
bcs_parameters['base_bc'] = 'fix_x'

# Create the problem
# When performing unloading you should pass the
# bcs parameters instead of the bcs directly
problem = pulse.MechanicsProblem(geometry, material,
                                 bcs_parameters=bcs_parameters)

# Suppose geometry is loaded with a pressure of 1 kPa
# and create the unloader
unloader = pulse.FixedPointUnloader(problem=problem,
                                    pressure=1.0)

# Unload the geometry
unloader.unload()

# Get the unloaded geometry
unloaded_geometry = unloader.unloaded_geometry

plt.figure()
dolfin.plot(geometry.mesh, alpha=0.1, edgecolor='k', color='w')
dolfin.plot(unloaded_geometry.mesh)
# plt.show()
plt.savefig('unloaded.png')
