Problem 3
=========

This code implements problem 3 in [Land2015]_.


Code
----

.. code:: python
	  
    import matplotlib.pyplot as plt
    import dolfin
    import pulse


    geometry = pulse.HeartGeometry.from_file(pulse.mesh_paths['benchmark'])

    # Create the material
    material_parameters = pulse.Guccione.default_parameters()
    material_parameters['CC'] = 2.0
    material_parameters['bf'] = 8.0
    material_parameters['bfs'] = 4.0
    material_parameters['bt'] = 2.0

    activation = dolfin.Constant(0.0)
    material = pulse.Guccione(params=material_parameters,
			      active_model='active_stress',
			      activation=activation)


    # Define Dirichlet boundary. Fix the base_spring
    def dirichlet_bc(W):
	V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
	return dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)),
				  geometry.ffun, geometry.markers['BASE'][0])


    # Traction at the bottom of the beam
    lvp = dolfin.Constant(0.0)
    neumann_bc = pulse.NeumannBC(traction=lvp,
				 marker=geometry.markers['ENDO'][0])

    # Collect Boundary Conditions
    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,),
				   neumann=(neumann_bc,))

    # Create problem
    problem = pulse.MechanicsProblem(geometry, material, bcs)

    # Solve problem
    pulse.iterate.iterate(problem, lvp, 15.0)
    pulse.iterate.iterate(problem, activation, 60.0)

    # Get displacement and hydrostatic pressure
    u, p = problem.state.split(deepcopy=True)


    endo_apex_marker = geometry.markers['ENDOPT'][0]
    endo_apex_idx = geometry.vfun.array().tolist().index(endo_apex_marker)
    endo_apex = geometry.mesh.coordinates()[endo_apex_idx, :]
    endo_apex_pos = endo_apex + u(endo_apex)

    print(('\n\nGet longitudinal position of endocardial apex: {:.4f} mm'
	   '').format(endo_apex_pos[0]))


    epi_apex_marker = geometry.markers['EPIPT'][0]
    epi_apex_idx = geometry.vfun.array().tolist().index(epi_apex_marker)
    epi_apex = geometry.mesh.coordinates()[epi_apex_idx, :]
    epi_apex_pos = epi_apex + u(epi_apex)

    print(('\n\nGet longitudinal position of epicardial apex: {:.4f} mm'
	   '').format(epi_apex_pos[0]))

    V = dolfin.VectorFunctionSpace(geometry.mesh, "CG", 1)
    u_int = dolfin.interpolate(u, V)
    mesh = dolfin.Mesh(geometry.mesh)
    dolfin.ALE.move(mesh, u_int)

    plt.figure()
    dolfin.plot(geometry.mesh, alpha=0.5, color='w',
		edgecolor='b', title='Original geometry')
    ax1 = plt.gca()

    plt.figure()
    dolfin.plot(mesh, color='r', edgecolor='k', alpha=0.7,
		title='Inflating ellipsoid')
    ax2 = plt.gca()

    for ax in (ax1, ax2):
	ax.view_init(elev=-83, azim=-179)
    plt.show()


Output
------

.. code:: shell

    Get longitudinal position of endocardial apex: 11.8550 mm
    Get longitudinal position of epicardial apex: 15.4904 mm

Plot
----

.. image:: problem3.png
