import itertools
import pytest
import dolfin

from pulse.geometry import (Geometry, Marker,
                            Microstructure, MarkerFunctions)
from pulse.dolfin_utils import QuadratureSpace
from pulse import kinematics
from pulse.utils import mpi_comm_world


from pulse.material import material_models

from pulse.mechanicsproblem import (MechanicsProblem,
                                    BoundaryConditions,
                                    NeumannBC)



class Free(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > (1.0 - dolfin.DOLFIN_EPS) and on_boundary


class Fixed(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < dolfin.DOLFIN_EPS and on_boundary


fixed = Fixed()
fixed_marker = 1

free = Free()
free_marker = 2


@pytest.fixture
def unitcube_geometry():

    N = 3
    mesh = dolfin.UnitCubeMesh(mpi_comm_world(), N, N, N)

    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    fixed.mark(ffun, fixed_marker)
    free.mark(ffun, free_marker)

    marker_functions = MarkerFunctions(ffun=ffun)
    fixed_marker_ = Marker(name='fixed', value=fixed_marker, dimension=2)
    free_marker_ = Marker(name='free', value=free_marker, dimension=2)
    markers = (fixed_marker_, free_marker_)

    # Fibers
    V_f = QuadratureSpace(mesh, 4)

    f0 = dolfin.interpolate(
        dolfin.Expression(("1.0", "0.0", "0.0"), degree=1),  V_f)
    s0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
    n0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "0.0", "1.0"), degree=1), V_f)

    microstructure = Microstructure(f0=f0, s0=s0, n0=n0)

    geometry = Geometry(mesh=mesh, markers=markers,
                        marker_functions=marker_functions,
                        microstructure=microstructure)

    return geometry


cases = itertools.product(material_models,
                          ('active_strain', 'active_stress'),
                          (True, False))


@pytest.mark.parametrize("Material, active_model, isochoric", cases)
def test_material(unitcube_geometry, Material, active_model, isochoric):

    compressible_model = "incompressible"

    if active_model == "active_stress":
        active_value = 20.0
        activation = dolfin.Constant(1.0)
        T_ref = active_value

        def dirichlet_bc(W):
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            return dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)),
                                      fixed)

    else:
        activation = dolfin.Constant(0.0)
        active_value = 0.0
        T_ref = 1.0

        def dirichlet_bc(W):
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            return dolfin.DirichletBC(V.sub(0),
                                      dolfin.Constant(0.0), fixed, 'pointwise')

    neumann_bc = NeumannBC(traction=dolfin.Constant(-active_value),
                           marker=free_marker)

    bcs = BoundaryConditions(dirichlet=(dirichlet_bc,),
                             neumann=(neumann_bc,))

    matparams = Material.default_parameters()

    material = Material(activation=activation,
                        parameters=matparams,
                        T_ref=T_ref, isochoric=isochoric,
                        compressible_model=compressible_model,
                        active_model=active_model)

    assert material.is_isochoric == isochoric

    problem = MechanicsProblem(unitcube_geometry, material, bcs)
    problem.solve()

    u, p = problem.state.split(deepcopy=True)

    print(material.name)

    if active_model == 'active_strain':

        tol = 1e-4

        if not isochoric:
            if material.name in ["guccione", "linear_elastic",
                                 "saint_venant_kirchhoff"]:
                assert all(abs(p.vector().get_local()) < tol)
            elif material.name == "holzapfel_ogden":

                assert all(abs(p.vector().get_local()
                               - material.parameters["a"]) < tol)
            elif material.name == "neo_hookean":
                assert all(abs(p.vector().get_local()
                               - material.parameters["mu"]) < tol)
            else:
                raise TypeError("Unkown material {}".format(material.name))

        else:
            assert all(abs(p.vector().get_local()) < tol)

    else:

        F = kinematics.DeformationGradient(u)
        T = material.CauchyStress(F, p)

        V_dg = dolfin.FunctionSpace(unitcube_geometry.mesh, "DG", 1)

        # Fiber on current geometry
        f = F * unitcube_geometry.f0

        # Fiber stress
        Tf = dolfin.inner(T * f / f**2, f)
        Tf_dg = dolfin.project(Tf, V_dg)

        tol = 1e-10

        assert all(abs(Tf_dg.vector().get_local() - active_value) < tol)
        assert all(abs(u.vector().get_local()) < tol)

        if not isochoric:
            if material.name in ["guccione", "linear_elastic",
                                 "saint_venant_kirchhoff"]:
                assert all(abs(p.vector().get_local()) < tol)
            elif material.name == "holzapfel_ogden":

                assert all(abs(p.vector().get_local()
                               - material.parameters["a"]) < tol)
            elif material.name == "neo_hookean":
                assert all(abs(p.vector().get_local()
                               - material.parameters["mu"]) < tol)
            else:
                raise TypeError("Unkown material {}".format(material.name))

        else:
            assert all(abs(p.vector().get_local()) < tol)


if __name__ == "__main__":

    active_model = "active_stress"
    dev_iso_split = True
    geo = unitcube_geometry()
    for m in material_models:
        test_material(geo, m,
                      active_model, dev_iso_split)
