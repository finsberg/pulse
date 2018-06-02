import pytest
import dolfin
import numpy as np

from pulse.geometry import (Geometry, Marker, CRLBasis,
                            Microstructure, MarkerFunctions)
from pulse.parameters import setup_general_parameters
from pulse.dolfin_utils import QuadratureSpace


setup_general_parameters()


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


def strain_markers_3d(mesh, nregions):
    strain_markers = dolfin.MeshFunction("size_t", mesh, 3)
    strain_markers.set_all(0)
    xs = np.linspace(0, 1, nregions+1)

    region = 0
    for it_x in range(nregions):
        for it_y in range(nregions):
            for it_z in range(nregions):

                region += 1
                domain_str = ""

                domain_str += "x[0] >= {}".format(xs[it_x])
                domain_str += " && x[1] >= {}".format(xs[it_y])
                domain_str += " && x[2] >= {}".format(xs[it_z])
                domain_str += " && x[0] <= {}".format(xs[it_x+1])
                domain_str += " && x[1] <= {}".format(xs[it_y+1])
                domain_str += " && x[2] <= {}".format(xs[it_z+1])

                len_sub = dolfin.CompiledSubDomain(domain_str)
                len_sub.mark(strain_markers, region)
    return strain_markers


@pytest.fixture
def unitcube_geometry():
    N = 2
    mesh = dolfin.UnitCubeMesh(N, N, N)

    V_f = QuadratureSpace(mesh, 4)

    l0 = dolfin.interpolate(
        dolfin.Expression(("1.0", "0.0", "0.0"), degree=1), V_f)
    r0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
    c0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "0.0", "1.0"), degree=1), V_f)

    crl_basis = CRLBasis(l0=l0, r0=r0, c0=c0)

    cfun = strain_markers_3d(mesh, 2)

    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)
    fixed.mark(ffun, fixed_marker)
    free.mark(ffun, free_marker)

    marker_functions = MarkerFunctions(ffun=ffun, cfun=cfun)

    fixed_marker_ = Marker(name='fixed', value=fixed_marker, dimension=2)
    free_marker_ = Marker(name='free', value=free_marker, dimension=2)

    markers = (fixed_marker_, free_marker_)

    # Fibers
    f0 = dolfin.interpolate(
        dolfin.Expression(("1.0", "0.0", "0.0"), degree=1),  V_f)
    s0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
    n0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "0.0", "1.0"), degree=1), V_f)

    microstructure = Microstructure(f0=f0, s0=s0, n0=n0)

    geometry = Geometry(mesh=mesh, markers=markers,
                        marker_functions=marker_functions,
                        microstructure=microstructure,
                        crl_basis=crl_basis)

    return geometry


def test_create_geometry(unitcube_geometry):

    # Check that attributes are set correctly
    assert unitcube_geometry.f0 is not None
    assert unitcube_geometry.s0 is not None
    assert unitcube_geometry.n0 is not None
    assert unitcube_geometry.microstructure is not None

    assert unitcube_geometry.l0 is not None
    assert unitcube_geometry.c0 is not None
    assert unitcube_geometry.r0 is not None
    assert unitcube_geometry.crl_basis is not None

    assert unitcube_geometry.ffun is not None
    assert unitcube_geometry.cfun is not None

    assert unitcube_geometry.mesh is not None
    assert unitcube_geometry.markers is not None


def test_mesvolumes(unitcube_geometry):

    assert (float(unitcube_geometry.meshvol) - 1.0) < dolfin.DOLFIN_EPS_LARGE
    assert all([(float(f) - 0.125) < dolfin.DOLFIN_EPS_LARGE
                for f in unitcube_geometry.meshvols])
    assert unitcube_geometry.nregions == 8


def test_crl_basis(unitcube_geometry):

    assert unitcube_geometry.crl_basis_list
    assert unitcube_geometry.nbasis == 3

    
if __name__ == "__main__":
    geo = unitcube_geometry()
    # test_create_geometry(geo)
    test_mesvolumes(geo)
