import dolfin
import numpy as np
import pytest

from pulse.dolfin_utils import QuadratureSpace
from pulse.example_meshes import mesh_paths
from pulse.geometry import (
    CRLBasis,
    Geometry,
    HeartGeometry,
    MarkerFunctions,
    Microstructure,
)


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
    xs = np.linspace(0, 1, nregions + 1)

    region = 0
    for it_x in range(nregions):
        for it_y in range(nregions):
            for it_z in range(nregions):

                region += 1
                domain_str = ""

                domain_str += "x[0] >= {}".format(xs[it_x])
                domain_str += " && x[1] >= {}".format(xs[it_y])
                domain_str += " && x[2] >= {}".format(xs[it_z])
                domain_str += " && x[0] <= {}".format(xs[it_x + 1])
                domain_str += " && x[1] <= {}".format(xs[it_y + 1])
                domain_str += " && x[2] <= {}".format(xs[it_z + 1])

                len_sub = dolfin.CompiledSubDomain(domain_str)
                len_sub.mark(strain_markers, region)
    return strain_markers


@pytest.fixture
def unitcube_geometry():
    N = 2
    mesh = dolfin.UnitCubeMesh(N, N, N)

    V_f = QuadratureSpace(mesh, 4)

    l0 = dolfin.interpolate(dolfin.Expression(("1.0", "0.0", "0.0"), degree=1), V_f)
    r0 = dolfin.interpolate(dolfin.Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
    c0 = dolfin.interpolate(dolfin.Expression(("0.0", "0.0", "1.0"), degree=1), V_f)

    crl_basis = CRLBasis(l0=l0, r0=r0, c0=c0)

    cfun = strain_markers_3d(mesh, 2)

    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)
    fixed.mark(ffun, fixed_marker)
    free.mark(ffun, free_marker)

    marker_functions = MarkerFunctions(ffun=ffun, cfun=cfun)

    # Fibers
    f0 = dolfin.interpolate(dolfin.Expression(("1.0", "0.0", "0.0"), degree=1), V_f)
    s0 = dolfin.interpolate(dolfin.Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
    n0 = dolfin.interpolate(dolfin.Expression(("0.0", "0.0", "1.0"), degree=1), V_f)

    microstructure = Microstructure(f0=f0, s0=s0, n0=n0)

    geometry = Geometry(
        mesh=mesh,
        marker_functions=marker_functions,
        microstructure=microstructure,
        crl_basis=crl_basis,
    )

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
    assert all(
        [
            (float(f) - 0.125) < dolfin.DOLFIN_EPS_LARGE
            for f in unitcube_geometry.meshvols
        ]
    )
    assert unitcube_geometry.nregions == 8


def test_crl_basis(unitcube_geometry):

    assert unitcube_geometry.crl_basis_list
    assert unitcube_geometry.nbasis == 3


def test_simple_ellipsoid():
    geometry = HeartGeometry.from_file(mesh_paths["simple_ellipsoid"])

    assert abs(geometry.cavity_volume() - 2.511304019359619) < 1e-14
    assert geometry.is_biv is False


def test_biv_ellipsoid():
    geometry = HeartGeometry.from_file(mesh_paths["biv_ellipsoid"])

    assert abs(geometry.cavity_volume(chamber="lv") - 0.7422747965172004) < 1e-14
    assert abs(geometry.cavity_volume(chamber="rv") - 0.8171389194959642) < 1e-14
    assert geometry.is_biv is True


if __name__ == "__main__":
    # geo = unitcube_geometry()
    # test_create_geometry(geo)
    # test_mesvolumes(geo)
    test_simple_ellipsoid()
    test_biv_ellipsoid()
