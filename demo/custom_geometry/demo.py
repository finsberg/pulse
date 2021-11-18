"""
Install gmsh and meshio

python -m pip install gmsh meshio
"""
import math
import subprocess
import sys
from pathlib import Path

import dolfin
import gmsh

import pulse


# https://bitbucket.org/peppu/fenicshotools/src/master/demos/generate_from_geo.py


def convert_gmsh_to_dolfin(msh_file):

    # Based on https://github.com/MiroK/tieler/blob/master/tiles/msh_convert.py

    mesh_path = Path(msh_file)
    mesh_name = mesh_path.stem
    xml_file = mesh_path.with_suffix(".xml")

    # Get the xml mesh
    subprocess.call(["dolfin-convert", msh_file, xml_file], shell=True)

    assert xml_file.is_file()

    mesh = dolfin.Mesh(xml_file.as_posix())
    cfun = dolfin.MeshFunction(
        "size_t",
        mesh,
        mesh_path.with_name(f"{mesh_name}_physical_region.xml").as_posix(),
    )
    ffun = dolfin.MeshFunction(
        "size_t",
        mesh,
        mesh_path.with_name(f"{mesh_name}_facet_region.xml").as_posix(),
    )

    ffun_markers = {
        "surf_{i}": (value, ffun.dim()) for i, value in enumerate(set(ffun.array()))
    }
    cfun_markers = {
        "vol_{i}": (value, cfun.dim()) for i, value in enumerate(set(cfun.array()))
    }

    return pulse.Geometry(
        mesh=mesh,
        marker_functions=pulse.MarkerFunctions(ffun=ffun, cfun=cfun),
        markers={**ffun_markers, **cfun_markers},
    )


def create_disk(mesh_name, mesh_size_factor=3.0):

    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)

    Rin = 0.5
    Rout = 2.0
    psize = 1.5

    p1 = gmsh.model.geo.addPoint(Rin, 0.0, 0.0, psize)
    p2 = gmsh.model.geo.addPoint(Rout, 0.0, 0.0, psize)

    l1 = gmsh.model.geo.addLine(p1, p2)

    line_id = l1
    surf = []
    bc_epi = []
    bc_endo = []
    for _ in range(4):
        out = gmsh.model.geo.revolve(
            [(1, line_id)],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            math.pi / 2,
        )
        line_id = out[0][1]

        surf.append(out[1][1])
        bc_epi.append(out[2][1])
        bc_endo.append(out[3][1])

    phys_epi = gmsh.model.addPhysicalGroup(1, bc_epi)
    gmsh.model.setPhysicalName(1, phys_epi, "Epicardium")

    phys_endo = gmsh.model.addPhysicalGroup(1, bc_endo)
    gmsh.model.setPhysicalName(1, phys_endo, "Endocardium")

    phys_myo = gmsh.model.addPhysicalGroup(2, surf)
    gmsh.model.setPhysicalName(2, phys_myo, "Myocardium")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_name)

    gmsh.finalize()


def create_square(mesh_name, mesh_size_factor=1.0):

    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)

    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 0.1)
    p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, 0.1)
    p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, 0.1)
    p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, 0.1)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    ll = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([ll])

    phys_point = gmsh.model.addPhysicalGroup(0, [p1])
    gmsh.model.setPhysicalName(0, phys_point, "Origin")

    phys_line = gmsh.model.addPhysicalGroup(1, [l1])
    gmsh.model.setPhysicalName(1, phys_line, "Bottom")

    phys_surf = gmsh.model.addPhysicalGroup(2, [surf])
    gmsh.model.setPhysicalName(2, phys_surf, "Tissue")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_name)

    gmsh.finalize()


def create_prolate_mesh(mesh_name, mesh_size_factor=3.0):

    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)

    d_focal = 3.7
    l_epi = 0.7
    l_endo = 0.4
    mu_base = 120.0 / 180.0 * math.pi

    def ellipsoid_point(mu, theta, r1, r2):
        return gmsh.model.geo.addPoint(
            r1 * math.sin(mu) * math.cos(theta),
            r1 * math.sin(mu) * math.sin(theta),
            r2 * math.cos(mu),
            0.5,
        )

    center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)

    apex_endo = ellipsoid_point(
        mu=0.0,
        theta=0.0,
        r1=d_focal * math.sinh(l_endo),
        r2=d_focal * math.cosh(l_endo),
    )

    base_endo = ellipsoid_point(
        mu=mu_base,
        theta=0.0,
        r1=d_focal * math.sinh(l_endo),
        r2=d_focal * math.cosh(l_endo),
    )

    apex_epi = ellipsoid_point(
        mu=0.0,
        theta=0.0,
        r1=d_focal * math.sinh(l_epi),
        r2=d_focal * math.cosh(l_epi),
    )

    base_epi = ellipsoid_point(
        mu=math.acos(math.cosh(l_endo) / math.cosh(l_epi) * math.cos(mu_base)),
        theta=0.0,
        r1=d_focal * math.sinh(l_epi),
        r2=d_focal * math.cosh(l_epi),
    )

    apex = gmsh.model.geo.addLine(apex_endo, apex_epi)
    base = gmsh.model.geo.addLine(base_endo, base_epi)
    endo = gmsh.model.geo.add_ellipse_arc(apex_endo, center, apex_endo, base_endo)
    epi = gmsh.model.geo.add_ellipse_arc(apex_epi, center, apex_epi, base_epi)

    ll1 = gmsh.model.geo.addCurveLoop([apex, epi, -base, -endo], tag=1)

    s1 = gmsh.model.geo.addPlaneSurface([ll1], tag=2)

    sendoringlist = []
    sepiringlist = []
    sendolist = []
    sepilist = []
    sbaselist = []
    vlist = []

    out = [(2, s1)]
    for _ in range(4):
        out = gmsh.model.geo.revolve(
            [out[0]],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            math.pi / 2,
        )

        sendolist.append(out[4][1])
        sepilist.append(out[2][1])
        sbaselist.append(out[3][1])
        vlist.append(out[1][1])

        gmsh.model.geo.synchronize()
        bnd = gmsh.model.getBoundary([out[0]])

        sendoringlist.append(bnd[1][1])
        sepiringlist.append(bnd[3][1])

    phys_apex_endo = gmsh.model.addPhysicalGroup(0, [apex_endo])
    gmsh.model.setPhysicalName(0, phys_apex_endo, "APEX_ENDO")

    phys_apex_epi = gmsh.model.addPhysicalGroup(0, [apex_epi])
    gmsh.model.setPhysicalName(0, phys_apex_epi, "APEX_EPI")

    phys_epiring = gmsh.model.addPhysicalGroup(1, sepiringlist)
    gmsh.model.setPhysicalName(1, phys_epiring, "EPIRING")

    phys_endoring = gmsh.model.addPhysicalGroup(1, sendoringlist)
    gmsh.model.setPhysicalName(1, phys_endoring, "ENDORING")

    phys_base = gmsh.model.addPhysicalGroup(2, sbaselist)
    gmsh.model.setPhysicalName(2, phys_base, "BASE")

    phys_endo = gmsh.model.addPhysicalGroup(2, sendolist)
    gmsh.model.setPhysicalName(2, phys_endo, "ENDO")

    phys_epi = gmsh.model.addPhysicalGroup(2, sepilist)
    gmsh.model.setPhysicalName(2, phys_epi, "EPI")

    phys_myo = gmsh.model.addPhysicalGroup(3, vlist)
    gmsh.model.setPhysicalName(3, phys_myo, "MYOCARDIUM")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_name)

    gmsh.finalize()


def main():

    msh_name = "test.msh"

    create_prolate_mesh(msh_name)
    # create_square(msh_name)
    # create_disk(msh_name)

    convert_gmsh_to_dolfin(msh_name)


if __name__ == "__main__":
    main()
