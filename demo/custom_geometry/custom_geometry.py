"""
Install gmsh, meshio and ldrb

python -m pip install gmsh meshio ldrb

These examples are based on the snippets from
https://bitbucket.org/peppu/fenicshotools/src/master/demos/generate_from_geo.py

"""
import math
import sys
from pathlib import Path

import dolfin
import gmsh
import ldrb
import meshio

import pulse


def create_mesh(mesh, cell_type, prune_z=True):
    # From http://jsdokken.com/converted_files/tutorial_pygmsh.html
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]},
    )
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh


def read_meshfunction(fname, obj):
    with dolfin.XDMFFile(Path(fname).as_posix()) as f:
        f.read(obj, "name_to_read")


def gmsh2dolfin(msh_file):

    msh = meshio.gmsh.read(msh_file)

    vertex_mesh = create_mesh(msh, "vertex")
    line_mesh = create_mesh(msh, "line")
    triangle_mesh = create_mesh(msh, "triangle")
    tetra_mesh = create_mesh(msh, "tetra")

    vertex_mesh_name = Path("vertex_mesh.xdmf")
    meshio.write(vertex_mesh_name, vertex_mesh)

    line_mesh_name = Path("line_mesh.xdmf")
    meshio.write(line_mesh_name, line_mesh)

    triangle_mesh_name = Path("triangle_mesh.xdmf")
    meshio.write(triangle_mesh_name, triangle_mesh)

    tetra_mesh_name = Path("mesh.xdmf")
    meshio.write(
        tetra_mesh_name,
        tetra_mesh,
    )

    mesh = dolfin.Mesh()

    with dolfin.XDMFFile(tetra_mesh_name.as_posix()) as infile:
        infile.read(mesh)

    cfun = dolfin.MeshFunction("size_t", mesh, 3)
    read_meshfunction(tetra_mesh_name, cfun)
    tetra_mesh_name.unlink()

    ffun_val = dolfin.MeshValueCollection("size_t", mesh, 2)
    read_meshfunction(triangle_mesh_name, ffun_val)
    ffun = dolfin.MeshFunction("size_t", mesh, ffun_val)
    ffun.array()[ffun.array() == max(ffun.array())] = 0
    triangle_mesh_name.unlink()

    efun_val = dolfin.MeshValueCollection("size_t", mesh, 1)
    read_meshfunction(line_mesh_name, efun_val)
    efun = dolfin.MeshFunction("size_t", mesh, efun_val)
    efun.array()[efun.array() == max(efun.array())] = 0
    line_mesh_name.unlink()

    vfun_val = dolfin.MeshValueCollection("size_t", mesh, 0)
    read_meshfunction(vertex_mesh_name, vfun_val)
    vfun = dolfin.MeshFunction("size_t", mesh, vfun_val)
    vfun.array()[vfun.array() == max(vfun.array())] = 0
    vertex_mesh_name.unlink()

    markers = msh.field_data

    ldrb_markers = {
        "base": markers["BASE"][0],
        "epi": markers["EPI"][0],
        "lv": markers["ENDO"][0],
    }

    fiber_sheet_system = ldrb.dolfin_ldrb(mesh, "CG_1", ffun, ldrb_markers)

    marker_functions = pulse.MarkerFunctions(vfun=vfun, efun=efun, ffun=ffun, cfun=cfun)
    microstructure = pulse.Microstructure(
        f0=fiber_sheet_system.fiber,
        s0=fiber_sheet_system.sheet,
        n0=fiber_sheet_system.sheet_normal,
    )
    geo = pulse.HeartGeometry(
        mesh=mesh,
        markers=markers,
        marker_functions=marker_functions,
        microstructure=microstructure,
    )
    return geo


def create_disk(mesh_name, mesh_size_factor=1.0):

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

    geo = gmsh2dolfin(msh_name)
    dolfin.File("ffun.pvd") << geo.ffun


if __name__ == "__main__":
    main()
