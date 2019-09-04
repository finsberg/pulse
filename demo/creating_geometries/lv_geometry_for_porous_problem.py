import matplotlib.pyplot as plt
import dolfin as df
import mshr

from pulse.geometry_utils import generate_fibers
from pulse.geometry import Microstructure, Geometry, MarkerFunctions


geo_file = "lv_ellipsoid.h5"


base_x = 0.0

# LV
# The center of the LV ellipsoid
center = df.Point(0.0, 0.0, 0.0)
a_epi = 2.0
b_epi = 1.0
c_epi = 1.0

a_endo = 1.5
b_endo = 0.5
c_endo = 0.5


# Markers (first index is the marker, second is the topological dimension)
markers = dict(BASE=(10, 2),
               ENDO=(30, 2),
               EPI=(40, 2))


# Some refinement level
res = [30]


class Endo(df.SubDomain):
    def inside(self, x, on_boundary):
        return (x[0]-center.x())**2/a_endo**2 \
            + (x[1]-center.y())**2/b_endo**2 \
            + (x[2]-center.z())**2/c_endo**2 -1.1 < df.DOLFIN_EPS \
            and on_boundary

class Base(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] - base_x < df.DOLFIN_EPS and on_boundary

class Epi(df.SubDomain):
    def inside(self, x, on_boundary):
        return (x[0]-center.x())**2/a_epi**2 \
            + (x[1]-center.y())**2/b_epi**2 \
            + (x[2]-center.z())**2/c_epi**2 - 0.9 > df.DOLFIN_EPS \
            and on_boundary


# The plane cutting the base
diam = -10.0
box = mshr.Box(df.Point(base_x, 2, 2), df.Point(diam, diam, diam))
# Generate mesh


# LV epicardium
el_lv = mshr.Ellipsoid(center, a_epi, b_epi, c_epi)
# LV endocardium
el_lv_endo = mshr.Ellipsoid(center, a_endo, b_endo, c_endo)

# LV geometry (subtract the smallest ellipsoid)
lv = el_lv - el_lv_endo

# LV geometry
m = lv - box


for N in res:
    # Create mesh
    mesh = mshr.generate_mesh(m, N)

    # Create facet function
    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    endo = Endo()
    endo.mark(ffun, markers['ENDO'][0])
    base = Base()
    base.mark(ffun, markers['BASE'][0])
    epi = Epi()
    epi.mark(ffun, markers['EPI'][0])

    # Mark mesh
    for facet in df.facets(mesh):
        mesh.domains().set_marker((facet.index(), ffun[facet]), 2)

    marker_functions = MarkerFunctions(ffun=ffun)


    # Make fiber field
    fiber_params = df.Parameters("Fibers")
    fiber_params.add("fiber_space", "CG_1")
    # fiber_params.add("fiber_space", "Quadrature_4")
    fiber_params.add("include_sheets", False)
    fiber_params.add("fiber_angle_epi", -60)
    fiber_params.add("fiber_angle_endo", 60)

    try:
        fields = generate_fibers(mesh, fiber_params)
    except ImportError:
        fields = []
        fields_names = []
    else:
        fields_names = ['f0', 's0', 'n0']

    microstructure = Microstructure(**dict(zip(fields_names, fields)))

    geometry = Geometry(mesh, markers=markers,
                        marker_functions=marker_functions,
                        microstructure=microstructure)
    geometry.save('N{}/lv_geometry'.format(N))


    ffile = df.XDMFFile('N{}/fiber.xdmf'.format(N))
    ffile.write_checkpoint(fields[0], 'fiber')
    ffile.close()

    ffile = df.XDMFFile('N{}/mesh.xdmf'.format(N))
    ffile.write(mesh)
    ffile.close()

    ffile = df.XDMFFile('N{}/markers.xdmf'.format(N))
    ffile.write(ffun)
    ffile.close()


    df.plot(mesh)
    ax = plt.gca()
    ax.view_init(elev=-67, azim=-179)
    ax.set_axis_off()

    plt.savefig('N{}/lv_geometry.png'.format(N))
    plt.close()

    if fields:
        df.plot(fields[0])
        ax = plt.gca()
        ax.view_init(elev=-67, azim=-179)
        ax.set_axis_off()

        plt.savefig('N{}/lv_geometry_fiber.png'.format(N))
