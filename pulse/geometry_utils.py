import os
import numpy as np
import h5py
import dolfin

from . import numpy_mpi
from . import io_utils
# from . import config
from .utils import make_logger


logger = make_logger(__name__, 10)


def move(mesh, u, factor=1.0):
    """
    Move mesh according to some displacement times some factor
    """
    W = dolfin.VectorFunctionSpace(u.function_space().mesh(), "CG", 1)

    # Use interpolation for now. It is the only thing that makes sense
    u_int = dolfin.interpolate(u, W)

    u0 = dolfin.Function(W)
    arr = factor * numpy_mpi.gather_broadcast(u_int.vector().array())
    numpy_mpi.assign_to_vector(u0.vector(), arr)

    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    U = dolfin.Function(V)
    numpy_mpi.assign_to_vector(U.vector(), arr)

    dolfin.ALE.move(mesh, U)

    return u0


def get_fiber_markers(mesh_type="lv"):
    """
    Get the markers for the mesh.
    This is the default markers for fiberrules.

    :param str mesh_type: type of mesh, 'lv' or 'biv'
    :returns: The markers
    :rtype: dict

    """
    if mesh_type == "lv":

        return {'BASE': 10,
                'ENDO': 30,
                'EPI': 40,
                'WALL': 50,
                'ENDORING': 300,
                'EPIRING': 400,
                'WALL': 50}

    elif mesh_type == "biv":

        return {'BASE': 10,
                'ENDO_RV': 20,
                'ENDO_LV': 30,
                'EPI': 40,
                'ENDORING_RV': 200,
                'ENDORING_LV': 300,
                'EPIRING': 400,
                'WALL': 50}


def get_markers(mesh_type="lv"):

    assert mesh_type in ["lv", "biv"]

    fiber_markers = get_fiber_markers(mesh_type)

    markers = {}
    markers["NONE"] = (0, 3)

    markers["BASE"] = (fiber_markers["BASE"], 2)
    markers["EPI"] = (fiber_markers["EPI"], 2)
    markers["EPIRING"] = (fiber_markers["EPIRING"], 1)

    if mesh_type == "lv":

        markers["ENDO"] = (fiber_markers["ENDO"], 2)

        markers["ENDORING"] = (fiber_markers["ENDORING"], 1)

    else:

        markers["ENDO_RV"] = (fiber_markers["ENDO_RV"], 2)
        markers["ENDO_LV"] = (fiber_markers["ENDO_LV"], 2)

        markers["ENDORING_RV"] = (fiber_markers["ENDORING_RV"], 1)
        markers["ENDORING_LV"] = (fiber_markers["ENDORING_LV"], 1)

    return markers


def load_geometry_from_h5(h5name, h5group="",
                          fendo=None, fepi=None,
                          include_sheets=True,
                          comm=dolfin.mpi_comm_world()):
    """Load geometry and other mesh data from
    a h5file to an object.
    If the file contains muliple fiber fields
    you can spefify the angles, and if the file
    contais sheets and cross-sheets this can also
    be included

    :param str h5name: Name of the h5file
    :param str h5group: The group within the file
    :param int fendo: Helix fiber angle (endocardium) (if available)
    :param int fepi: Helix fiber angle (epicardium) (if available)
    :param bool include_sheets: Include sheets and cross-sheets
    :returns: An object with geometry data
    :rtype: object

    """

    logger.info("\nLoad mesh from h5")
    # Set default groups
    ggroup = '{}/geometry'.format(h5group)
    mgroup = '{}/mesh'.format(ggroup)
    lgroup = "{}/local basis functions".format(h5group)
    fgroup = "{}/microstructure/".format(h5group)

    if not os.path.isfile(h5name):
        raise IOError("File {} does not exist".format(h5name))

    # Check that the given file contains
    # the geometry in the given h5group
    if not io_utils.check_h5group(h5name, mgroup,
                                  delete=False, comm=comm):
        msg = ("Warning!\nGroup: '{}' does not exist in file:"
               "\n{}").format(mgroup, h5name)

        with h5py.File(h5name) as h:
            keys = h.keys()
        msg += "\nPossible values for the h5group are {}".format(keys)
        raise IOError(msg)

    # Create a dummy object for easy parsing
    class Geometry(object):
        pass
    geo = Geometry()

    with dolfin.HDF5File(comm, h5name, "r") as h5file:

        # Load mesh
        mesh = dolfin.Mesh(comm)
        # h5file.read(mesh, mgroup, False)
        h5file.read(mesh, mgroup, True)
        geo.mesh = mesh

        # Get mesh functions
        for dim, attr in zip(range(4), ["vfun", "efun", "ffun", "cfun"]):

            dgroup = '{}/mesh/meshfunction_{}'.format(ggroup, dim)
            mf = dolfin.MeshFunction("size_t", mesh, dim, mesh.domains())
            if h5file.has_dataset(dgroup):
                h5file.read(mf, dgroup)

            setattr(geo, attr, mf)

        load_local_basis(h5file, lgroup, mesh, geo)

        load_microstructure(h5file, fgroup, mesh, geo, include_sheets)

        # Load the boundary markers
        markers = load_markers(h5file, mesh, ggroup, dgroup)
        geo.markers = markers

        origmeshgroup = "{}/original_geometry".format(h5group)
        if h5file.has_dataset(origmeshgroup):
            original_mesh = dolfin.Mesh(comm)
            h5file.read(original_mesh, origmeshgroup, True)
            setattr(geo, "original_geometry", original_mesh)

    return geo


def load_markers(h5file, mesh, ggroup, dgroup):
    try: 
        markers = {}
        for dim in range(mesh.ufl_domain().topological_dimension()+1):
            for key_str in ["domain", "meshfunction"]:
                dgroup = '{}/mesh/{}_{}'.format(ggroup, key_str, dim)

                # If dataset is not present
                if not h5file.has_dataset(dgroup):
                    continue

                for aname in h5file.attributes(dgroup).str().strip()\
                                                            .split(' '):
                    if aname.startswith('marker_name'):

                        name = aname.rsplit('marker_name_')[-1]
                        marker = h5file.attributes(dgroup)['marker_name_{}'.format(name)]
                        markers[name] = (int(marker), dim)
        
    except Exception as ex:
        logger.info("Unable to load makers")
        logger.info(ex)
        markers = get_markers()

    return markers


def full_arctangent(x, y):
    t = np.arctan2(x, y)
    if t < 0:
        return t + 2*np.pi
    else:
        return t


def cartesian_to_prolate_ellipsoidal(x, y, z, a):

    b1 = np.sqrt((x + a)**2 + y**2 + z**2)
    b2 = np.sqrt((x - a)**2 + y**2 + z**2)

    sigma = 1/(2.0*a)*(b1 + b2)
    tau = 1/(2.0*a)*(b1 - b2)
    phi = full_arctangent(z, y)
    mu = np.arccosh(sigma)
    nu = np.arccos(tau)
    return mu, nu, phi


def prolate_ellipsoidal_to_cartesian(mu, nu, phi, a):
    x = a*np.cosh(mu)*np.cos(nu)
    y = a*np.sinh(mu)*np.sin(nu)*np.cos(phi)
    z = a*np.sinh(mu)*np.sin(nu)*np.sin(phi)
    return x, y, z


def fill_coordinates_ec(i, e_c_x, e_c_y, e_c_z, coord, foci):
    norm = dolfin.sqrt(coord[1]**2 + coord[2]**2)
    if not dolfin.near(norm, 0):
        e_c_y.vector()[i] = -coord[2]/norm
        e_c_z.vector()[i] = coord[1]/norm
    else:
        # We are at the apex where crl system doesn't make sense
        # So just pick something.
        e_c_y.vector()[i] = 1
        e_c_z.vector()[i] = 0

        
def fill_coordinates_el(i, e_c_x, e_c_y, e_c_z, coord, foci):

    norm = dolfin.sqrt(coord[1]**2 + coord[2]**2)
    if not dolfin.near(norm, 0):
        mu, nu, phi \
            = cartesian_to_prolate_ellipsoidal(*(coord.tolist() + [foci]))
        x, y, z = prolate_ellipsoidal_to_cartesian(mu, nu + 0.01, phi, foci)
        r = np.array([coord[0] - x,
                      coord[1] - y,
                      coord[2] - z])
        e_r = r/np.linalg.norm(r)
        e_c_x.vector()[i] = e_r[0]
        e_c_y.vector()[i] = e_r[1]
        e_c_z.vector()[i] = e_r[2]
    else:
        e_c_y.vector()[i] = 0
        e_c_z.vector()[i] = 1


def calc_cross_products(e1, e2, VV):
    e_crossed = dolfin.Function(VV)

    e1_arr = e1.vector().array().reshape((-1, 3))
    e2_arr = e2.vector().array().reshape((-1, 3))

    crosses = []
    for c1, c2 in zip(e1_arr, e2_arr):
        crosses.extend(np.cross(c1, c2.tolist()))

    e_crossed.vector()[:] = np.array(crosses)[:]
    return e_crossed


def check_norms(e):

    e_arr = e.vector().array().reshape((-1, 3))
    for e_i in e_arr:
        assert(dolfin.near(np.linalg.norm(e_i), 1.0))


def make_unit_vector(V, VV, dofs_x, fill_coordinates, foc=None):
    e_c_x = dolfin.Function(V)
    e_c_y = dolfin.Function(V)
    e_c_z = dolfin.Function(V)

    for i, coord in enumerate(dofs_x):
        fill_coordinates(i, e_c_x, e_c_y, e_c_z, coord, foc)

    e = dolfin.Function(VV)

    fa = [dolfin.FunctionAssigner(VV.sub(i), V) for i in range(3)]
    for i, e_c_comp in enumerate([e_c_x, e_c_y, e_c_z]):
        fa[i].assign(e.split()[i], e_c_comp)
    return e


def make_LV_crl_basis(mesh, foc):
    """
    Makes the crl  basis for the LV mesh (prolate ellipsoidal)
    with prespecified focal length.
    """

    VV = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    V = dolfin.FunctionSpace(mesh, "CG", 1)

    if dolfin.DOLFIN_VERSION_MAJOR > 1.6:
        dofs_x = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
    else:
        dm = V.dofmap()
        dofs_x = dm.tabulate_all_coordinates(mesh).reshape((-1, mesh.geometry().dim()))

    e_c = make_unit_vector(V, VV, dofs_x, fill_coordinates_ec)
    e_l = make_unit_vector(V, VV, dofs_x, fill_coordinates_el, foc)
    e_r = calc_cross_products(e_c, e_l, VV)

    e_c.rename("c0", "local_basis_function")
    e_r.rename("r0", "local_basis_function")
    e_l.rename("l0", "local_basis_function")

    return e_c, e_r, e_l


def make_BiV_crl_basis(mesh):

    c0 = get_circ_field(mesh)
    l0 = get_long_field(mesh, "biv")
    r0 = calc_cross_products(c0, l0, c0.function_space())

    return c0, r0, l0


def generate_local_basis_functions(mesh, focal_point, mesh_type="lv"):

    assert mesh_type in ["lv", "biv"], "Excepted mesh_type to be 'lv' or 'biv'"

    if mesh_type == "biv":

        try:
            c0, r0, l0 = make_BiV_crl_basis(mesh)

        except ImportError:
            pass
        else:
            return c0, r0, l0

    # Make basis functions
    c0, r0, l0 = make_crl_basis(mesh, focal_point)

    return c0, r0, l0


def fibers(mesh, fiber_endo = 60, fiber_epi=-60):
    
    fiber_params = Parameters("Fibers")
    fiber_params.add("fiber_space", "Quadrature_4")
    fiber_params.add("include_sheets", False)

    fiber_params.add("fiber_angle_epi", fiber_epi)
    fiber_params.add("fiber_angle_endo", fiber_endo)
    fiber_params.add("sheet_angle_epi", 0)
    fiber_params.add("sheet_angle_endo", 0)

    f0 = generate_fibers(mesh, fiber_params)[0]
    return f0

def get_circ_field(mesh):

    fiber_params = Parameters("Fibers")
    fiber_params.add("fiber_space", "Quadrature_4")
    # fiber_params.add("fiber_space", "Lagrange_1")
    fiber_params.add("include_sheets", False)

    # Parameter set from Bayer et al.
    fiber_params.add("fiber_angle_epi", 0)
    fiber_params.add("fiber_angle_endo", 0)
    fiber_params.add("sheet_angle_epi", 0)
    fiber_params.add("sheet_angle_endo", 0)

    f0 = generate_fibers(mesh, fiber_params)[0]
    f0.rename("circumferential", "local_basis_function")
   
    return f0

def get_long_field(mesh, mesh_type="biv"):
    fiber_params = Parameters("Fibers")
    fiber_params.add("fiber_space", "Quadrature_4")
    # fiber_params.add("fiber_space", "Lagrange_1")
    fiber_params.add("include_sheets", False)

    # Parameter set from Bayer et al.
    fiber_params.add("fiber_angle_epi", -90)
    fiber_params.add("fiber_angle_endo", -90)
    fiber_params.add("sheet_angle_epi", 0)
    fiber_params.add("sheet_angle_endo", 0)

    if mesh_type == "biv":
        # We need to set the markers for then LV and RV facets 
        ffun = dolfin.MeshFunction("size_t", mesh, 2, mesh.domains())

        markers = get_fiber_markers("biv")
        # Mark the mesh with same markers on the LV and RV before
        # running the LDRB algorithm
        markers["ENDO_RV"] = markers["ENDO_LV"]
        for facet in doflin.facets(mesh):
            if ffun[facet] != 0:
                mesh.domains().set_marker((facet.index(),
                                           markers[ffun[facet]]), 2)


    f0 = generate_fibers(mesh, fiber_params)[0]
    f0.rename("longitudinal", "local_basis_function")

    if mesh_type == "biv":
        # Put the correct markers
        markers = get_fiber_markers("biv")
        for facet in doflin.facets(mesh):
            if ffun[facet] != 0:
                mesh.domains().set_marker((facet.index(),
                                           markers[ffun[facet]]), 2)
    return f0


def generate_fibers(mesh, fiber_params):

    try:
        from fiberrules import dolfin_fiberrules
    except ImportError as ex:
        logger.error("Unable to generate fibers without fiberrules package")
        logger.error("This package is protected by copyright")
        raise ex

    logger.info("\nGENERATING FIBERS ...")

    fiber_space_name = fiber_params["fiber_space"]

    assert len(fiber_space_name.split("_")) == 2, \
        "expected fiber_space_name in 'FamilyName_Degree' format"

    family, degree = fiber_space_name.split("_")

    if dolfin.DOLFIN_VERSION_MAJOR > 1.6:
        el = dolfin.FiniteElement(family=family,
                                  cell=mesh.ufl_cell(),
                                  degree=int(degree),
                                  quad_scheme="default")
        fiber_space = dolfin.FunctionSpace(mesh, el)
    else:
        fiber_space = dolfin.FunctionSpace(mesh, family, int(degree))

    if family == "Quadrature":
        dolfin.parameters["form_compiler"]["quadrature_degree"] = int(degree)
        if dolfin.DOLFIN_VERSION_MAJOR > 2016:
            dolfin.parameters["form_compiler"]["representation"] = "quadrature"

    # There are some strange shifting in the fiberrults angles
    fiber_angle_epi = 90 - (-fiber_params["fiber_angle_epi"])
    fiber_angle_endo = 90 - (fiber_params["fiber_angle_endo"])
    sheet_angle_endo = fiber_params["sheet_angle_endo"]
    sheet_angle_epi = fiber_params["sheet_angle_epi"]

    microstructures = dolfin_fiberrules(mesh,
                                        fiber_space,
                                        fiber_angle_epi,
                                        fiber_angle_endo,
                                        sheet_angle_epi,
                                        sheet_angle_endo)

    microstructures[0].rename("fiber",
                              "epi{}_endo{}".format(fiber_params["fiber_angle_epi"], 
                                                    fiber_params["fiber_angle_endo"]))
    fields = [microstructures[0]]


    if fiber_params["include_sheets"]:
        microstructures[1].rename("sheet",
                                  "epi{}_endo{}".format(sheet_angle_epi, 
                                                                sheet_angle_endo))
        microstructures[2].rename("cross_sheet",
                                  "fepi{}_fendo{}_sepi{}_sendo{}".format(fiber_params["fiber_angle_epi"], 
                                                                         fiber_params["fiber_angle_endo"],
                                                                         sheet_angle_epi, 
                                                                         sheet_angle_endo))
        fields.append(microstructures[1])
        fields.append(microstructures[2])

    return fields

def load_local_basis(h5file, lgroup, mesh, geo):

    if h5file.has_dataset(lgroup):
        # Get local bais functions
        local_basis_attrs = h5file.attributes(lgroup)
        lspace = local_basis_attrs["space"]
        family, order = lspace.split('_')

        namesstr = local_basis_attrs["names"]
        names = namesstr.split(":")

        if dolfin.DOLFIN_VERSION_MAJOR > 1.6:
            elm = dolfin.VectorElement(family=family,
                                       cell=mesh.ufl_cell(),
                                       degree=int(order),
                                       quad_scheme="default")
            V = dolfin.FunctionSpace(mesh, elm)
        else:
            V = dolfin.VectorFunctionSpace(mesh, family, int(order))

        for name in names:
            lb = dolfin.Function(V, name=name)
            h5file.read(lb, lgroup+"/{}".format(name))
            setattr(geo, name, lb)


def load_microstructure(h5file, fgroup, mesh, geo, include_sheets=True):

    if h5file.has_dataset(fgroup):
        # Get fibers
        fiber_attrs = h5file.attributes(fgroup)
        fspace = fiber_attrs["space"]
        if fspace is None:
            # Assume quadrature 4
            family = "Quadrature"
            order = 4
        else:
            family, order = fspace.split('_')

        namesstr = fiber_attrs["names"]
        if namesstr is None:
            names = ["fiber"]
        else:
            names = namesstr.split(":")

        # Check that these fibers exists
        for name in names:
            fsubgroup = fgroup+"/{}".format(name)
            if not h5file.has_dataset(fsubgroup):
                msg = ("H5File does not have dataset {}").format(fsubgroup)
                logger.warning(msg)

        if dolfin.DOLFIN_VERSION_MAJOR > 1.6:
            elm = dolfin.VectorElement(family=family,
                                       cell=mesh.ufl_cell(),
                                       degree=int(order),
                                       quad_scheme="default")
            V = dolfin.FunctionSpace(mesh, elm)
        else:
            V = dolfin.VectorFunctionSpace(mesh, family, int(order))

        attrs = ["f0", "s0", "n0"]
        for i, name in enumerate(names):
            func = dolfin.Function(V, name=name)
            fsubgroup = fgroup+"/{}".format(name)
            h5file.read(func, fsubgroup)

            setattr(geo, attrs[i], func)


def save_geometry_to_h5(mesh, h5name, h5group="", markers=None,
                        fields=None, local_basis=None, comm=None,
                        other_functions=None, other_attributes=None,
                        overwrite_file=False, overwrite_group=True):
    """
    Save geometry and other geometrical functions to a HDF file.

    Parameters
    ----------

    """

    logger.info("\nSave mesh to h5")
    assert isinstance(mesh, dolfin.Mesh)
    if comm is None:
        comm = mesh.mpi_comm()
    file_mode = "a" if os.path.isfile(h5name) and not overwrite_file else "w"

    # IF we should append the file but overwrite the group we need to
    # check that the group does not exist. If so we need to open it in
    # h5py and delete it.
    if file_mode == "a" and overwrite_group and h5group != "":
        io_utils.check_h5group(h5name, h5group, delete=True, comm=comm)

    with dolfin.HDF5File(comm, h5name, file_mode) as h5file:

        # Save mesh
        ggroup = '{}/geometry'.format(h5group)

        mgroup = '{}/mesh'.format(ggroup)

        h5file.write(mesh, mgroup)

        for dim in range(4):
            mf = dolfin.MeshFunction("size_t", mesh, dim, mesh.domains())
            save_mf = dolfin.MPI.max(comm, len(set(mf.array()))) > 1

            if save_mf:
                dgroup = '{}/mesh/meshfunction_{}'.format(ggroup, dim)
                h5file.write(mf, dgroup)

        if markers is not None:
            # Save the boundary markers
            for name, (marker, dim) in markers.items():

                for key_str in ["domain", "meshfunction"]:

                    dgroup = '{}/mesh/{}_{}'.format(ggroup, key_str, dim)

                    if h5file.has_dataset(dgroup):
                        aname = 'marker_name_{}'.format(name)
                        h5file.attributes(dgroup)[aname] = marker

        if local_basis is not None:
            # Save local basis functions
            lgroup = "{}/local basis functions".format(h5group)
            names = []
            for l in local_basis:
                h5file.write(l, lgroup + "/{}".format(l.name()))
                names.append(l.name())

            elm = l.function_space().ufl_element()
            family, degree = elm.family(), elm.degree()
            lspace = '{}_{}'.format(family, degree)
            h5file.attributes(lgroup)['space'] = lspace
            h5file.attributes(lgroup)['names'] = ":".join(names)

        if fields is not None:
            # Save fiber field
            fgroup = "{}/microstructure".format(h5group)
            names = []
            for field in fields:
                label = field.label() \
                        if field.label().rfind('a Function') == -1 else ""
                name = "_".join(filter(None, [str(field), label]))
                fsubgroup = "{}/{}".format(fgroup, name)
                h5file.write(field, fsubgroup)
                h5file.attributes(fsubgroup)['name'] = field.name()
                names.append(name)

            elm = field.function_space().ufl_element()
            family, degree = elm.family(), elm.degree()
            fspace = '{}_{}'.format(family, degree)
            h5file.attributes(fgroup)['space'] = fspace
            h5file.attributes(fgroup)['names'] = ":".join(names)

        for k, fun in other_functions.items():
            fungroup = "/".join([h5group, k])
            h5file.write(fun, fungroup)

            if isinstance(fun, dolfin.Function):
                elm = fun.function_space().ufl_element()
                family, degree, vsize \
                    = elm.family(), elm.degree(), elm.value_size()
                fspace = '{}_{}'.format(family, degree)
                h5file.attributes(fungroup)['space'] = fspace
                h5file.attributes(fungroup)['value_size'] = vsize

        for k, v in other_attributes.iteritems():
            if isinstance(v, str) and isinstance(k, str):
                h5file.attributes(h5group)[k] = v
            else:
                logger.warning("Invalid attribute {} = {}".format(k, v))

    logger.info("Geometry saved to {}".format(h5name))
