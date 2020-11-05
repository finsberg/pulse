import os
import numpy as np
import h5py
import dolfin

try:
    from dolfin_adjoint import Constant, Function
except ImportError:
    from dolfin import Constant, Function

# For newer versions of dolfin_adjoint
try:
    from dolfin_adjoint import Mesh
except ImportError:
    from dolfin import Mesh

from . import numpy_mpi
from . import io_utils
from . import parameters
from .utils import make_logger, mpi_comm_world, DOLFIN_VERSION_MAJOR


logger = make_logger(__name__, parameters["log_level"])


def move(mesh, u, factor=1.0):
    """
    Move mesh according to some displacement times some factor
    """
    W = dolfin.VectorFunctionSpace(u.function_space().mesh(), "CG", 1)

    # Use interpolation for now. It is the only thing that makes sense
    u_int = dolfin.interpolate(u, W)

    u0 = dolfin.Function(W)
    # arr = factor * numpy_mpi.gather_vector(u_int.vector())
    # numpy_mpi.assign_to_vector(u0.vector(), arr)
    u0.vector()[:] = factor * u_int.vector()
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    U = dolfin.Function(V)
    U.vector()[:] = u0.vector()
    # numpy_mpi.assign_to_vector(U.vector(), arr)

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

        return {
            "BASE": 10,
            "ENDO": 30,
            "EPI": 40,
            "WALL": 50,
            "ENDORING": 300,
            "EPIRING": 400,
            "WALL": 50,
        }

    elif mesh_type == "biv":

        return {
            "BASE": 10,
            "ENDO_RV": 20,
            "ENDO_LV": 30,
            "EPI": 40,
            "ENDORING_RV": 200,
            "ENDORING_LV": 300,
            "EPIRING": 400,
            "WALL": 50,
        }


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


def load_geometry_from_h5(
    h5name,
    h5group="",
    fendo=None,
    fepi=None,
    include_sheets=True,
    comm=mpi_comm_world(),
):
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
    ggroup = "{}/geometry".format(h5group)
    mgroup = "{}/mesh".format(ggroup)
    lgroup = "{}/local basis functions".format(h5group)
    fgroup = "{}/microstructure/".format(h5group)

    if not os.path.isfile(h5name):
        raise IOError("File {} does not exist".format(h5name))

    # Check that the given file contains
    # the geometry in the given h5group
    if not io_utils.check_h5group(h5name, mgroup, delete=False, comm=comm):
        msg = ("Warning!\nGroup: '{}' does not exist in file:" "\n{}").format(
            mgroup, h5name
        )

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
        mesh = Mesh(comm)
        io_utils.read_h5file(h5file, mesh, mgroup, True)
        geo.mesh = mesh

        # Get mesh functions
        for dim, attr in enumerate(["vfun", "efun", "ffun", "cfun"]):

            if dim > mesh.geometric_dimension():
                setattr(geo, attr, None)
                continue

            dgroup = "{}/mesh/meshfunction_{}".format(ggroup, dim)
            mf = dolfin.MeshFunction("size_t", mesh, dim, mesh.domains())

            if h5file.has_dataset(dgroup):
                io_utils.read_h5file(h5file, mf, dgroup)
            setattr(geo, attr, mf)

        load_local_basis(h5file, lgroup, mesh, geo)
        load_microstructure(h5file, fgroup, mesh, geo, include_sheets)

        # Load the boundary markers
        markers = load_markers(h5file, mesh, ggroup, dgroup)
        geo.markers = markers

        origmeshgroup = "{}/original_geometry".format(h5group)
        if h5file.has_dataset(origmeshgroup):
            original_mesh = Mesh(comm)
            io_utils.read_h5file(h5file, original_mesh, origmeshgroup, True)
            setattr(geo, "original_geometry", original_mesh)

    for attr in ["f0", "s0", "n0", "r0", "c0", "l0", "cfun", "vfun", "efun", "ffun"]:
        if not hasattr(geo, attr):
            setattr(geo, attr, None)

    return geo


def load_markers(h5file, mesh, ggroup, dgroup):
    try:
        markers = {}
        for dim in range(mesh.ufl_domain().topological_dimension() + 1):
            for key_str in ["domain", "meshfunction"]:
                dgroup = "{}/mesh/{}_{}".format(ggroup, key_str, dim)

                # If dataset is not present
                if not h5file.has_dataset(dgroup):
                    continue

                def get_attributes():
                    if DOLFIN_VERSION_MAJOR >= 2018:
                        return h5file.attributes(dgroup).list_attributes()
                    else:
                        return h5file.attributes(dgroup).str().strip().split(" ")

                for aname in get_attributes():
                    if aname.startswith("marker_name"):

                        name = aname.rsplit("marker_name_")[-1]
                        marker = h5file.attributes(dgroup)[
                            "marker_name_{}".format(name)
                        ]
                        markers[name] = (int(marker), dim)

    except Exception as ex:
        logger.info("Unable to load makers")
        logger.info(ex)
        markers = get_markers()

    return markers


def setup_fiber_parameters():
    fiber_params = dolfin.Parameters("Fibers")
    fiber_params.add("fiber_space", "Quadrature_4")
    fiber_params.add("include_sheets", False)

    fiber_params.add("fiber_angle_epi", -60)
    fiber_params.add("fiber_angle_endo", 60)
    fiber_params.add("sheet_angle_epi", 0)
    fiber_params.add("sheet_angle_endo", 0)
    return fiber_params


def full_arctangent(x, y):
    t = np.arctan2(x, y)
    if t < 0:
        return t + 2 * np.pi
    else:
        return t


def cartesian_to_prolate_ellipsoidal(x, y, z, a):

    b1 = np.sqrt((x + a) ** 2 + y ** 2 + z ** 2)
    b2 = np.sqrt((x - a) ** 2 + y ** 2 + z ** 2)

    sigma = 1 / (2.0 * a) * (b1 + b2)
    tau = 1 / (2.0 * a) * (b1 - b2)
    phi = full_arctangent(z, y)
    mu = np.arccosh(sigma)
    nu = np.arccos(tau)
    return mu, nu, phi


def prolate_ellipsoidal_to_cartesian(mu, nu, phi, a):
    x = a * np.cosh(mu) * np.cos(nu)
    y = a * np.sinh(mu) * np.sin(nu) * np.cos(phi)
    z = a * np.sinh(mu) * np.sin(nu) * np.sin(phi)
    return x, y, z


def fill_coordinates_ec(i, e_c_x, e_c_y, e_c_z, coord, foci):
    norm = dolfin.sqrt(coord[1] ** 2 + coord[2] ** 2)
    if not dolfin.near(norm, 0):
        e_c_y.vector()[i] = -coord[2] / norm
        e_c_z.vector()[i] = coord[1] / norm
    else:
        # We are at the apex where crl system doesn't make sense
        # So just pick something.
        e_c_y.vector()[i] = 1
        e_c_z.vector()[i] = 0


def fill_coordinates_el(i, e_c_x, e_c_y, e_c_z, coord, foci):

    norm = dolfin.sqrt(coord[1] ** 2 + coord[2] ** 2)
    if not dolfin.near(norm, 0):
        mu, nu, phi = cartesian_to_prolate_ellipsoidal(*(coord.tolist() + [foci]))
        x, y, z = prolate_ellipsoidal_to_cartesian(mu, nu + 0.01, phi, foci)
        r = np.array([coord[0] - x, coord[1] - y, coord[2] - z])
        e_r = r / np.linalg.norm(r)
        e_c_x.vector()[i] = e_r[0]
        e_c_y.vector()[i] = e_r[1]
        e_c_z.vector()[i] = e_r[2]
    else:
        e_c_y.vector()[i] = 0
        e_c_z.vector()[i] = 1


def calc_cross_products(e1, e2, VV):
    e_crossed = dolfin.Function(VV)

    e1_arr = e1.vector().get_local().reshape((-1, 3))
    e2_arr = e2.vector().get_local().reshape((-1, 3))

    crosses = []
    for c1, c2 in zip(e1_arr, e2_arr):
        crosses.extend(np.cross(c1, c2.tolist()))

    e_crossed.vector()[:] = np.array(crosses)[:]
    return e_crossed


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


def make_crl_basis(mesh, foc):
    """
    Makes the crl  basis for the LV mesh (prolate ellipsoidal)
    with prespecified focal length.
    """

    VV = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    V = dolfin.FunctionSpace(mesh, "CG", 1)

    if DOLFIN_VERSION_MAJOR > 1.6:
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


def fibers(mesh, fiber_endo=60, fiber_epi=-60):

    fiber_params = setup_fiber_parameters()
    fiber_params["fiber_angle_endo"] = fiber_endo
    fiber_params["fiber_angle_epi"] = fiber_epi

    f0 = generate_fibers(mesh, fiber_params)[0]
    return f0


def get_circ_field(mesh):

    fiber_params = dolfin.Parameters("Fibers")
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

    fiber_params = setup_fiber_parameters()
    fiber_params["fiber_angle_endo"] = -90
    fiber_params["fiber_angle_epi"] = -90

    if mesh_type == "biv":
        # We need to set the markers for then LV and RV facets
        ffun = dolfin.MeshFunction("size_t", mesh, 2, mesh.domains())

        markers = get_fiber_markers("biv")
        # Mark the mesh with same markers on the LV and RV before
        # running the LDRB algorithm
        markers["ENDO_RV"] = markers["ENDO_LV"]
        for facet in doflin.facets(mesh):
            if ffun[facet] != 0:
                mesh.domains().set_marker((facet.index(), markers[ffun[facet]]), 2)

    f0 = generate_fibers(mesh, fiber_params)[0]
    f0.rename("longitudinal", "local_basis_function")

    if mesh_type == "biv":
        # Put the correct markers
        markers = get_fiber_markers("biv")
        for facet in doflin.facets(mesh):
            if ffun[facet] != 0:
                mesh.domains().set_marker((facet.index(), markers[ffun[facet]]), 2)
    return f0


def generate_fibers(mesh, fiber_params, ffun=None):
    """
    Generate fibers on mesh based on provided parameters.
    It is not recomemmended to use this function.
    Use the `ldrb` package directly instead. This function is
    mainly used to ensure version compatability.
    """

    try:
        from ldrb import dolfin_ldrb
    except ImportError:
        msg = (
            '"ldrb" package not found. Please go to '
            "https://github.com/finsberg/ldrb to see how you can get it!"
        )
        print(msg)
        raise ImportError(msg)

    if isinstance(fiber_params, dolfin.Parameters):
        p = fiber_params.to_dict()
    else:
        p = fiber_params

    angles = dict(
        alpha_endo_lv=p.get("fiber_angle_endo"), alpha_epi_lv=p.get("fiber_angle_epi")
    )
    for name, a in angles.items():
        if isinstance(a, dolfin.cpp.parameter.Parameter):
            angles[name] = a.value()

    return dolfin_ldrb(mesh, ffun=ffun, **angles)


def generate_fibers_old(mesh, fiber_params):

    try:
        from fiberrules import dolfin_fiberrules
    except ImportError as ex:
        logger.error("Unable to generate fibers without fiberrules package")
        logger.error("This package is protected by copyright")
        raise ex

    logger.info("\nGENERATING FIBERS ...")

    fiber_space_name = fiber_params["fiber_space"]

    assert (
        len(fiber_space_name.split("_")) == 2
    ), "expected fiber_space_name in 'FamilyName_Degree' format"

    family, degree = fiber_space_name.split("_")

    if DOLFIN_VERSION_MAJOR > 1.6:
        el = dolfin.FiniteElement(
            family=family,
            cell=mesh.ufl_cell(),
            degree=int(degree),
            quad_scheme="default",
        )
        fiber_space = dolfin.FunctionSpace(mesh, el)
    else:
        fiber_space = dolfin.FunctionSpace(mesh, family, int(degree))

    if family == "Quadrature":
        dolfin.parameters["form_compiler"]["quadrature_degree"] = int(degree)
        if DOLFIN_VERSION_MAJOR > 2016:
            dolfin.parameters["form_compiler"]["representation"] = "quadrature"

    # There are some strange shifting in the fiberrults angles
    fiber_angle_epi = 90 - (-fiber_params["fiber_angle_epi"])
    fiber_angle_endo = 90 - (fiber_params["fiber_angle_endo"])
    sheet_angle_endo = fiber_params.to_dict().get("sheet_angle_endo", 0)
    sheet_angle_epi = fiber_params.to_dict().get("sheet_angle_epi", 0)

    microstructures = dolfin_fiberrules(
        mesh,
        fiber_space,
        fiber_angle_epi,
        fiber_angle_endo,
        sheet_angle_epi,
        sheet_angle_endo,
    )

    if DOLFIN_VERSION_MAJOR > 2016:
        dolfin.parameters["form_compiler"]["representation"] = "uflacs"

    microstructures[0].rename(
        "fiber",
        "epi{}_endo{}".format(
            fiber_params["fiber_angle_epi"], fiber_params["fiber_angle_endo"]
        ),
    )
    fields = [microstructures[0]]

    if fiber_params["include_sheets"]:
        microstructures[1].rename(
            "sheet", "epi{}_endo{}".format(sheet_angle_epi, sheet_angle_endo)
        )
        microstructures[2].rename(
            "cross_sheet",
            "fepi{}_fendo{}_sepi{}_sendo{}".format(
                fiber_params["fiber_angle_epi"],
                fiber_params["fiber_angle_endo"],
                sheet_angle_epi,
                sheet_angle_endo,
            ),
        )
        fields.append(microstructures[1])
        fields.append(microstructures[2])

    return fields


def load_local_basis(h5file, lgroup, mesh, geo):

    if h5file.has_dataset(lgroup):
        # Get local bais functions
        local_basis_attrs = h5file.attributes(lgroup)
        lspace = local_basis_attrs["space"]
        family, order = lspace.split("_")

        namesstr = local_basis_attrs["names"]
        names = namesstr.split(":")

        if DOLFIN_VERSION_MAJOR > 1.6:
            elm = dolfin.VectorElement(
                family=family,
                cell=mesh.ufl_cell(),
                degree=int(order),
                quad_scheme="default",
            )
            V = dolfin.FunctionSpace(mesh, elm)
        else:
            V = dolfin.VectorFunctionSpace(mesh, family, int(order))

        for name in names:
            lb = Function(V, name=name)

            io_utils.read_h5file(h5file, lb, lgroup + "/{}".format(name))
            setattr(geo, name, lb)
    else:
        setattr(geo, "circumferential", None)
        setattr(geo, "radial", None)
        setattr(geo, "longitudinal", None)


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
            family, order = fspace.split("_")

        namesstr = fiber_attrs["names"]
        if namesstr is None:
            names = ["fiber"]
        else:
            names = namesstr.split(":")

        # Check that these fibers exists
        for name in names:
            fsubgroup = fgroup + "/{}".format(name)
            if not h5file.has_dataset(fsubgroup):
                msg = ("H5File does not have dataset {}").format(fsubgroup)
                logger.warning(msg)

        if DOLFIN_VERSION_MAJOR > 1.6:
            elm = dolfin.VectorElement(
                family=family,
                cell=mesh.ufl_cell(),
                degree=int(order),
                quad_scheme="default",
            )
            V = dolfin.FunctionSpace(mesh, elm)
        else:
            V = dolfin.VectorFunctionSpace(mesh, family, int(order))

        attrs = ["f0", "s0", "n0"]
        for i, name in enumerate(names):
            func = Function(V, name=name)
            fsubgroup = fgroup + "/{}".format(name)

            io_utils.read_h5file(h5file, func, fsubgroup)

            setattr(geo, attrs[i], func)


def save_geometry_to_h5(
    mesh,
    h5name,
    h5group="",
    markers=None,
    fields=None,
    local_basis=None,
    meshfunctions=None,
    comm=None,
    other_functions=None,
    other_attributes=None,
    overwrite_file=False,
    overwrite_group=True,
):
    """
    Save geometry and other geometrical functions to a HDF file.

    Parameters
    ----------
    mesh : :class:`dolfin.mesh`
        The mesh
    h5name : str
        Path to the file
    h5group : str
        Folder within the file. Default is "" which means in
        the top folder.
    markers : dict
        A dictionary with markers. See `get_markers`.
    fields : list
        A list of functions for the microstructure
    local_basis : list
        A list of functions for the crl basis
    meshfunctions : dict
        A dictionary with keys being the dimensions the the values
        beeing the meshfunctions.
    comm : :class:`dolfin.MPI`
        MPI communicator
    other_functions : dict
        Dictionary with other functions you want to save
    other_attributes: dict
        Dictionary with other attributes you want to save
    overwrite_file : bool
        If true, and the file exists, the file will be overwritten (default: False)
    overwrite_group : bool
        If true and h5group exist, the group will be overwritten.

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
        ggroup = "{}/geometry".format(h5group)

        mgroup = "{}/mesh".format(ggroup)

        h5file.write(mesh, mgroup)

        for dim in range(mesh.geometric_dimension() + 1):

            if meshfunctions is not None and dim in meshfunctions:
                mf = meshfunctions[dim]
            else:
                mf = dolfin.MeshFunction("size_t", mesh, dim, mesh.domains())

            save_mf = dolfin.MPI.max(comm, len(set(mf.array()))) > 1

            if save_mf:
                dgroup = "{}/mesh/meshfunction_{}".format(ggroup, dim)
                h5file.write(mf, dgroup)

        if markers is not None:
            # Save the boundary markers
            for name, (marker, dim) in markers.items():

                for key_str in ["domain", "meshfunction"]:

                    dgroup = "{}/mesh/{}_{}".format(ggroup, key_str, dim)

                    if h5file.has_dataset(dgroup):
                        aname = "marker_name_{}".format(name)
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
            lspace = "{}_{}".format(family, degree)
            h5file.attributes(lgroup)["space"] = lspace
            h5file.attributes(lgroup)["names"] = ":".join(names)

        if fields is not None:
            # Save fiber field
            fgroup = "{}/microstructure".format(h5group)
            names = []
            for field in fields:
                try:
                    label = (
                        field.label() if field.label().rfind("a Function") == -1 else ""
                    )
                except AttributeError:
                    label = field.name()
                name = "_".join(filter(None, [str(field), label]))
                fsubgroup = "{}/{}".format(fgroup, name)
                h5file.write(field, fsubgroup)
                h5file.attributes(fsubgroup)["name"] = field.name()
                names.append(name)

            elm = field.function_space().ufl_element()
            family, degree = elm.family(), elm.degree()
            fspace = "{}_{}".format(family, degree)
            h5file.attributes(fgroup)["space"] = fspace
            h5file.attributes(fgroup)["names"] = ":".join(names)

        if other_functions is not None:
            for k, fun in other_functions.items():
                fungroup = "/".join([h5group, k])
                h5file.write(fun, fungroup)

            if isinstance(fun, dolfin.Function):
                elm = fun.function_space().ufl_element()
                family, degree, vsize = elm.family(), elm.degree(), elm.value_size()
                fspace = "{}_{}".format(family, degree)
                h5file.attributes(fungroup)["space"] = fspace
                h5file.attributes(fungroup)["value_size"] = vsize

        if other_attributes is not None:
            for k, v in other_attributes.iteritems():
                if isinstance(v, str) and isinstance(k, str):
                    h5file.attributes(h5group)[k] = v
                else:
                    logger.warning("Invalid attribute {} = {}".format(k, v))

    logger.info("Geometry saved to {}".format(h5name))


def mark_strain_regions(mesh, foc=None, nsectors=(6, 6, 4, 1), mark_mesh=True):
    """Mark the cells in the mesh.

    For instance if you want to mark this mesh accoring to
    the  AHA 17-segment model, then nsector = [6,6,4,1],
    i.e 6 basal, 6 mid, 4 apical and one apex

     """

    fun = dolfin.MeshFunction("size_t", mesh, 3)
    nlevels = len(nsectors)

    pi = np.pi

    assert nlevels <= 4

    if nlevels == 4:
        mus = [90, 60, 30, 10, 0]
    elif nlevels == 3:
        mus = [90, 60, 30, 0]
    elif nlevels == 2:
        mus = [90, 45, 0]
    else:
        mus = [90, 0]

    thetas = [np.linspace(pi, 3 * pi, s + 1)[:-1].tolist() + [pi] for s in nsectors]

    start = 0
    end = nsectors[0]
    regions = np.zeros((sum(nsectors), 4))
    for i in range(nlevels):
        regions.T[0][start:end] = mus[i] * pi / 180
        regions.T[3][start:end] = mus[i + 1] * pi / 180
        if i != len(nsectors) - 1:
            start += nsectors[i]
            end += nsectors[i + 1]

    start = 0
    for j, t in enumerate(thetas):
        for i in range(nsectors[j]):
            regions.T[1][i + start] = t[i]
            regions.T[2][i + start] = t[i + 1]
        start += nsectors[j]

    sfun = mark_cell_function(fun, mesh, foc, regions)

    if mark_mesh:
        # Mark the cells accordingly
        for cell in dolfin.cells(mesh):
            mesh.domains().set_marker((cell.index(), sfun[cell]), 3)

    return sfun


def mark_cell_function(fun, mesh, foc, regions):
    """
    Iterates over the mesh and stores the
    region number in a meshfunction
    """

    if foc is None:
        foc = estimate_focal_point(mesh)

    for cell in dolfin.cells(mesh):

        # Get coordinates to cell midpoint
        x = cell.midpoint().x()
        y = cell.midpoint().y()
        z = cell.midpoint().z()

        T = cartesian_to_prolate_ellipsoidal(x, y, z, foc)

        fun[cell] = strain_region_number(T, regions)

    return fun


def strain_region_number(T, regions):
    """
    For a given point in prolate coordinates,
    return the region it belongs to.

    :param regions: Array of all coordinates for the strain 
                    regions taken from the strain mesh.
    :type regions: :py:class:`numpy.array`

    :param T: Some value i prolate coordinates
    :type T: :py:class:`numpy.array`

    Resturn the region number that
    T belongs to
    """

    """
    The cricumferential direction is a bit
    tricky because it goes from -pi to pi.
    To overcome this we add pi so that the
    direction goes from 0 to 2*pi
    """

    lam, mu, theta = T

    theta = theta + np.pi

    levels = get_level(regions, mu)

    if np.shape(regions)[0] + 1 in levels:
        return np.shape(regions)[0] + 1

    sector = get_sector(regions, theta)

    assert len(np.intersect1d(levels, sector)) == 1

    return np.intersect1d(levels, sector)[0] + 1


def get_level(regions, mu):

    A = np.intersect1d(
        np.where((regions.T[3] <= mu))[0], np.where((mu <= regions.T[0]))[0]
    )
    if len(A) == 0:
        return [np.shape(regions)[0] + 1]
    else:
        return A


def get_sector(regions, theta):

    if not (
        np.count_nonzero(regions.T[1] <= regions.T[2]) >= 0.5 * np.shape(regions)[0]
    ):
        raise ValueError("Surfaces are flipped")

    sectors = []
    for i, r in enumerate(regions):

        if r[1] == r[2]:
            sectors.append(i)
        else:
            if r[1] > r[2]:
                if theta > r[1] or r[2] > theta:
                    sectors.append(i)

            else:
                if r[1] < theta < r[2]:
                    sectors.append(i)

    return sectors


def estimate_focal_point(mesh):
    """Copmute the focal point based on approximating the 
    endocardial surfaces as a ellipsoidal cap.
    
    .. math::

           focal = \sqrt{ l^2 - s^2}


    Arguments
    ---------
    mesh: `dolfin.mesh`
        The mesh

    Returns
    -------
    focal_point: float
        The focal point

    """

    max_coord = np.max(mesh.coordinates(), 0)
    min_coord = np.min(mesh.coordinates(), 0)

    axis = np.abs(max_coord - min_coord)
    long_axis = np.max(axis)
    short_axis = np.min(axis)

    focal = np.sqrt(long_axis ** 2 - short_axis ** 2)

    return focal
