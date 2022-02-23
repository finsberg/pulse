from pathlib import Path

import dolfin
import h5py

from . import Function
from . import interpolate
from . import io_utils
from . import Mesh
from .utils import DOLFIN_VERSION_MAJOR
from .utils import getLogger
from .utils import mpi_comm_world

logger = getLogger(__name__)


def move(mesh, u, factor=1.0):
    """
    Move mesh according to some displacement times some factor
    """
    W = dolfin.VectorFunctionSpace(u.function_space().mesh(), "CG", 1)

    # Use interpolation for now. It is the only thing that makes sense
    u_int = interpolate(u, W)

    u0 = Function(W)
    # arr = factor * numpy_mpi.gather_vector(u_int.vector())
    # numpy_mpi.assign_to_vector(u0.vector(), arr)
    u0.vector()[:] = factor * u_int.vector()
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    U = Function(V)
    U.vector()[:] = u0.vector()
    # numpy_mpi.assign_to_vector(U.vector(), arr)

    dolfin.ALE.move(mesh, U)

    return u0


def get_geometric_dimension(mesh):
    try:
        return mesh.geometric_dimension()
    except AttributeError:
        return mesh.geometry().dim()


def load_geometry_from_h5(
    h5name,
    h5group="",
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
    :param bool include_sheets: Include sheets and cross-sheets
    :returns: An object with geometry data
    :rtype: object

    """
    h5name = Path(h5name)

    logger.info("\nLoad mesh from h5")
    # Set default groups
    ggroup = f"{h5group}/geometry"
    mgroup = f"{ggroup}/mesh"
    lgroup = f"{h5group}/local basis functions"
    fgroup = f"{h5group}/microstructure/"

    if not h5name.is_file():
        raise IOError(f"File {h5name} does not exist")

    # Check that the given file contains
    # the geometry in the given h5group
    if not io_utils.check_h5group(h5name, mgroup, delete=False, comm=comm):
        msg = ("Warning!\nGroup: '{}' does not exist in file:" "\n{}").format(
            mgroup,
            h5name,
        )

        with h5py.File(h5name) as h:
            keys = h.keys()
        msg += f"\nPossible values for the h5group are {keys}"
        raise IOError(msg)

    # Create a dummy object for easy parsing
    class Geometry(object):
        pass

    geo = Geometry()

    with dolfin.HDF5File(comm, h5name.as_posix(), "r") as h5file:

        # Load mesh
        mesh = Mesh(comm)
        io_utils.read_h5file(h5file, mesh, mgroup, True)
        geo.mesh = mesh

        # Get mesh functions
        for dim, attr in enumerate(["vfun", "efun", "ffun", "cfun"]):

            if dim > get_geometric_dimension(mesh):
                setattr(geo, attr, None)
                continue

            dgroup = f"{ggroup}/mesh/meshfunction_{dim}"
            mf = dolfin.MeshFunction("size_t", mesh, dim, mesh.domains())

            if h5file.has_dataset(dgroup):
                io_utils.read_h5file(h5file, mf, dgroup)
            setattr(geo, attr, mf)

        load_local_basis(h5file, lgroup, mesh, geo)
        load_microstructure(h5file, fgroup, mesh, geo, include_sheets)

        # Load the boundary markers
        markers = load_markers(h5file, mesh, ggroup, dgroup)
        geo.markers = markers

        origmeshgroup = f"{h5group}/original_geometry"
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
                dgroup = f"{ggroup}/mesh/{key_str}_{dim}"

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
                        marker = h5file.attributes(dgroup)[f"marker_name_{name}"]
                        markers[name] = (int(marker), dim)

    except Exception as ex:
        logger.info("Unable to load makers")
        logger.info(ex)
        markers = {}

    return markers


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

            io_utils.read_h5file(h5file, lb, lgroup + f"/{name}")
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
            # family = "Quadrature"
            # order = 4
            family = "CG"
            order = 1
        else:
            family, order = fspace.split("_")

        namesstr = fiber_attrs["names"]
        if namesstr is None:
            names = ["fiber"]
        else:
            names = namesstr.split(":")

        # Check that these fibers exists
        for name in names:
            fsubgroup = fgroup + f"/{name}"
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
            fsubgroup = fgroup + f"/{name}"

            io_utils.read_h5file(h5file, func, fsubgroup)

            setattr(geo, attrs[i], func)


def save_meshfunctions(h5file, h5group, mesh, meshfunctions, comm):
    for dim in range(get_geometric_dimension(mesh) + 1):

        if meshfunctions is not None and dim in meshfunctions:
            mf = meshfunctions[dim]
        else:
            mf = dolfin.MeshFunction("size_t", mesh, dim, mesh.domains())

        save_mf = dolfin.MPI.max(comm, len(set(mf.array()))) > 1

        if save_mf:
            dgroup = f"{ggroup(h5group)}/mesh/meshfunction_{dim}"
            h5file.write(mf, dgroup)


def save_markers(h5file, h5group, markers):
    if markers is None:
        return
    # Save the boundary markers
    for name, (marker, dim) in markers.items():

        for key_str in ["domain", "meshfunction"]:
            dgroup = f"{ggroup(h5group)}/mesh/{key_str}_{dim}"

            if h5file.has_dataset(dgroup):
                aname = f"marker_name_{name}"
                h5file.attributes(dgroup)[aname] = marker


def ggroup(h5group):
    return f"{h5group}/geometry"


def mgroup(h5group):
    return f"{ggroup(h5group)}/mesh"


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

    h5name = Path(h5name)
    file_mode = "a" if h5name.is_file() and not overwrite_file else "w"

    # IF we should append the file but overwrite the group we need to
    # check that the group does not exist. If so we need to open it in
    # h5py and delete it.
    if file_mode == "a" and overwrite_group:
        if h5group != "":
            io_utils.check_h5group(h5name, h5group, delete=True, comm=comm)
        else:
            h5name.unlink()
            file_mode = "w"

    with dolfin.HDF5File(comm, h5name.as_posix(), file_mode) as h5file:

        # Save mesh
        h5file.write(mesh, mgroup(h5group))
        save_meshfunctions(h5file, h5group, mesh, meshfunctions, comm)
        save_markers(h5file, h5group, markers)
        save_local_basis(h5file, h5group, local_basis)
        save_fields(h5file, h5group, fields)
        save_other_functions(h5file, h5group, other_functions)
        save_other_attributes(h5file, h5group, other_attributes)
    logger.info(f"Geometry saved to {h5name}")


def save_fields(h5file, h5group, fields):
    """Save mictrostructure to h5file

    Parameters
    ----------
    h5file : dolfin.HDF5File
        The file to write to
    h5group : str
        The folder inside the file to write to
    fields : list
        List of dolfin functions to write
    """
    if fields is None:
        return
    fgroup = f"{h5group}/microstructure"
    names = []
    for field in fields:
        try:
            label = field.label() if field.label().rfind("a Function") == -1 else ""
        except AttributeError:
            label = field.name()
        name = "_".join(filter(None, [str(field), label]))
        fsubgroup = f"{fgroup}/{name}"
        h5file.write(field, fsubgroup)
        h5file.attributes(fsubgroup)["name"] = field.name()
        names.append(name)

    elm = field.function_space().ufl_element()
    family, degree = elm.family(), elm.degree()
    fspace = f"{family}_{degree}"
    h5file.attributes(fgroup)["space"] = fspace
    h5file.attributes(fgroup)["names"] = ":".join(names)


def save_other_functions(h5file, h5group, other_functions):
    if other_functions is None:
        return
    for k, fun in other_functions.items():
        fungroup = "/".join([h5group, k])
        h5file.write(fun, fungroup)

        if isinstance(fun, dolfin.Function):
            elm = fun.function_space().ufl_element()
            family, degree, vsize = elm.family(), elm.degree(), elm.value_size()
            fspace = f"{family}_{degree}"
            h5file.attributes(fungroup)["space"] = fspace
            h5file.attributes(fungroup)["value_size"] = vsize


def save_other_attributes(h5file, h5group, other_attributes):
    if other_attributes is None:
        return
    for k, v in other_attributes.iteritems():
        if isinstance(v, str) and isinstance(k, str):
            h5file.attributes(h5group)[k] = v
        else:
            logger.warning(f"Invalid attribute {k} = {v}")


def save_local_basis(h5file, h5group, local_basis):
    """Save local basis functions

    Parameters
    ----------
    h5file : dolfin.HDF5File
        The file to write to
    h5group : str
        Folder inside file to store the local basis
    local_basis : list
        List of dolfin functions with local basis functions
    """
    if local_basis is None:
        return
    lgroup = f"{h5group}/local basis functions"
    names = []
    for basis in local_basis:
        h5file.write(basis, lgroup + f"/{basis.name()}")
        names.append(basis.name())

    elm = basis.function_space().ufl_element()
    family, degree = elm.family(), elm.degree()
    lspace = f"{family}_{degree}"
    h5file.attributes(lgroup)["space"] = lspace
    h5file.attributes(lgroup)["names"] = ":".join(names)
