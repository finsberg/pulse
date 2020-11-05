import numpy as np
import dolfin

try:
    from dolfin_adjoint import (
        Function,
        interpolate,
        Constant,
        project,
        assemble,
        FunctionAssigner,
    )
except ImportError:
    from dolfin import (
        Function,
        interpolate,
        Constant,
        project,
        assemble,
        FunctionAssigner,
    )


from . import utils
from . import numpy_mpi

from .utils import logger, mpi_comm_world, DOLFIN_VERSION_MAJOR


def map_vector_field(f0, new_mesh, u=None, name="fiber", normalize=True):
    """
    Map a vector field (f0) onto a new mesh (new_mesh) where the new mesh
    can be a moved version of the original one according to some
    displacement (u). In that case we will just a Piola transform to
    map the vector field.
    """

    if DOLFIN_VERSION_MAJOR > 2016:
        dolfin.parameters["form_compiler"]["representation"] = "quadrature"

    dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

    ufl_elem = f0.function_space().ufl_element()
    f0_new = dolfin.Function(dolfin.FunctionSpace(new_mesh, ufl_elem))

    if u is not None:

        f0_mesh = f0.function_space().mesh()
        u_elm = u.function_space().ufl_element()
        V = dolfin.FunctionSpace(f0_mesh, u_elm)
        u0 = dolfin.Function(V)
        # arr = numpy_mpi.gather_vector(u.vector())
        # numpy_mpi.assign_to_vector(u0.vector(), arr)
        u0.vector()[:] = u.vector()
        from .kinematics import DeformationGradient

        F = DeformationGradient(u0)

        f0_updated = dolfin.project(F * f0, f0.function_space())

        if normalize:
            f0_updated = normalize_vector_field(f0_updated)

        f0_new.vector()[:] = f0_updated.vector()
        # f0_arr = numpy_mpi.gather_vector(f0_updated.vector())
        # numpy_mpi.assign_to_vector(f0_new.vector(), f0_arr)

    else:
        # f0_arr = numpy_mpi.gather_vector(f0.vector())
        # numpy_mpi.assign_to_vector(f0_new.vector(), f0_arr)
        f0_new.vector()[:] = f0.vector()

    if DOLFIN_VERSION_MAJOR > 2016:
        dolfin.parameters["form_compiler"]["representation"] = "uflacs"

    return f0_new


def update_function(mesh, f):
    """Given a function :math:`f` defined on some domain,
    update the function so that it now is defined on the domain
    given in the mesh
    """

    f_new = Function(dolfin.FunctionSpace(mesh, f.ufl_element()))
    numpy_mpi.assign_to_vector(f_new.vector(), numpy_mpi.gather_vector(f.vector()))
    return f_new


def normalize_vector_field(u):
    """Given a vector field, return a vector field with an L2 norm equal to 1.0
    """
    dim = len(u)
    S = u.function_space().sub(0).collapse()

    components = vectorfield_to_components(u, S, dim)

    normarray = np.sqrt(sum(components[i].vector().norm("l2") ** 2 for i in range(dim)))

    for comp in components:
        comp.vector()[:] = comp.vector() / normarray

    assigners = [FunctionAssigner(u.function_space().sub(i), S) for i in range(dim)]
    for i, comp, assigner in zip(range(dim), components, assigners):
        assigner.assign(u.sub(i), comp)

    return u


def vectorfield_to_components(u, S, dim):
    components = [dolfin.Function(S) for i in range(dim)]
    assigners = [FunctionAssigner(S, u.function_space().sub(i)) for i in range(dim)]
    for i, comp, assigner in zip(range(dim), components, assigners):
        assigner.assign(comp, u.sub(i))

    return components


def get_pressure(problem):
    """Returns p_lv (and p_rv if BiV mesh)
    """

    plv = [p.traction for p in problem.bcs.neumann if p.name == "lv"]
    prv = [p.traction for p in problem.bcs.neumann if p.name == "rv"]

    assert len(plv) > 0, "Problem has no Neumann BC for LV endo"
    pressure = [plv[0]]
    if prv:
        pressure.append(prv[0])
        return tuple(pressure)
    else:
        return pressure[0]


def read_hdf5(h5name, func, h5group="", comm=mpi_comm_world()):

    try:
        with dolfin.HDF5File(comm, h5name, "r") as h5file:

            h5file.read(func, h5group)

    except IOError as ex:
        logger.error(ex)
        logger.error("Make sure file {} exist".format(h5name))
        raise ex

    except RuntimeError as ex:
        logger.errro(ex)
        logger.error(
            (
                "Something went wrong when reading file "
                "{h5name} into function {func} from group "
                "{h5group)"
            ).format(h5name=h5name, func=func, h5group=h5group)
        )
        raise ex


def map_displacement(u, old_space, new_space, approx, name="mapped displacement"):

    if approx == "interpolate":
        # Do we need dolfin-adjoint here or is dolfin enough?
        u_int = interpolate(project(u, old_space), new_space)  # , name=name)

    elif approx == "project":
        # Do we need dolfin-adjoint here or is dolfin enough?
        u_int = project(u, new_space)  # , name=name)

    else:
        u_int = u

    return u_int


def compute_meshvolume(domain=None, dx=dolfin.dx, subdomain_id=None):
    return Constant(
        dolfin.assemble(
            dolfin.Constant(1.0) * dx(domain=domain, subdomain_id=subdomain_id)
        )
    )


def get_cavity_volume(geometry, unload=False, chamber="lv", u=None, xshift=0.0):

    if unload:
        mesh = geometry.original_geometry
        ffun = dolfin.MeshFunction("size_t", mesh, 2, mesh.domains())
    else:
        mesh = geometry.mesh
        ffun = geometry.ffun

    if chamber == "lv":
        if "ENDO_LV" in geometry.markers:
            endo_marker = geometry.markers["ENDO_LV"]
        else:
            endo_marker = geometry.markers["ENDO"]

    else:
        endo_marker = geometry.markers["ENDO_RV"]

    if hasattr(endo_marker, "__len__"):
        endo_marker = endo_marker[0]

    ds = dolfin.Measure("exterior_facet", subdomain_data=ffun, domain=mesh)(endo_marker)

    vol_form = get_cavity_volume_form(geometry.mesh, u, xshift)
    return assemble(vol_form * ds)


def get_cavity_volume_form(mesh, u=None, xshift=0.0):

    from . import kinematics

    shift = Constant((xshift, 0.0, 0.0))
    X = dolfin.SpatialCoordinate(mesh) - shift
    N = dolfin.FacetNormal(mesh)

    if u is None:
        vol_form = (-1.0 / 3.0) * dolfin.dot(X, N)
    else:
        F = kinematics.DeformationGradient(u)
        J = kinematics.Jacobian(F)
        vol_form = (-1.0 / 3.0) * dolfin.dot(X + u, J * dolfin.inv(F).T * N)

    return vol_form


def get_constant(val, value_size=None, value_rank=0, constant=Constant):

    if isinstance(val, (Constant, dolfin.Constant)):
        return val
    elif isinstance(val, (Function, dolfin.Function)):
        arr = numpy_mpi.gather_vector(val.vector())
        return constant(arr)
    elif isinstance(val, dolfin.GenericVector):
        arr = numpy_mpi.gather_vector(val)
        return constant(arr)

    if value_size is None:
        if np.isscalar(val):
            value_size = 1
        else:
            try:
                value_size = len(val)
                val = np.array(val)
            except Exception as ex:
                logger.debug(ex)
                # Hope for the best
                value_size = 1
    if value_size == 1:
        if value_rank == 0:
            c = constant(val)
        else:
            c = constant([val])
    else:
        c = constant([val] * value_size)

    return c


def get_dimesion(u):

    # TODO : Check argument
    try:
        if DOLFIN_VERSION_MAJOR > 1.6:
            from ufl.domain import find_geometric_dimension

            dim = find_geometric_dimension(u)
        else:
            dim = u.geometric_dimension()

    except Exception as ex:

        try:
            dim = len(u)
        except Exception as ex2:
            logger.warning(ex)
            logger.warning(ex2)
            # Assume dimension is 3
            logger.warning("Assume dimension is 3")
            dim = 3

    return dim


def subplus(x):
    r"""
    Ramp function

    .. math::

       \max\{x,0\}

    """

    return dolfin.conditional(dolfin.ge(x, 0.0), x, 0.0)


def heaviside(x):
    r"""
    Heaviside function

    .. math::

       \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}

    """

    return dolfin.conditional(dolfin.ge(x, 0.0), 1.0, 0.0)


def list_sum(l):
    """
    Return the sum of a list, when the convetiional
    method (like `sum`) it not working.
    For example if you have a list of dolfin functions.

    :param list l: a list of objects
    :returns: The sum of the list. The type depends on
              the type of elemets in the list

    """

    if not isinstance(l, list):
        return l

    out = l[0]
    for item in l[1:]:
        out += item
    return out


def get_spaces(mesh):
    """
    Return an object of dolfin FunctionSpace, to
    be used in the optimization pipeline

    :param mesh: The mesh
    :type mesh: :py:class:`dolfin.Mesh`
    :returns: An object of functionspaces
    :rtype: object

    """

    # Make a dummy object
    spaces = utils.Object()

    # A real space with scalars used for dolfin adjoint
    spaces.r_space = dolfin.FunctionSpace(mesh, "R", 0)

    # A space for the strain fields
    spaces.strainfieldspace = dolfin.VectorFunctionSpace(mesh, "CG", 1, dim=3)

    # A space used for scalar strains
    spaces.strainspace = dolfin.VectorFunctionSpace(mesh, "R", 0, dim=3)

    # Spaces for the strain weights
    spaces.strain_weight_space = dolfin.TensorFunctionSpace(mesh, "R", 0)

    return spaces


def QuadratureSpace(mesh, degree, dim=3):
    """
    From FEniCS version 1.6 to 2016.1 there was a change in how
    FunctionSpace is defined for quadrature spaces.
    This functions checks your dolfin version and returns the correct
    quadrature space

    :param mesh: The mesh
    :type mesh: :py:class:`dolfin.Mesh`
    :param int degree: The degree of the element
    :param int dim: For a mesh of topological dimension 3,
                    dim = 1 would be a scalar function, and
                    dim = 3 would be a vector function.
    :returns: The quadrature space
    :rtype: :py:class:`dolfin.FunctionSpace`

    """

    if DOLFIN_VERSION_MAJOR > 1.6:
        if dim == 1:
            element = dolfin.FiniteElement(
                family="Quadrature",
                cell=mesh.ufl_cell(),
                degree=degree,
                quad_scheme="default",
            )
        else:
            element = dolfin.VectorElement(
                family="Quadrature",
                cell=mesh.ufl_cell(),
                degree=degree,
                quad_scheme="default",
            )

        return dolfin.FunctionSpace(mesh, element)
    else:
        if dim == 1:
            return dolfin.FunctionSpace(mesh, "Quadrature", degree)
        else:
            return dolfin.VectorFunctionSpace(mesh, "Quadrature", degree)


class VertexDomain(dolfin.SubDomain):
    """
    A subdomain defined in terms of
    a given set of coordinates.
    A point that is close to the given coordinates
    within a given tolerance will be marked as inside
    the domain.
    """

    def __init__(self, coords, tol=1e-4):
        """
        *Arguments*
          coords (list)
            List of coordinates for vertices in reference geometry
            defining this domains

          tol (float)
            Tolerance for how close a pointa should be to the given coordinates
            to be marked as inside the domain
        """

        self.coords = np.array(coords)
        self.tol = tol
        dolfin.SubDomain.__init__(self)

    def inside(self, x, on_boundary):

        if np.all([np.any(abs(x[i] - self.coords.T[i]) < self.tol) for i in range(3)]):
            return True

        return False


class BaseExpression(dolfin.Expression):
    """
    A class for assigning boundary condition according to segmented surfaces
    Since the base is located at x = a (usually a=0), two classes must be set:
    One for the y-direction and one for the z-direction

    Point on the endocardium and epicardium is given and the
    points on the mesh base is set accordingly.
    Points that lie on the base but not on the epi- or endoring
    will be given a zero value.
    """

    def __init__(self, mesh_verts, seg_verts, sub, it, name):
        """

        *Arguments*
          mesh: (dolfin.mesh)
            The mesh

          u: (dolfin.GenericFunction)
            Initial displacement

          mesh_verts (numpy.ndarray or list)
            Point of endocardial base from mesh

          seg_verts (numpy.ndarray or list)
            Point of endocardial base from segmentation

          sub (str)
            Either "y" or "z". The displacement in this direction is returned

          it (dolfin.Expression)
            Can be used to incrment the direclet bc

        """

        assert sub in ["y", "z"]
        self._mesh_verts = np.array(mesh_verts)
        self._all_seg_verts = np.array(seg_verts)
        self.point = 0
        self.npoints = len(seg_verts) - 1

        self._seg_verts = self._all_seg_verts[0]

        self._sub = sub
        self._it = it
        self.rename(name, name)

    def next(self):
        self._it.t = 0
        self.point += 1
        self._seg_verts = self._all_seg_verts[self.point]

    def reset(self):
        self.point = 0
        self._it.t = 0

    def eval(self, value, x):

        # Check if given coordinate is in the endoring vertices
        # and find the cooresponding index
        d = [np.where(x[i] == self._mesh_verts.T[i])[0] for i in range(3)]
        d_intersect = set.intersection(*map(set, d))
        assert len(d_intersect) < 2
        if len(d_intersect) == 1:

            idx = d_intersect.pop()
            prev_seg_verts = self._all_seg_verts[self.point - 1]

            # Return the displacement in the given direction
            # Iterated starting from the previous displacemet
            # to the current one
            if self._sub == "y":
                u_prev = self._mesh_verts[idx][1] - prev_seg_verts[idx][1]
                u_current = self._mesh_verts[idx][1] - self._seg_verts[idx][1]
                # value[0] = u_prev + self._it.t*(u_current - u_prev)
            else:  # sub == "z"
                u_prev = self._mesh_verts[idx][2] - prev_seg_verts[idx][2]
                u_current = self._mesh_verts[idx][2] - self._seg_verts[idx][2]

            val = u_prev + self._it.t * (u_current - u_prev)
            value[0] = val

        else:
            value[0] = 0


class MixedParameter(dolfin.Function):
    def __init__(self, fun, n, name="material_parameters"):
        """
        Initialize Mixed parameter.

        This will instanciate a function in a dolfin.MixedFunctionSpace
        consiting of `n` subspaces of the same type as `fun`.
        This is of course easy for the case when `fun` is a normal
        dolfin function, but in the case of a `RegionalParameter` it
        is not that straight forward.
        This class handles this case as well.


        :param fun: The type of you want to make a du
        :type fun: (:py:class:`dolfin.Function`)
        :param int n: number of subspaces
        :param str name: Name of the function

        .. todo::

           Implement support for MixedParameter with different
           types of subspaces, e.g [RegionalParamter, R_0, CG_1]

        """

        msg = "Please provide a dolin function as argument to MixedParameter"
        assert isinstance(fun, (dolfin.Function, Function, RegionalParameter)), msg

        if isinstance(fun, RegionalParameter):
            raise NotImplementedError

        # We can just make a usual mixed function space
        # with n copies of the original one
        V = fun.function_space()
        W = dolfin.MixedFunctionSpace([V] * n)

        Function.__init__(self, W, name=name)

        # Create a function assigner
        self.function_assigner = [FunctionAssigner(W.sub(i), V) for i in range(n)]

        # Store the original function space
        self.basespace = V

        if isinstance(fun, RegionalParameter):
            self._meshfunction = fun._meshfunction

    def assign_sub(self, f, i):
        """
        Assign subfunction

        :param f: The function you want to assign
        :param int i: The subspace number

        """
        f_ = dolfin.Function(self.basespace)
        f_.assign(f)
        self.function_assigner[i].assign(self.split()[i], f_)


class RegionalParameter(dolfin.Function):
    """A regional paramerter defined in terms of a dolfin.MeshFunction

    Suppose you have a MeshFunction defined different regions in your mesh,
    and you want to define different parameters on different regions,
    then this is what you want.
    """

    def __init__(self, meshfunction):

        # assert isinstance(
        #     meshfunction, dolfin.MeshFunctionSizet
        # ), "Invalid meshfunction for regional gamma"

        mesh = meshfunction.mesh()

        # FIXME
        self._values = set(numpy_mpi.gather_broadcast(meshfunction.array()))
        self._nvalues = len(self._values)

        V = dolfin.VectorFunctionSpace(mesh, "R", 0, dim=self._nvalues)

        dolfin.Function.__init__(self, V)
        self._meshfunction = meshfunction

        # Functionspace for the indicator functions
        self._proj_space = dolfin.FunctionSpace(mesh, "DG", 0)

        # Make indicator functions
        self._ind_functions = []
        for v in self._values:
            self._ind_functions.append(self._make_indicator_function(v))

    def get_values(self):
        return self._values

    @property
    def function(self):
        """
        Return linear combination of coefficents
        and basis functions

        :returns: A function with parameter values at each segment
                  specified by the meshfunction
        :rtype:  :py:class`dolfin.Function             
             
        """
        return self._sum()

    @property
    def proj_space(self):
        """
        Space for projecting the scalars.
        This is a DG 0 space.
        """
        return self._proj_space

    def _make_indicator_function(self, marker):

        dofs = self._meshfunction.where_equal(marker)
        f = dolfin.Function(self._proj_space)
        f.vector()[dofs] = 1.0
        return f

    def _sum(self):
        coeffs = dolfin.split(self)
        fun = coeffs[0] * self._ind_functions[0]

        for c, f in zip(coeffs[1:], self._ind_functions[1:]):
            fun += c * f

        return fun
