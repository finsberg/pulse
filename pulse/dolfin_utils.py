import dolfin
import numpy as np
import ufl

from . import assemble
from . import Constant
from . import Function
from . import FunctionAssigner
from . import numpy_mpi
from . import project
from .utils import DOLFIN_VERSION_MAJOR
from .utils import getLogger

logger = getLogger(__name__)


def map_vector_field(f0, new_mesh, u=None, name="fiber", normalize=True):
    """
    Map a vector field (f0) onto a new mesh (new_mesh) where the new mesh
    can be a moved version of the original one according to some
    displacement (u). In that case we will just a Piola transform to
    map the vector field.
    """
    representation = dolfin.parameters["form_compiler"]["representation"]
    if DOLFIN_VERSION_MAJOR > 2016:
        dolfin.parameters["form_compiler"]["representation"] = "quadrature"

    dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

    ufl_elem = f0.function_space().ufl_element()
    f0_new = Function(dolfin.FunctionSpace(new_mesh, ufl_elem))

    if u is not None:

        f0_mesh = f0.function_space().mesh()
        u_elm = u.function_space().ufl_element()
        V = dolfin.FunctionSpace(f0_mesh, u_elm)
        u0 = Function(V)
        # arr = numpy_mpi.gather_vector(u.vector())
        # numpy_mpi.assign_to_vector(u0.vector(), arr)
        u0.vector()[:] = u.vector()
        from .kinematics import DeformationGradient

        F = DeformationGradient(u0)

        f0_updated = project(F * f0, f0.function_space())

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
        dolfin.parameters["form_compiler"]["representation"] = representation

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
    """Given a vector field, return a vector field with an L2 norm equal to 1.0"""
    dim = len(u)
    S = u.function_space().sub(0).collapse()

    components = vectorfield_to_components(u, S, dim)

    normarray = np.sqrt(sum(components[i].vector().norm("l2") ** 2 for i in range(dim)))

    for comp in components:
        comp.vector()[:] = comp.vector() / normarray

    assigners = [FunctionAssigner(u.function_space().sub(i), S) for i in range(dim)]
    for i, comp, assigner in zip(range(dim), components, assigners):
        assigner.assign(u.split()[i], comp)

    return u


def vectorfield_to_components(u, S, dim):
    components = [Function(S) for i in range(dim)]
    assigners = [FunctionAssigner(S, u.function_space().sub(i)) for i in range(dim)]
    for i, comp, assigner in zip(range(dim), components, assigners):
        assigner.assign(comp, u.split()[i])

    return components


def get_pressure(problem):
    """Returns p_lv (and p_rv if BiV mesh)"""

    plv = [p.traction for p in problem.bcs.neumann if p.name == "lv"]
    prv = [p.traction for p in problem.bcs.neumann if p.name == "rv"]

    assert len(plv) > 0, "Problem has no Neumann BC for LV endo"
    pressure = [plv[0]]
    if prv:
        pressure.append(prv[0])
        return tuple(pressure)
    else:
        return pressure[0]


def compute_meshvolume(domain=None, dx=dolfin.dx, subdomain_id=None):
    return Constant(
        assemble(
            Constant(1.0) * dx(domain=domain, subdomain_id=subdomain_id),
        ),
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
        from ufl.domain import find_geometric_dimension

        dim = find_geometric_dimension(u)

    except ufl.UFLException as ex:

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


def list_sum(lst):
    """
    Return the sum of a list, when the convetiional
    method (like `sum`) it not working.
    For example if you have a list of dolfin functions.

    :param list l: a list of objects
    :returns: The sum of the list. The type depends on
              the type of elemets in the list

    """

    if not isinstance(lst, list):
        return lst

    out = lst[0]
    for item in lst[1:]:
        out += item
    return out


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


class MixedParameter(Function):
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
        f_ = Function(self.basespace)
        f_.assign(f)
        self.function_assigner[i].assign(self.split()[i], f_)


class RegionalParameter(Function):
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

        Function.__init__(self, V)
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
        f = Function(self._proj_space)
        f.vector()[dofs] = 1.0
        return f

    def _sum(self):
        coeffs = dolfin.split(self)
        fun = coeffs[0] * self._ind_functions[0]

        for c, f in zip(coeffs[1:], self._ind_functions[1:]):
            fun += c * f

        return fun
