
from collections import namedtuple
import dolfin
try:
    from dolfin_adjoint import (Function, NonlinearVariationalSolver,
                                NonlinearVariationalProblem,
                                FunctionAssigner)
except ImportError:
    from dolfin import (Function, NonlinearVariationalSolver,
                        NonlinearVariationalProblem,
                        FunctionAssigner)

from . import kinematics
from .utils import set_default_none, make_logger

logger = make_logger(__name__, 10)

BoundaryConditions = namedtuple('BoundaryConditions',
                                ['dirichlet', 'neumann',
                                 'robin', 'body_force'])
set_default_none(BoundaryConditions, ())

NeumannBC = namedtuple('NeumannBC', ['traction', 'marker', 'name'])
# Name is optional
NeumannBC.__new__.__defaults__ = ('',)
RobinBC = namedtuple('RobinBC', ['value', 'marker'])


def dirichlet_fix_base(W, ffun, marker):
    '''Fix the basal plane.
    '''
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = dolfin.DirichletBC(V, dolfin.Constant((0, 0, 0)),
                            ffun, marker)
    return bc


def dirichlet_fix_base_directional(W, ffun, marker, direction=0):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = dolfin.DirichletBC(V.sub(direction),
                            dolfin.Constant(0.0),
                            ffun, marker)
    return bc


class SolverDidNotConverge(Exception):
    pass


class MechanicsProblem(object):
    """
    Base class for mechanics problem
    """
    def __init__(self, geometry, material, bcs):

        logger.debug('Initialize mechanics problem')
        self.geometry = geometry
        self.material = material
        self.bcs = bcs

        # Make sure that the material has microstructure information
        for attr in ("f0", "s0", "n0"):
            setattr(self.material, attr, getattr(self.geometry, attr))

        self._init_spaces()
        self._init_forms()

    def _init_spaces(self):

        logger.debug('Initialize spaces for mechanics problem')
        mesh = self.geometry.mesh

        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

        # P2_space = FunctionSpace(mesh, P2)
        # P1_space = FunctionSpace(mesh, P1)
        self.state_space = dolfin.FunctionSpace(mesh, P2*P1)

        self.state = Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

    def _init_forms(self):

        logger.debug('Initialize forms mechanics problem')
        # Displacement and hydrostatic_pressure
        u, p = dolfin.split(self.state)
        v, q = dolfin.split(self.state_test)

        # Some mechanical quantities
        F = dolfin.variable(kinematics.DeformationGradient(u))
        J = kinematics.Jacobian(F)

        # Some geometrical quantities
        N = self.geometry.facet_normal
        ds = self.geometry.ds

        internal_energy = self.material.strain_energy(F) \
            + self.material.compressibility(p, J)

        self._virtual_work \
            = dolfin.derivative(internal_energy * dolfin.dx,
                                self.state, self.state_test)

        # DirichletBC
        for dirichlet_bc in self.bcs.dirichlet:

            msg = ('DirichletBC only implemented for as '
                   'callable. Please provide DirichletBC '
                   'as a callable with argument being the '
                   'state space only')

            if hasattr(dirichlet_bc, '__call__'):
                try:
                    self._dirichlet_bc \
                        = dirichlet_bc(self.state_space)
                except Exception as ex:
                    logger.error(msg)
                    raise ex
            else:

                raise NotImplementedError(msg)

        for neumann in self.bcs.neumann:

            n = neumann.traction * dolfin.cofac(F) * N
            self._virtual_work += dolfin.inner(v, n) * ds(neumann.marker)

        for robin in self.bcs.robin:

            self._virtual_work += dolfin.inner(robin.value * u, v) \
                                  * ds(robin.marker)

        for body_force in self.bcs.body_force:

            self._virtual_work \
                += -dolfin.derivative(dolfin.inner(body_force, u)
                                      * dolfin.dx, u, v)

        self._jacobian \
            = dolfin.derivative(self._virtual_work, self.state,
                                dolfin.TrialFunction(self.state_space))

    def reinit(self, state, annotate=False):
        """Reinitialze state
        """
        self.state.assign(state, annotate=annotate)
        self._init_forms()

    def solve(self):
        r"""
        Solve the variational problem

        .. math::

           \delta W = 0

        """
        logger.debug('Solving variational problem')
        # Get old state in case of non-convergence
        old_state = self.state.copy(deepcopy=True)
        problem \
            = NonlinearVariationalProblem(self._virtual_work,
                                          self.state,
                                          self._dirichlet_bc,
                                          self._jacobian)

        solver = NonlinearVariationalSolver(problem)

        try:
            logger.debug('Try to solve without annotation')
            nliter, nlconv = solver.solve()
            if not nlconv:
                logger.debug('Failed to solve without annotation')
                raise SolverDidNotConverge("Solver did not converge...")

        except RuntimeError as ex:
            logger.debug('Failed to solve without annotation')
            logger.debug('Reintialize old state and raise exception')

            self.reinit(old_state)

            raise SolverDidNotConverge(ex)

        return nliter, nlconv

    def get_displacement(self, annotate=True):

        D = self.state_space.sub(0)
        V = D.collapse()

        fa = FunctionAssigner(V, D)
        u = Function(V, name='displacement')
        fa.assign(u, self.state.split()[0],
                  annotate=annotate)
        return u
