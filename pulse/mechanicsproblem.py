import typing
from dataclasses import dataclass
from functools import partial

import dolfin
import ufl

from . import Constant
from . import DirichletBC
from . import Function
from . import FunctionAssigner
from . import has_dolfin_adjoint
from . import kinematics
from .dolfin_utils import list_sum
from .geometry import Geometry
from .geometry import HeartGeometry
from .material import Material
from .solver import NonlinearProblem
from .solver import NonlinearSolver
from .utils import get_lv_marker
from .utils import getLogger

logger = getLogger(__name__)


@dataclass
class NeumannBC:
    traction: typing.Union[float, ufl.Coefficient]
    marker: int
    name: str = ""


@dataclass
class RobinBC:
    value: typing.Union[float, ufl.Coefficient]
    marker: int


dirichlet_types = typing.Union[
    typing.Callable[[dolfin.FunctionSpace], dolfin.DirichletBC],
    dolfin.DirichletBC,
]


@dataclass
class BoundaryConditions:
    neumann: typing.Sequence[NeumannBC] = ()
    dirichlet: typing.Sequence[dirichlet_types] = ()
    robin: typing.Sequence[RobinBC] = ()
    body_force: typing.Sequence[ufl.Coefficient] = ()


def dirichlet_fix_base(W, ffun, marker):
    """Fix the basal plane."""
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = DirichletBC(V, Constant((0, 0, 0)), ffun, marker)
    return bc


def dirichlet_fix_base_directional(W, ffun, marker, direction=0):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = DirichletBC(V.sub(direction), Constant(0.0), ffun, marker)
    return bc


def cardiac_boundary_conditions(
    geometry, pericardium_spring=0.0, base_spring=0.0, base_bc="fix_x", **kwargs
):

    msg = (
        "Cardiac boundary conditions can only be applied to a "
        "HeartGeometry got {}".format(type(geometry))
    )
    assert isinstance(geometry, HeartGeometry), msg

    # Neumann BC
    lv_marker = get_lv_marker(geometry)
    lv_pressure = NeumannBC(
        traction=Constant(0.0, name="lv_pressure"),
        marker=lv_marker,
        name="lv",
    )
    neumann_bc = [lv_pressure]

    if geometry.is_biv:

        rv_pressure = NeumannBC(
            traction=Constant(0.0, name="rv_pressure"),
            marker=geometry.markers["ENDO_RV"][0],
            name="rv",
        )

        neumann_bc += [rv_pressure]

    # Robin BC
    if pericardium_spring > 0.0:

        robin_bc = [
            RobinBC(
                value=Constant(pericardium_spring),
                marker=geometry.markers["EPI"][0],
            ),
        ]

    else:
        robin_bc = []

    # Apply a linear sprint robin type BC to limit motion
    if base_spring > 0.0:
        robin_bc += [
            RobinBC(value=Constant(base_spring), marker=geometry.markers["BASE"][0]),
        ]

    # Dirichlet BC
    if base_bc == "fixed":

        dirichlet_bc = [
            partial(
                dirichlet_fix_base,
                ffun=geometry.ffun,
                marker=geometry.markers["BASE"][0],
            ),
        ]

    elif base_bc == "fix_x":

        dirichlet_bc = [
            partial(
                dirichlet_fix_base_directional,
                ffun=geometry.ffun,
                marker=geometry.markers["BASE"][0],
            ),
        ]
    else:
        raise ValueError(f"Unknown base bc {base_bc}")

    boundary_conditions = BoundaryConditions(
        dirichlet=dirichlet_bc,
        neumann=neumann_bc,
        robin=robin_bc,
    )

    return boundary_conditions


class SolverDidNotConverge(Exception):
    pass


class MechanicsProblem(object):
    """
    Base class for mechanics problem
    """

    def __init__(
        self,
        geometry: Geometry,
        material: Material,
        bcs=None,
        bcs_parameters=None,
        solver_parameters=None,
    ):

        logger.debug("Initialize mechanics problem")
        self.geometry = geometry
        self.material = material

        self._handle_bcs(bcs=bcs, bcs_parameters=bcs_parameters)

        # Make sure that the material has microstructure information
        for attr in ("f0", "s0", "n0"):
            setattr(self.material, attr, getattr(self.geometry, attr))

        self.solver_parameters = NonlinearSolver.default_solver_parameters()
        if solver_parameters is not None:
            self.solver_parameters.update(**solver_parameters)

        self._init_spaces()
        self._init_forms()

    def _handle_bcs(self, bcs, bcs_parameters):
        if bcs is None:
            if isinstance(self.geometry, HeartGeometry):
                self.bcs_parameters = MechanicsProblem.default_bcs_parameters()
                if bcs_parameters is not None:
                    self.bcs_parameters.update(**bcs_parameters)
            else:
                raise ValueError(
                    ("Please provive boundary conditions " "to MechanicsProblem"),
                )

            self.bcs = cardiac_boundary_conditions(self.geometry, **self.bcs_parameters)

        else:
            self.bcs = bcs

            # TODO: FIX THIS or require this
            # Just store this as well in case both is provided
            self.bcs_parameters = MechanicsProblem.default_bcs_parameters()
            if bcs_parameters is not None:
                self.bcs_parameters.update(**bcs_parameters)

    @staticmethod
    def default_bcs_parameters():
        return dict(pericardium_spring=0.0, base_spring=0.0, base_bc="fixed")

    def _init_spaces(self):

        logger.debug("Initialize spaces for mechanics problem")
        mesh = self.geometry.mesh

        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

        # P2_space = FunctionSpace(mesh, P2)
        # P1_space = FunctionSpace(mesh, P1)
        self.state_space = dolfin.FunctionSpace(mesh, P2 * P1)

        self.state = Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

    def _init_forms(self):

        logger.debug("Initialize forms mechanics problem")
        # Displacement and hydrostatic_pressure
        u, p = dolfin.split(self.state)
        v, q = dolfin.split(self.state_test)

        # Some mechanical quantities
        F = dolfin.variable(kinematics.DeformationGradient(u))
        J = kinematics.Jacobian(F)
        dx = self.geometry.dx

        internal_energy = self.material.strain_energy(
            F,
        ) + self.material.compressibility(p, J)

        self._virtual_work = dolfin.derivative(
            internal_energy * dx,
            self.state,
            self.state_test,
        )

        external_work = self._external_work(u, v)
        if external_work is not None:
            self._virtual_work += external_work

        self._set_dirichlet_bc()
        self._jacobian = dolfin.derivative(
            self._virtual_work,
            self.state,
            dolfin.TrialFunction(self.state_space),
        )
        self._init_solver()

    def _init_solver(self):
        if has_dolfin_adjoint:
            from dolfin_adjoint import (
                NonlinearVariationalProblem,
                NonlinearVariationalSolver,
            )

            self._problem = NonlinearVariationalProblem(
                J=self._jacobian,
                F=self._virtual_work,
                u=self.state,
                bcs=self._dirichlet_bc,
            )
            self.solver = NonlinearVariationalSolver(self._problem)
        else:
            self._problem = NonlinearProblem(
                J=self._jacobian,
                F=self._virtual_work,
                bcs=self._dirichlet_bc,
            )
            self.solver = NonlinearSolver(
                self._problem,
                self.state,
                parameters=self.solver_parameters,
            )

    def _set_dirichlet_bc(self):
        # DirichletBC
        for dirichlet_bc in self.bcs.dirichlet:

            msg = (
                "DirichletBC only implemented for as "
                "callable. Please provide DirichletBC "
                "as a callable with argument being the "
                "state space only"
            )

            if hasattr(dirichlet_bc, "__call__"):
                try:
                    self._dirichlet_bc = dirichlet_bc(self.state_space)
                except Exception as ex:
                    logger.error(msg)
                    raise ex
            else:

                raise NotImplementedError(msg)

    def _external_work(self, u, v):

        F = dolfin.variable(kinematics.DeformationGradient(u))

        N = self.geometry.facet_normal
        ds = self.geometry.ds
        dx = self.geometry.dx

        external_work = []

        for neumann in self.bcs.neumann:

            n = neumann.traction * ufl.cofac(F) * N
            external_work.append(dolfin.inner(v, n) * ds(neumann.marker))

        for robin in self.bcs.robin:

            external_work.append(dolfin.inner(robin.value * u, v) * ds(robin.marker))

        for body_force in self.bcs.body_force:

            external_work.append(
                -dolfin.derivative(dolfin.inner(body_force, u) * dx, u, v),
            )

        if len(external_work) > 0:
            return list_sum(external_work)

        return None

    def reinit(self, state, annotate=False):
        """Reinitialze state"""

        if has_dolfin_adjoint:
            try:
                self.state.assign(state, annotate=annotate)
            except Exception as ex:
                print(ex)
                self.state.assign(state)
        else:
            self.state.assign(state)

        self._init_forms()

    @staticmethod
    def default_solver_parameters():
        return NonlinearSolver.default_solver_parameters()

    def solve(self):
        r"""
        Solve the variational problem

        .. math::

           \delta W = 0

        """

        logger.debug("Solving variational problem")

        try:
            logger.debug("Try to solve")
            nliter, nlconv = self.solver.solve()

            if not nlconv:
                logger.debug("Failed")
                raise SolverDidNotConverge("Solver did not converge...")

        except RuntimeError as ex:
            logger.debug("Failed")
            logger.debug("Reintialize old state and raise exception")
            raise SolverDidNotConverge(ex) from ex
        else:
            logger.debug("Sucess")

        return nliter, nlconv

    def get_displacement(self, annotate=False):

        D = self.state_space.sub(0)
        V = D.collapse()

        fa = FunctionAssigner(V, D)
        u = Function(V, name="displacement")

        if has_dolfin_adjoint:
            fa.assign(u, self.state.split()[0], annotate=annotate)
        else:
            fa.assign(u, self.state.split()[0])

        return u

    def SecondPiolaStress(self):
        u, p = self.state.split(deepcopy=True)
        F = kinematics.DeformationGradient(u)
        return self.material.SecondPiolaStress(F, p)

    def FirstPiolaStress(self):
        u, p = self.state.split(deepcopy=True)
        F = kinematics.DeformationGradient(u)
        return self.material.FirstPiolaStress(F, p)

    def ChachyStress(self):
        u, p = self.state.split(deepcopy=True)
        F = kinematics.DeformationGradient(u)
        return self.material.CauchyStress(F, p)

    def GreenLagrangeStrain(self):
        u, p = self.state.split(deepcopy=True)
        F = kinematics.DeformationGradient(u)
        return kinematics.GreenLagrangeStrain(F)
