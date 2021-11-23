# # Rigid motion
# In this example we show to use a Lagrange Multiplier Formuations to enforce a solution that is free of rigid motions as explained in [1]
#
# > [1] Kuchta, M., Mardal, K. A., & Mortensen, M. (2019). On the singular Neumann problem in linear elasticity. Numerical Linear Algebra with Applications, 26(1), e2212.
#
# Credits to [Aashild Telle](https://github.com/aashildte)
#
import dolfin
from fenics_plotly import plot

import pulse

# Make sure to use dolfin-adjoint version of object if using dolfin_adjoint
try:
    from dolfin_adjoint import (
        Constant,
        Function,
        Mesh,
        UnitCubeMesh,
        interpolate,
        Expression,
    )
except ImportError:
    from dolfin import Function, Constant, Mesh, UnitCubeMesh, interpolate, Expression

from pulse import DeformationGradient, Jacobian


def rigid_motion_term(mesh, u, r):
    position = dolfin.SpatialCoordinate(mesh)
    RM = [
        Constant((1, 0, 0)),
        Constant((0, 1, 0)),
        Constant((0, 0, 1)),
        dolfin.cross(position, Constant((1, 0, 0))),
        dolfin.cross(position, Constant((0, 1, 0))),
        dolfin.cross(position, Constant((0, 0, 1))),
    ]

    return sum(dolfin.dot(u, zi) * r[i] * dolfin.dx for i, zi in enumerate(RM))


class RigidMotionProblem(pulse.MechanicsProblem):
    def _init_spaces(self):

        mesh = self.geometry.mesh

        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P3 = dolfin.VectorElement("Real", mesh.ufl_cell(), 0, 6)

        self.state_space = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P1, P2, P3]))

        self.state = Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

    def _handle_bcs(self, bcs, bcs_parameters):
        self.bcs = pulse.BoundaryConditions()

    def _init_forms(self):

        p, u, r = dolfin.split(self.state)
        q, v, w = dolfin.split(self.state_test)

        F = dolfin.variable(DeformationGradient(u))
        J = Jacobian(F)

        dx = self.geometry.dx

        # Add penalty term
        internal_energy = (
            self.material.strain_energy(
                F,
            )
            + self.material.compressibility(p, J)
        )

        self._virtual_work = dolfin.derivative(
            internal_energy * dx,
            self.state,
            self.state_test,
        )

        self._virtual_work += dolfin.derivative(
            rigid_motion_term(
                mesh=self.geometry.mesh,
                u=u,
                r=r,
            ),
            self.state,
            self.state_test,
        )

        self._jacobian = dolfin.derivative(
            self._virtual_work,
            self.state,
            dolfin.TrialFunction(self.state_space),
        )

        self._init_solver()

    def _init_solver(self):
        self._problem = pulse.NonlinearProblem(
            J=self._jacobian,
            F=self._virtual_work,
            bcs=[],
        )
        self.solver = pulse.NonlinearSolver(
            self._problem,
            self.state,
            parameters=self.solver_parameters,
        )


N = 4
mesh = UnitCubeMesh(N, N, N)

V_f = dolfin.VectorFunctionSpace(mesh, "CG", 1)
# Fibers
f0 = interpolate(Expression(("1.0", "0.0", "0.0"), degree=1), V_f)
# Sheets
s0 = interpolate(Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
# Fiber-sheet normal
n0 = interpolate(Expression(("0.0", "0.0", "1.0"), degree=1), V_f)


microstructure = pulse.Microstructure(f0=f0, s0=s0, n0=n0)

# Create the geometry
geometry = pulse.Geometry(
    mesh=mesh,
    microstructure=microstructure,
)
# -

activation = Function(dolfin.FunctionSpace(geometry.mesh, "R", 0))
activation.assign(Constant(0.2))
matparams = pulse.HolzapfelOgden.default_parameters()
material = pulse.HolzapfelOgden(
    activation=activation,
    parameters=matparams,
    f0=geometry.f0,
    s0=geometry.s0,
    n0=geometry.n0,
)

problem = RigidMotionProblem(geometry, material)

problem.solve()

u = problem.state.split(deepcopy=True)[1]
V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
u_int = interpolate(u, V)
new_mesh = Mesh(mesh)
dolfin.ALE.move(new_mesh, u_int)

fig = plot(geometry.mesh, opacity=0.4, show=False)
fig.add_plot(plot(new_mesh, color="red", show=False))
fig.show()
