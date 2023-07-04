import sys
import time
from pathlib import Path

import dolfin

from . import as_backend_type
from . import assemble
from .utils import enlist
from .utils import getLogger
from .utils import mpi_comm_world

logger = getLogger(__name__)


def dump_matrix_to_mtx(A: dolfin.PETScMatrix, filename: Path) -> None:
    with open(filename, "w") as f:
        mat = as_backend_type(A).mat()
        (num_rows, num_columns) = mat.size
        (ai, aj, av) = mat.getValuesCSR()
        num_nonzeros = len(av)
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"% Generated by {' '.join(sys.argv)}\n")
        f.write(f"{num_rows} {num_columns} {num_nonzeros}\n")
        for i in range(num_rows):
            for k in range(ai[i], ai[i + 1]):
                f.write(f"{i + 1} {aj[k] + 1} {av[k]}\n")


class NonlinearProblem(dolfin.NonlinearProblem):
    def __init__(
        self, J, F, bcs, output_matrix=False, output_matrix_path="output", **kwargs,
    ):
        super().__init__(**kwargs)
        self._J = J
        self._F = F

        self.bcs = enlist(bcs)
        self.output_matrix = output_matrix
        self.output_matrix_path = output_matrix_path
        self.verbose = True
        self.n = 0

    def F(self, b: dolfin.PETScVector, x: dolfin.PETScVector):
        assemble(self._F, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A: dolfin.PETScMatrix, x: dolfin.PETScVector):
        assemble(self._J, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

        if self.output_matrix:
            filename = Path("{self.output_matrix_path}/J_{self.n:04d}.mtx")
            filename.parent.mkdir(exist_ok=True, parents=True)
            dump_matrix_to_mtx(A, filename)
            if self.verbose:
                print(f"Assembled matrix written to {filename}")
            self.n = self.n + 1


class NonlinearSolver:
    def __init__(
        self,
        problem: NonlinearProblem,
        state,
        parameters=None,
    ):
        dolfin.PETScOptions.clear()
        self.update_parameters(parameters)
        self._problem = problem
        self._state = state

        self._solver = dolfin.PETScSNESSolver(mpi_comm_world())
        self._solver.set_from_options()

        self._solver.parameters.update(self.parameters)
        self._snes = self._solver.snes()
        self._snes.setConvergenceHistory()

        logger.debug(f"Linear Solver : {self._solver.parameters['linear_solver']}")
        logger.debug(f"Preconditioner:  {self._solver.parameters['preconditioner']}")
        logger.debug(f"atol: {self._solver.parameters['absolute_tolerance']}")
        logger.debug(f"rtol: {self._solver.parameters['relative_tolerance']}")
        logger.debug(f" Size          : {self._state.function_space().dim()}")
        dolfin.PETScOptions.clear()

    def update_parameters(self, parameters):
        ps = NonlinearSolver.default_solver_parameters()
        if hasattr(self, "parameters"):
            ps.update(self.parameters)
        if parameters is not None:
            ps.update(parameters)
        petsc = ps.pop("petsc")

        for k, v in petsc.items():
            if v is not None:
                dolfin.PETScOptions.set(k, v)
        self.verbose = ps.pop("verbose", False)
        if self.verbose:
            dolfin.PETScOptions.set("ksp_monitor")
            dolfin.PETScOptions.set("log_view")
            dolfin.PETScOptions.set("ksp_view")
            dolfin.PETScOptions.set("pc_view")
            dolfin.PETScOptions.set("mat_superlu_dist_statprint", True)
            ps["lu_solver"]["report"] = True
            ps["lu_solver"]["verbose"] = True
            ps["report"] = True
            ps["krylov_solver"]["monitor_convergence"] = True
        self.parameters = ps

    @staticmethod
    def default_solver_parameters():
        return {
            "petsc": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_33": 0,
            },
            "verbose": False,
            "linear_solver": "mumps",
            "preconditioner": "lu",
            "error_on_nonconvergence": False,
            "relative_tolerance": 1e-5,
            "absolute_tolerance": 1e-5,
            "maximum_iterations": 20,
            "report": False,
            "krylov_solver": {
                "absolute_tolerance": 1e-13,
                "relative_tolerance": 1e-13,
                "maximum_iterations": 1000,
                "monitor_convergence": False,
            },
            "lu_solver": {"report": False, "symmetric": False, "verbose": False},
        }

    def solve(self):
        """Solve the problem.

        Returns
        -------
        residual : _solver.snes (???)
            A measure of the accuracy (convergence and error)
            of the performed computation.
        """

        logger.debug(" Solving NonLinearProblem ...")

        start = time.time()
        self._solver.solve(self._problem, self._state.vector())
        end = time.time()

        logger.debug(f" ... Done in {end - start:.3f} s")

        residuals = self._snes.getConvergenceHistory()[0]
        num_iterations = self._snes.getLinearSolveIterations()
        logger.debug(f"Iterations    : {num_iterations}")
        if num_iterations > 0:
            logger.debug(f"Resiudal      : {residuals[-1]}")

        return num_iterations, self._snes.converged
