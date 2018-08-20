import dolfin
from pulse import (MechanicsProblem, DeformationGradient, Jacobian)


class CompressibleProblem(MechanicsProblem):
    """
    This class implements a compressbile model with a penalized
    compressibility term, solving for the displacement only.

    """
    def _init_spaces(self):

        mesh = self.geometry.mesh

        element = dolfin.VectorElement("P", mesh.ufl_cell(), 1)
        self.state_space = dolfin.FunctionSpace(mesh, element)
        self.state = dolfin.Function(self.state_space)
        self.state_test = dolfin.TestFunction(self.state_space)

        # Add penalty factor
        self.kappa = dolfin.Constant(1e3)
        
    def _init_forms(self):

        u = self.state
        v = self.state_test

        F = dolfin.variable(DeformationGradient(u))
        J = Jacobian(F)

        dx = self.geometry.dx

        # Add penalty term
        internal_energy = self.material.strain_energy(F) \
            + self.kappa * (J * dolfin.ln(J) - J + 1)

        self._virtual_work \
            = dolfin.derivative(internal_energy * dx,
                                self.state, self.state_test)

        self._virtual_work += self._external_work(u, v)

        self._jacobian \
            = dolfin.derivative(self._virtual_work, self.state,
                                dolfin.TrialFunction(self.state_space))

        self._set_dirichlet_bc()




