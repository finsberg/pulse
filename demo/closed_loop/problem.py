import dolfin
from pulse import (MechanicsProblem, DeformationGradient, Jacobian)
from pulse.dolfin_utils import get_cavity_volume_form


class Problem(MechanicsProblem):
    """
    This class implements a three field variational form with
    u,p,pinner as the three field variables

    """
    @property
    def volume(self):
        return self._V0

    @volume.setter
    def volume(self, V):
        self._V0.assign(V)
    
    def _init_spaces(self):

        mesh = self.geometry.mesh

        V = dolfin.VectorElement("P", mesh.ufl_cell(), 2)
        Q = dolfin.FiniteElement("P", mesh.ufl_cell(), 1)
        R = dolfin.FiniteElement("Real", mesh.ufl_cell(), 0)

        el = dolfin.MixedElement([V, Q, R])
        self.state_space = dolfin.FunctionSpace(mesh, el)
        self.state = dolfin.Function(self.state_space)
        self.state_test = dolfin.TestFunction(self.state_space)

        self._Vu = get_cavity_volume_form(self.geometry.mesh,
                                          u=dolfin.split(self.state)[0],
                                          xshift=self.geometry.xshift)
        self._V0 = dolfin.Constant(self.geometry.cavity_volume())
        # self._constrain_volume = False
        # self._switch = dolfin.Constant(0.0)

    # @property
    # def constrain_volume(self):
        # return self._constrain_volume

    # @constrain_volume.setter
    # def constrain_volume(self, constrain_volume):

        # switch = 1.0 if constrain_volume else 0.0
        # self._switch.assign(switch)
        # self._constrain_volume = constrain_volume
        
    def _init_forms(self):

        u, p, pinn = dolfin.split(self.state)
        v, q, qinn = dolfin.split(self.state_test)

        F = dolfin.variable(DeformationGradient(u))
        J = Jacobian(F)

        ds = self.geometry.ds
        dsendo = ds(self.geometry.markers['ENDO'])
        dx = self.geometry.dx

        endoarea = dolfin.assemble(dolfin.Constant(1.0) * dsendo)

        internal_energy = self.material.strain_energy(F) * dx \
            + self.material.compressibility(p, J) * dx \
            + (pinn * (self._V0/endoarea - self._Vu)) * dsendo

        self._virtual_work \
            = dolfin.derivative(internal_energy,
                                self.state, self.state_test)

        self._virtual_work += self.external_work(u, v)

        self._jacobian \
            = dolfin.derivative(self._virtual_work, self.state,
                                dolfin.TrialFunction(self.state_space))

        self.set_dirichlet_bc()


