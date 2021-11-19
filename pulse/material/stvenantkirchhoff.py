import dolfin

from .. import kinematics
from .material_model import Material


class StVenantKirchhoff(Material):
    """
    Class for linear elastic material
    """

    name = "saint_venant_kirchhoff"

    @staticmethod
    def default_parameters():
        return {"mu": 300.0, "lmbda": 1.0}

    def strain_energy(self, F_):

        F = self.Fe(F_)
        E = kinematics.GreenLagrangeStrain(F, isochoric=self.isochoric)
        W = self.lmbda / 2 * (dolfin.tr(E) ** 2) + self.mu * dolfin.tr(E * E)

        # Active stress
        Wactive = self.Wactive(F, diff=0)

        return W + Wactive
