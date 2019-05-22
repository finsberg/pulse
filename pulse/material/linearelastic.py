import dolfin
from ..dolfin_utils import get_dimesion
from .material_model import Material


class LinearElastic(Material):
    """
    Class for linear elastic material
    """

    name = "linear_elastic"

    @staticmethod
    def default_parameters():
        return {"mu": 100.0, "lmbda": 1.0}

    def strain_energy(self, F_):

        F = self.active.Fe(F_)

        dim = get_dimesion(F)
        gradu = F - dolfin.Identity(dim)
        epsilon = 0.5 * (gradu + gradu.T)
        W = self.lmbda / 2 * (dolfin.tr(epsilon) ** 2) + self.mu * dolfin.tr(
            epsilon * epsilon
        )

        # Active stress
        Wactive = self.active.Wactive(F, diff=0)

        return W + Wactive
