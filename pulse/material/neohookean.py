from .. import kinematics
from ..dolfin_utils import get_dimesion
from .material_model import Material


class NeoHookean(Material):
    """
    Class for Neo Hookean material
    """

    name = "neo_hookean"

    @staticmethod
    def default_parameters():
        return {"mu": 15.0}

    def W_1(self, I_1, diff=0, dim=3, *args, **kwargs):

        mu = self.mu

        if diff == 0:
            return 0.5 * mu * (I_1 - dim)
        elif diff == 1:
            return 0.5 * mu
        elif diff == 2:
            return 0

    def strain_energy(self, F):
        r"""
        Strain-energy density function.

        .. math::

           \mathcal{W} = \mathcal{W}_1 + \mathcal{W}_{4f}
           + \mathcal{W}_{\mathrm{active}}

        where

        .. math::

           \mathcal{W}_{\mathrm{active}} =
           \begin{cases}
             0 & \text{if acitve strain} \\
             \gamma I_{4f} & \text{if active stress}
           \end{cases}


        :param F: Deformation gradient
        :type F: :py:class:`dolfin.Function`

        """

        # Invariants
        I1 = kinematics.I1(self.Fe(F), isochoric=self.isochoric)

        # Active stress
        Wactive = self.Wactive(F, diff=0)

        dim = get_dimesion(F)
        W1 = self.W_1(I1, diff=0, dim=dim)

        W = W1 + Wactive

        return W
