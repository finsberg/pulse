import dolfin

try:
    from dolfin_adjoint import Constant
except ImportError:
    from dolfin import Constant

from .. import kinematics
from ..dolfin_utils import get_dimesion
from .material_model import Material


class Guccione(Material):
    """
    Guccione material model.
    """

    name = "guccione"

    @staticmethod
    def default_parameters():

        p = {"C": 2.0, "bf": 8.0, "bt": 2.0, "bfs": 4.0}

        return p

    def SecondPiolaStress(self, F, p, *args, **kwargs):

        P = self.FirstPiolaStress(F, p)
        S = dolfin.inv(F) * P
        return S

    def is_isotropic(self):
        """
        Return True if the material is isotropic.
        """

        p = self.parameters
        return p["bt"] == 1.0 and p["bf"] == 1.0 and p["bfs"] == 1.0

    def strain_energy(self, F_):
        """
        UFL form of the strain energy.
        """

        params = self.parameters

        # Elastic part of deformation gradient
        F = self.active.Fe(F_)

        E = kinematics.GreenLagrangeStrain(F, isochoric=self.is_isochoric)

        CC = Constant(params["C"], name="C")

        e1 = self.active.f0
        e2 = self.active.s0
        e3 = self.active.n0

        if any(e is None for e in (e1, e2, e3)):
            msg = (
                "Need to provide the full orthotropic basis "
                "for the Guccione model got \ne1 = {e1}\n"
                "e2 = {e2}\ne3 = {e3}"
            ).format(e1=e1, e2=e2, e3=e3)
            raise ValueError(msg)

        if self.is_isotropic():
            # isotropic case
            Q = dolfin.inner(E, E)

        else:
            # fully anisotropic
            bt = Constant(params["bt"], name="bt")
            bf = Constant(params["bf"], name="bf")
            bfs = Constant(params["bfs"], name="bfs")

            E11, E12, E13 = (
                dolfin.inner(E * e1, e1),
                dolfin.inner(E * e1, e2),
                dolfin.inner(E * e1, e3),
            )
            E21, E22, E23 = (
                dolfin.inner(E * e2, e1),
                dolfin.inner(E * e2, e2),
                dolfin.inner(E * e2, e3),
            )
            E31, E32, E33 = (
                dolfin.inner(E * e3, e1),
                dolfin.inner(E * e3, e2),
                dolfin.inner(E * e3, e3),
            )

            Q = (
                bf * E11 ** 2
                + bt * (E22 ** 2 + E33 ** 2 + 2 * E23 ** 2)
                + bfs * (2 * E12 ** 2 + 2 * E13 ** 2)
            )

        # passive strain energy
        Wpassive = CC / 2.0 * (dolfin.exp(Q) - 1)
        Wactive = self.active.Wactive(F, diff=0)

        return Wpassive + Wactive
