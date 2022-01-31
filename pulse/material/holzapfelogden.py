import dolfin

from .. import kinematics
from ..dolfin_utils import get_dimesion
from ..dolfin_utils import heaviside
from ..dolfin_utils import subplus
from .active_model import ActiveModels
from .material_model import Material


class HolzapfelOgden(Material):
    r"""
    Orthotropic model by Holzapfel and Ogden

    .. math::

       \mathcal{W}(I_1, I_{4f_0})
       = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
       + \frac{a_f}{2 b_f} \left( e^{ b_f (I_{4f_0} - 1)_+^2} -1 \right)
       + \frac{a_s}{2 b_s} \left( e^{ b_s (I_{4s_0} - 1)_+^2} -1 \right)
       + \frac{a_fs}{2 b_fs} \left( e^{ b_fs I_{8fs}^2} -1 \right)
    where

    .. math::

       (\cdot)_+ = \max\{x,0\}


    .. rubric:: Reference

    [1] Holzapfel, Gerhard A., and Ray W. Ogden.
    "Constitutive modelling of passive myocardium:
    a structurally based framework for material characterization.
    "Philosophical Transactions of the Royal Society of London A:
    Mathematical, Physical and Engineering Sciences 367.1902 (2009): 3445-3475.

    """
    name = "holzapfel_ogden"

    @staticmethod
    def default_parameters():
        """
        Default matereial parameter for the Holzapfel Ogden model

        Taken from Table 1 row 3 of [1]
        """

        return {
            "a": 0.059,
            "b": 0.023,
            "a_f": 18.472,
            "b_f": 16.026,
            "a_s": 2.481,
            "b_s": 11.120,
            "a_fs": 0.216,
            "b_fs": 11.436,
        }

    def W_1(self, I1, diff=0, *args, **kwargs):
        r"""
        Isotropic contribution.

        If `diff = 0`, return

        .. math::

           \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)

        If `diff = 1`, return

        .. math::

           \frac{a}{b} e^{ b (I_1 - 3)}

        If `diff = 2`, return

        .. math::

           \frac{a b}{2}  e^{ b (I_1 - 3)}

        """

        a = self.a
        b = self.b

        if diff == 0:
            try:
                if float(a) > dolfin.DOLFIN_EPS:
                    if float(b) > dolfin.DOLFIN_EPS:
                        return a / (2.0 * b) * (dolfin.exp(b * (I1 - 3)) - 1.0)
                    else:
                        return a / 2.0 * (I1 - 3)
                else:
                    return 0.0
            except Exception:
                return a / (2.0 * b) * (dolfin.exp(b * (I1 - 3)) - 1)
        elif diff == 1:
            return a / 2.0 * dolfin.exp(b * (I1 - 3))
        elif diff == 2:
            return a * b / 2.0 * dolfin.exp(b * (I1 - 3))

    def W_4(self, I4, direction, diff=0, use_heaviside=False, *args, **kwargs):
        r"""
        Anisotropic contribution.

        If `diff = 0`, return

        .. math::

           \frac{a_f}{2 b_f} \left( e^{ b_f (I_{4f_0} - 1)_+^2} -1 \right)

        If `diff = 1`, return

        .. math::

           a_f (I_{4f_0} - 1)_+ e^{ b_f (I_{4f_0} - 1)^2}

        If `diff = 2`, return

        .. math::

           a_f h(I_{4f_0} - 1) (1 + 2b(I_{4f_0} - 1))
           e^{ b_f (I_{4f_0} - 1)_+^2}

        where

        .. math::

           h(x) = \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}

        is the Heaviside function.

        """
        assert direction in ["f", "s", "n"]
        a = getattr(self, f"a_{direction}")
        b = getattr(self, f"b_{direction}")

        if I4 == 0:
            return 0

        if diff == 0:
            try:
                if float(a) > dolfin.DOLFIN_EPS:
                    if float(b) > dolfin.DOLFIN_EPS:
                        return (
                            a / (2.0 * b) * (dolfin.exp(b * subplus(I4 - 1) ** 2) - 1.0)
                        )
                    else:
                        return a / 2.0 * subplus(I4 - 1) ** 2
                else:
                    return 0.0
            except Exception:
                # Probably failed to convert a and b to float
                return a / (2.0 * b) * (dolfin.exp(b * subplus(I4 - 1) ** 2) - 1.0)

        elif diff == 1:
            return a * subplus(I4 - 1) * dolfin.exp(b * pow(I4 - 1, 2))
        elif diff == 2:
            return (
                a
                * heaviside(I4 - 1)
                * (1 + 2.0 * b * pow(I4 - 1, 2))
                * dolfin.exp(b * pow(I4 - 1, 2))
            )

    def W_8(self, I8, *args, **kwargs):
        """
        Cross fiber-sheet contribution.
        """
        a = self.a_fs
        b = self.b_fs

        try:
            if float(a) > dolfin.DOLFIN_EPS:
                if float(b) > dolfin.DOLFIN_EPS:
                    return a / (2.0 * b) * (dolfin.exp(b * I8**2) - 1.0)
                else:
                    return a / 2.0 * I8**2
            else:
                return 0.0
        except Exception:
            return a / (2.0 * b) * (dolfin.exp(b * I8**2) - 1.0)

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
        I1 = kinematics.I1(F, isochoric=self.isochoric)
        I4f = kinematics.I4(F, self.f0, isochoric=self.isochoric)
        I4s = kinematics.I4(F, self.s0, isochoric=self.isochoric)
        I8fs = kinematics.I8(F, self.f0, self.s0)

        if self.active_model == ActiveModels.active_strain:
            mgamma = 1 - self.activation_field
            I1e = mgamma * I1 + (1 / mgamma**2 - mgamma) * I4f
            I4fe = 1 / mgamma**2 * I4f
            I4se = mgamma * I4s
            I8fse = 1 / dolfin.sqrt(mgamma) * I8fs
        else:
            I1e = I1
            I4fe = I4f
            I4se = I4s
            I8fse = I8fs

        # Active stress
        Wactive = self.Wactive(F, diff=0)

        dim = get_dimesion(F)
        W1 = self.W_1(I1e, diff=0, dim=dim)
        W4f = self.W_4(I4fe, "f", diff=0)
        W4s = self.W_4(I4se, "s", diff=0)
        W8fs = self.W_8(I8fse, diff=0)

        W = W1 + W4f + W4s + W8fs + Wactive

        return W
