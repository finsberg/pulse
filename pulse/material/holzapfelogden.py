import dolfin
from ..dolfin_utils import subplus, heaviside
from .material_model import Material


class HolzapfelOgden(Material):
    r"""
    Transversally isotropic version of the
    Holzapfel and Ogden material model

    .. math::

       \mathcal{W}(I_1, I_{4f_0})
       = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
       + \frac{a_f}{2 b_f} \left( e^{ b_f (I_{4f_0} - 1)_+^2} -1 \right)

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

        return {"a": 2.28, "a_f": 1.685, "b": 9.726, "b_f": 15.779}

    # def CauchyStress(self, F, p=None, deviatoric=False):

    #     I1  = self.active.I1(F)
    #     I4f = self.active.I4(F)

    #     # Active stress
    #     wactive = self.active.Wactive(F, diff = 1)

    #     dim = get_dimesion(F)
    #     I = Identity(dim)
    #     w1   = self.W_1(I1, diff = 1, dim = dim)
    #     w4f  = self.W_4(I4f, diff = 1)

    #     Fe = self.active.Fe(F)
    #     Be = Fe*Fe.T
    #     B = F*F.T

    #     fe = Fe*self.active.get_component("fiber")
    #     f = F*self.active.get_component("fiber")

    #     fefe = outer(fe, fe)
    #     ff = outer(f,f)

    #     # T = 2*w1*Be + 2*w4f*fefe + wactive*ff
    #     # T = 2*w4f*ff + wactive*ff  #2*w1*B
    #     T = wactive*ff
    #     if deviatoric:
    #         return T - tr(T)*I

    #     if p is None:
    #         return T

    #     else:
    #         return T - p*I

    def W_1(self, I_1, diff=0, *args, **kwargs):
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
            return a / (2.0 * b) * (dolfin.exp(b * (I_1 - 3)) - 1)
        elif diff == 1:
            return a / 2.0 * dolfin.exp(b * (I_1 - 3))
        elif diff == 2:
            return a * b / 2.0 * dolfin.exp(b * (I_1 - 3))

    def W_4(self, I_4, diff=0, *args, **kwargs):
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
        a = self.a_f
        b = self.b_f

        if I_4 == 0:
            return 0

        if diff == 0:
            return (
                a
                / (2.0 * b)
                * heaviside(I_4 - 1)
                * (dolfin.exp(b * pow(I_4 - 1, 2)) - 1)
            )

        elif diff == 1:
            return a * subplus(I_4 - 1) * dolfin.exp(b * pow(I_4 - 1, 2))
        elif diff == 2:
            return (
                a
                * heaviside(I_4 - 1)
                * (1 + 2.0 * b * pow(I_4 - 1, 2))
                * dolfin.exp(b * pow(I_4 - 1, 2))
            )
