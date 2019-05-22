import dolfin
from .. import kinematics
from .active_model import ActiveModel, check_component


def Wactive_transversally(Ta, C, f0, eta=0.0):
    """
    Return active strain energy when activation is only
    working along the fibers, with a possible transverse
    component defined by eta

    Arguments
    ---------
    Ta : dolfin.Function or dolfin.Constant
        A scalar function representng the mangnitude of the
        active stress in the reference configuration (firt Pioala)
    C : ufl.Form
        The right Cauchy-Green deformation tensor
    f0 : dolfin.Function
        A vector function representng the direction of the
        active stress
    eta : float
        Amount of active stress in the transverse direction
        (relative to f0)
    """

    I4f = dolfin.inner(C * f0, f0)
    I1 = dolfin.tr(C)
    return dolfin.Constant(0.5) * Ta * ((I4f - 1) + eta * ((I1 - 3) - (I4f - 1)))


def Wactive_orthotropic(Ta, C, f0, s0, n0):
    """Return active strain energy for an orthotropic
    active stress

    
    Arguments
    ---------
    Ta : dolfin.Function or dolfin.Constant
        A vector function representng the mangnitude of the
        active stress in the reference configuration (firt Pioala).
        Ta = (Ta_f0, Ta_s0, Ta_n0)
    C : ufl.Form
        The right Cauchy-Green deformation tensor
    f0 : dolfin.Function
        A vector function representng the direction of the
        first component
    s0 : dolfin.Function
        A vector function representng the direction of the
        second component
    n0 : dolfin.Function
        A vector function representng the direction of the
        third component
    """
    I4f = dolfin.inner(C * f0, f0)
    I4s = dolfin.inner(C * s0, s0)
    I4n = dolfin.inner(C * n0, n0)

    I4 = dolfin.as_vector([I4f - 1, I4s - 1, I4n - 1])
    return dolfin.Constant(0.5) * dolfin.inner(Ta, I4)


def Wactive_anisotropic(Ta, C, f0, s0, n0):
    """Return active strain energy for a fully anisotropic
    acitvation.
    Note that the three basis vectors are assumed to be
    orthogonal
    
    Arguments
    ---------
    Ta : dolfin.Function or dolfin.Constant
        A full tensor function representng the active stress tensor
        of the active stress in the reference configuration
        (firt Pioala).
    C : ufl.Form
        The right Cauchy-Green deformation tensor
    f0 : dolfin.Function
        A vector function representng the direction of the
        first component
    s0 : dolfin.Function
        A vector function representng the direction of the
        second component
    n0 : dolfin.Function
        A vector function representng the direction of the
        third component
    """
    I4f = dolfin.inner(C * f0, f0)
    I4s = dolfin.inner(C * s0, s0)
    I4n = dolfin.inner(C * n0, n0)

    I8fs = dolfin.inner(C * f0, s0)
    I8fn = dolfin.inner(C * f0, n0)
    I8sn = dolfin.inner(C * s0, n0)

    A = dolfin.as_matrix(
        (
            (0.5 * (I4f - 1), I8fs, I8fn),
            (I8fs, 0.5 * (I4s - 1), I8sn),
            (I8fn, I8sn, 0.5 * (I4n - 1)),
        )
    )

    return dolfin.inner(Ta, A)


class ActiveStress(ActiveModel):
    """
    Active stress model
    """

    _model = "active_stress"

    def __init__(self, *args, **kwargs):

        # Fraction of transverse stress
        # (0 = active only along fiber, 1 = equal
        # amout of tension in all directions)
        self._eta = dolfin.Constant(kwargs.pop("eta", 0.0))

        self.active_isotropy = kwargs.pop("active_isotropy", "transversally")

        ActiveModel.__init__(self, *args, **kwargs)

    @property
    def eta(self):
        return self._eta

    def Wactive(self, F, diff=0):
        """Active stress energy
        """

        C = F.T * F

        if diff == 0:

            if self.active_isotropy == "transversally":
                return Wactive_transversally(
                    Ta=self.activation_field, C=C, f0=self.f0, eta=self.eta
                )

            elif self.active_isotropy == "orthotropic":
                return Wactive_orthotropic(
                    Ta=self.activation_field, C=C, f0=self.f0, s0=self.s0, n0=self.n0
                )

            elif self.active_isotropy == "fully_anisotropic":

                return Wactive_anisotropic(
                    Ta=self.activation_field, C=C, f0=self.f0, s0=self.s0, n0=self.n0
                )
            else:
                msg = ("Unknown acitve isotropy " "{}").format(self.active_isotropy)
                raise ValueError(msg)

        elif diff == 1:
            return self.activation_field

    @property
    def type(self):
        return "ActiveStress"

    def I1(self, F, *args):
        return self._I1(F)

    def I4(self, F, component="f0", *args):

        check_component(component)
        a0 = getattr(self, component)
        return self._I4(F, a0)

    @property
    def Fa(self):
        return kinematics.SecondOrderIdentity(self.f0)

    def Fe(self, F):
        return F
