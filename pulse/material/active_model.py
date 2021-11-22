from enum import Enum
from typing import Optional

import dolfin

try:
    from dolfin_adjoint import Constant
except ImportError:
    from dolfin import Constant

from ..dolfin_utils import get_dimesion, RegionalParameter


class ActiveModels(str, Enum):
    active_stress = "active_stress"
    active_strain = "active_strain"


class ActiveStressModels(str, Enum):
    transversally = "transversally"
    orthotropic = "orthotropic"
    fully_anisotropic = "fully_anisotropic"


class ActiveModel:
    def __init__(
        self,
        activation=None,
        f0=None,
        s0=None,
        n0=None,
        model: ActiveModels = ActiveModels.active_strain,
        active_isotropy: ActiveStressModels = ActiveStressModels.transversally,
        eta: Optional[float] = None,
        T_ref: Optional[float] = None,
        isochoric=True,
    ):
        # Fiber system
        self.f0 = f0
        self.s0 = s0
        self.n0 = n0

        self._activation = (
            Constant(0, name="activation") if activation is None else activation
        )

        self.T_ref = (
            Constant(T_ref, name="T_ref") if T_ref else Constant(1.0, name="T_ref")
        )

        self.eta = Constant(eta, name="eta") if eta else Constant(0.0, name="eta")
        self.isochoric = isochoric
        self.active_isotropy = active_isotropy
        self.model = model

    @property
    def dim(self):
        if not hasattr(self, "_dim"):
            try:
                self._dim = get_dimesion(self.f0)
            except Exception:
                # just assume three dimensions
                self._dim = 3
        return self._dim

    @property
    def activation(self):
        """
        Return the activation paramter.
        If regional, this will return one parameter
        for each segment.
        """
        return self._activation

    @property
    def activation_field(self):
        """
        Return the activation field.
        If regional, this will return a piecewise
        constant function (DG_0)
        """

        # Activation
        if isinstance(self._activation, RegionalParameter):
            # This means a regional activation
            # Could probably make this a bit more clean
            activation = self._activation.function
        else:
            activation = self._activation

        return self.T_ref * activation

    @property
    def Fa(self):

        if self.model == ActiveModels.active_stress:
            return dolfin.Identity(self.dim)

        f0 = self.f0
        f0f0 = dolfin.outer(f0, f0)
        Id = dolfin.Identity(self.dim)

        mgamma = 1 - self.activation_field
        Fa = mgamma * f0f0 + pow(mgamma, -1.0 / float(self.dim - 1)) * (Id - f0f0)

        return Fa

    def Fe(self, F):
        if self.model == ActiveModels.active_stress:
            return F
        Fa = self.Fa
        Fe = F * dolfin.inv(Fa)

        return Fe

    def Wactive(self, F=None, diff=0):
        """Active stress energy"""
        if self.model == ActiveModels.active_strain:
            return 0

        C = F.T * F

        if diff == 0:

            return Wactive(
                Ta=self.activation_field,
                C=C,
                f0=self.f0,
                s0=self.s0,
                n0=self.n0,
                eta=self.eta,
                active_isotropy=self.active_isotropy,
            )

        elif diff == 1:
            return self.activation_field
        raise ValueError(f"Unknown diff {diff}")


def Wactive(
    Ta,
    C,
    f0,
    eta,
    active_isotropy=ActiveStressModels.transversally,
    s0=None,
    n0=None,
):
    if active_isotropy == "transversally":
        return Wactive_transversally(
            Ta=Ta,
            C=C,
            f0=f0,
            eta=eta,
        )

    elif active_isotropy == "orthotropic":
        return Wactive_orthotropic(
            Ta=Ta,
            C=C,
            f0=f0,
            s0=s0,
            n0=n0,
        )

    elif active_isotropy == "fully_anisotropic":

        return Wactive_anisotropic(
            Ta=Ta,
            C=C,
            f0=f0,
            s0=s0,
            n0=n0,
        )
    else:
        msg = ("Unknown acitve isotropy " "{}").format(active_isotropy)
        raise ValueError(msg)


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
    return Constant(0.5) * Ta * ((I4f - 1) + eta * ((I1 - 3) - (I4f - 1)))


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
    return Constant(0.5) * dolfin.inner(Ta, I4)


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
        ),
    )

    return dolfin.inner(Ta, A)
