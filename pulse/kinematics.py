#!/usr/bin/env python
from dolfin import det, grad as Grad, inv, Identity, tr, inner

from .dolfin_utils import get_dimesion


# Strain tensors #####

# Second order identity tensor
def SecondOrderIdentity(F):
    """Return identity with same dimension as input
    """
    dim = get_dimesion(F)
    return Identity(dim)


def DeformationGradient(u, isochoric=False):
    """Return deformation gradient from displacement.
    """
    I = SecondOrderIdentity(u)
    F = I + Grad(u)
    if isochoric:
        return IsochoricDeformationGradient(F)
    else:
        return F


def IsochoricDeformationGradient(F):
    """Return the isochoric deformation gradient
    """
    J = Jacobian(F)
    dim = get_dimesion(F)
    return pow(J, -1.0 / float(dim)) * F


def Jacobian(F):
    """Determinant of the deformation gradient
    """
    return det(F)


def EngineeringStrain(F, isochoric=False):
    """Equivalent of engineering strain
    """
    I = SecondOrderIdentity(F)
    if isochoric:
        return IsochoricDeformationGradient(F) - I
    else:
        return F - I


def GreenLagrangeStrain(F, isochoric=False):
    """Green-Lagrange strain tensor
    """
    I = SecondOrderIdentity(F)
    C = RightCauchyGreen(F, isochoric)
    return 0.5 * (C - I)


def LeftCauchyGreen(F, isochoric=False):
    """Left Cauchy-Green tensor
    """
    if isochoric:
        F_ = IsochoricDeformationGradient(F)
    else:
        F_ = F

    return F_ * F_.T


def RightCauchyGreen(F, isochoric=False):
    """Left Cauchy-Green tensor
    """
    if isochoric:
        F_ = IsochoricDeformationGradient(F)
    else:
        F_ = F

    return F_.T * F_


def EulerAlmansiStrain(F, isochoric=False):
    """Euler-Almansi strain tensor
    """
    I = SecondOrderIdentity(F)
    b = LeftCauchyGreen(F, isochoric)
    return 0.5 * (I - inv(b))


# Invariants #####
class Invariants(object):
    def __init__(self, isochoric=True, *args):
        self._isochoric = isochoric

    @property
    def is_isochoric(self):
        return self._isochoric

    def _I1(self, F):

        C = RightCauchyGreen(F, self._isochoric)
        I1 = tr(C)
        return I1

    def _I2(self, F):
        C = RightCauchyGreen(F, self._isochoric)
        return 0.5 * (self._I1(F) * self._I1(F) - tr(C * C))

    def _I3(self, F):
        C = RightCauchyGreen(F, self._isochoric)
        return det(C)

    def _I4(self, F, a0):

        C = RightCauchyGreen(F, self._isochoric)
        I4 = inner(C * a0, a0)
        return I4

    def _I5(self, F, a0):
        C = RightCauchyGreen(F, self._isochoric)
        I5 = inner(C * a0, C * a0)
        return I5

    def _I6(self, F, a0):
        return self._I4(F, a0)

    def _I7(self, F, a0):
        return self._I5(F, a0)

    def _I8(self, u, a0, b0):
        C = RightCauchyGreen(F, self._isochoric)
        I8 = inner(C * a0, C * b0)
        return I8


# Transforms #####
# Pull-back of a two-tensor from the current to the reference
# configuration
def PiolaTransform(A, F):
    J = Jacobian(F)
    B = J * A * inv(F).T
    return B


# Push-forward of a two-tensor from the reference to the current
# configuration
def InversePiolaTransform(A, F):
    J = Jacobian(F)
    B = (1 / J) * A * F.T
    return B
