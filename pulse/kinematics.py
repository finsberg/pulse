#!/usr/bin/env python
from dolfin import det
from dolfin import grad as Grad
from dolfin import Identity
from dolfin import inner
from dolfin import inv
from dolfin import tr

from . import Constant
from .dolfin_utils import get_dimesion


# Second order identity tensor
def SecondOrderIdentity(F):
    """Return identity with same dimension as input"""
    dim = get_dimesion(F)
    return Identity(dim)


def DeformationGradient(u, isochoric=False):
    """Return deformation gradient from displacement."""
    Id = SecondOrderIdentity(u)
    F = Id + Grad(u)
    if isochoric:
        return IsochoricDeformationGradient(F)
    else:
        return F


def IsochoricDeformationGradient(F):
    """Return the isochoric deformation gradient"""
    J = Jacobian(F)
    dim = get_dimesion(F)
    return pow(J, -1.0 / float(dim)) * F


def Jacobian(F):
    """Determinant of the deformation gradient"""
    return det(F)


def EngineeringStrain(F, isochoric=False):
    """Equivalent of engineering strain"""
    Id = SecondOrderIdentity(F)
    if isochoric:
        return IsochoricDeformationGradient(F) - Id
    else:
        return F - Id


def GreenLagrangeStrain(F, isochoric=False):
    """Green-Lagrange strain tensor"""
    Id = SecondOrderIdentity(F)
    C = RightCauchyGreen(F, isochoric)
    return 0.5 * (C - Id)


def LeftCauchyGreen(F, isochoric=False):
    """Left Cauchy-Green tensor"""
    if isochoric:
        F_ = IsochoricDeformationGradient(F)
    else:
        F_ = F

    return F_ * F_.T


def RightCauchyGreen(F, isochoric=False):
    """Left Cauchy-Green tensor"""
    if isochoric:
        F_ = IsochoricDeformationGradient(F)
    else:
        F_ = F

    return F_.T * F_


def EulerAlmansiStrain(F, isochoric=False):
    """Euler-Almansi strain tensor"""
    Id = SecondOrderIdentity(F)
    b = LeftCauchyGreen(F, isochoric)
    return 0.5 * (Id - inv(b))


def I1(F, isochoric=False):

    C = RightCauchyGreen(F, isochoric)
    I1 = tr(C)
    return I1


def I2(F, isochoric=False):
    C = RightCauchyGreen(F, isochoric)
    return 0.5 * (I1(F) * I1(F) - tr(C * C))


def I3(F, isochoric=False):
    C = RightCauchyGreen(F, isochoric)
    return det(C)


def I4(F, a0=None, isochoric=False):

    if a0 is not None:
        C = RightCauchyGreen(F, isochoric)
        I4 = inner(C * a0, a0)
    else:
        I4 = Constant(0.0)
    return I4


def I5(F, a0, isochoric=False):
    if a0 is not None:
        C = RightCauchyGreen(F, isochoric)
        I5 = inner(C * a0, C * a0)
    else:
        I5 = Constant(0.0)
    return I5


def I6(F, b0):
    return I4(F, b0)


def I7(F, b0):
    return I5(F, b0)


def I8(F, a0, b0):
    if a0 is None or b0 is None:
        I8 = Constant(0.0)
    else:
        I8 = inner(F * a0, F * b0)
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
