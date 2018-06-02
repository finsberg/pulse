#!/usr/bin/env python
# c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED
# Authors: Henrik Finsberg
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS:
# post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
from dolfin import (det, grad as Grad,
                    inv, Identity, tr, inner)

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
    return pow(J, -1.0/float(dim))*F


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
    return 0.5*(C - I)


def LeftCauchyGreen(F, isochoric=False):
    """Left Cauchy-Green tensor
    """
    if isochoric:
        F_ = IsochoricDeformationGradient(F)
    else:
        F_ = F

    return F_*F_.T


def RightCauchyGreen(F, isochoric=False):
    """Left Cauchy-Green tensor
    """
    if isochoric:
        F_ = IsochoricDeformationGradient(F)
    else:
        F_ = F

    return F_.T*F_


def EulerAlmansiStrain(F, isochoric=False):
    """Euler-Almansi strain tensor
    """
    I = SecondOrderIdentity(F)
    b = LeftCauchyGreen(F, isochoric)
    return 0.5*(I - inv(b))


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
        raise NotImplementedError

    def _I3(self, F):
        raise NotImplementedError

    def _I4(self, F, a0):

        C = RightCauchyGreen(F, self._isochoric)
        I4 = inner(C*a0, a0)
        return I4

    def _I5(self, u, a0):
        raise NotImplementedError

    def _I6(self, u, a0):
        raise NotImplementedError

    def _I8(self, u, a0, b0):
        raise NotImplementedError


# Transforms #####
# Pull-back of a two-tensor from the current to the reference
# configuration
def PiolaTransform(A, F):
    J = Jacobian(F)
    B = J*A*inv(F).T
    return B


# Push-forward of a two-tensor from the reference to the current
# configuration
def InversePiolaTransform(A, F):
    J = Jacobian(F)
    B = (1/J)*A*F.T
    return B
