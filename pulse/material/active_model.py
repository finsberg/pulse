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
import dolfin

try:
    from dolfin_adjoint import Constant
except ImportError:
    from dolfin import Constant

from .. import kinematics
from ..dolfin_utils import RegionalParameter


def check_component(component):
    components = ("f0", "s0", "n0")
    msg = ("Component must be one " "of {}, got {}").format(component, components)
    assert component in components, msg


class ActiveModel(kinematics.Invariants):
    def __init__(
        self,
        activation=None,
        f0=None,
        s0=None,
        n0=None,
        T_ref=None,
        isochoric=True,
        *args,
        **kwargs
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

        kinematics.Invariants.__init__(self, isochoric, *args)

    @property
    def model_type(self):
        return self._model

    def Wactive(self, *args, **kwargs):
        return 0

    @property
    def eta(self):
        return 0

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
    def activation(self):
        """
        Return the activation paramter.
        If regional, this will return one parameter
        for each segment.
        """
        return self._activation

    @activation.setter
    def activation(self, f):
        self._activation = f

    def I1(self, F):
        return self._I1(F)

    def I2(self, F):
        return self._I2(F)

    def I3(self, F):
        return self._I3(F)

    def I4(self, F, a0):
        return self._I4(F, a0)

    def I5(self, F, a0):
        return self._I5(F, a0)

    def I6(self, F, a0):
        return self._I6(F, a0)

    def I7(self, F, a0):
        return self._I7(F, a0)

    def I8(self, F, a0, b0):
        return self._I8(F, a0, b0)
