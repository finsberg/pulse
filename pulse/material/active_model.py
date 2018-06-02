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
import dolfin_adjoint

from .. import kinematics
from ..dolfin_utils import RegionalParameter


def check_component(component):
    components = ('f0', 's0', 'n0')
    msg = ('Component must be one '
           'of {}, got {}').format(component,
                                   components)
    assert component in components, msg


class ActiveModel(kinematics.Invariants):
    def __init__(self, activation=None,
                 f0=None, s0=None,
                 n0=None, T_ref=None,
                 isochoric=True, *args):

        # Fiber system
        self.f0 = f0
        self.s0 = s0
        self.n0 = n0

        self._activation = dolfin_adjoint.Constant(0, name="activation") \
            if activation is None else activation

        self.T_ref = dolfin.Constant(T_ref) \
            if T_ref else dolfin.Constant(1.0)

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
            activation = self._activation.get_function()
        else:
            activation = self._activation

        return self.T_ref*activation

    @property
    def activation(self):
        """
        Return the activation paramter.
        If regional, this will return one parameter
        for each segment.
        """
        return self._activation


if __name__ == "__main__":

    from patient_data import LVTestPatient
    patient = LVTestPatient()

    from pulse_adjoint.setup_parameters import setup_general_parameters
    setup_general_parameters()

    V = dolfin.VectorFunctionSpace(patient.mesh, "CG", 2)
    u0 = dolfin.Function(V)
    # u1 = df.Function(V, "../tests/data/inflate_mesh_simple_1.xml")

    I = dolfin.Identity(3)
    F0 = dolfin.grad(u0) + I
    # F1 = df.grad(u1) + I

    f0 = patient.fiber
    s0 = None  # patient.sheet
    n0 = None  # patient.sheet_normal
    T_ref = None
    activation = None  # dolfin_adjoint.Constant(0.0)
    dev_iso_split = False

    active_args = (activation, f0, s0, n0,
                   T_ref, dev_iso_split)

    for Active in [ActiveStrain, ActiveStress]:

        active = Active(*active_args)

        print active.type()

        active.Fa()
        active.Fa()

        active.Fe(F0)
        # active.Fe(F1)

        active.I1(F0)
        # active.I1(F1)

        active.I4(F0, "fiber")
        # active.I4(F1, "fiber")

        active.Wactive(F0)
        # active.Wactive(F1)

        active.get_gamma()
        active.get_activation()

        active.is_isochoric()
