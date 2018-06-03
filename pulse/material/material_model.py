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
    from dolfin_adjoint import Constant, Function, project
except ImportError:
    from dolfin import Constant, Function, project
    
from .. import kinematics
from ..dolfin_utils import get_dimesion, RegionalParameter


from .active_strain import ActiveStrain
from .active_stress import ActiveStress


def compressibility(model, *args, **kwargs):

    if model == "incompressible":
        return incompressible(*args, **kwargs)


def incompressible(p, J):
    return -p*(J-1.0)


class Material(object):
    """
    Initialize material model

    Parameters
    ----------

    f0 : :py:class`dolfin.Function`
        Fiber field
    gamma : :py:class`dolfin.Function`
        Activation parameter
    params : dict
        Material parameters
    active_model : str
        Active model - active strain or active stress
    s0 : :py:class`dolfin.Function`
        Sheets
    n0 : :py:class`dolfin.Function`
        Sheet - normal
    T_ref : float
        Scale factor for activation parameter (default = 1.0)
    dev_iso_split : bool
        Decouple deformation into deviatoric and isochoric deformations
    eta : float
        Fraction of transverse active tesion for active stress formulation.
        0 = active only along fiber, 1 = equal forces in all directions
        (default=0.0).
    """
    def __init__(self,  activation=None, parameters=None,
                 active_model="active_strain", T_ref=None,
                 eta=0.0, isochoric=True,
                 compressible_model="incompressible",
                 f0=None, s0=None, n0=None, *args, **kwargs):

        # Parameters
        if parameters is None:
            parameters = self.default_parameters()

        self.parameters = parameters

        for k, v in parameters.items():

            if isinstance(v, (float, int)):
                setattr(self, k, Constant(v))

            elif isinstance(v, RegionalParameter):

                setattr(self, k, Function(v.get_ind_space(), name=k))

                mat = getattr(self, k)
                matfun = v.get_function()
                ind_space = v.get_ind_space()
                mat.assign(project(matfun, ind_space))

            else:
                setattr(self, k, v)

        # Active model
        assert active_model in \
            ["active_stress", "active_strain"], \
            "The active model '{}' is not implemented.".format(active_model)

        active_args = (activation, f0, s0, n0,
                       T_ref, isochoric)
        # Activation
        if active_model == "active_stress":
            self.active = ActiveStress(*active_args, eta=eta)
        else:
            self.active = ActiveStrain(*active_args)

        self.compressible_model = compressible_model

    @property
    def name(self):
        return "generic_material"

    @property
    def f0(self):
        return self.active.f0

    @f0.setter
    def f0(self, f0):
        self.active.f0 = f0

    @property
    def s0(self):
        return self.active.s0

    @s0.setter
    def s0(self, s0):
        self.active.s0 = s0

    @property
    def n0(self):
        return self.active.n0

    @n0.setter
    def n0(self, n0):
        self.active.n0 = n0

    def __repr__(self):
        return ('{self.__class__.__name__}({self.parameters}, '
                '{self.active._model}, '
                '{self.compressible_model})').format(self=self)

    def compressibility(self, p, J):

        return compressibility(self.compressible_model, p, J)

    @property
    def active_model(self):
        return self.active.model_type

    @property
    def material_model(self):
        return self._model

    @property
    def is_isochoric(self):
        return self.active.is_isochoric

    @property
    def activation(self):
        """
        Return the activation paramter.
        If regional, this will return one parameter
        for each segment.
        """
        return self.active.activation

    @property
    def activation_field(self):
        """
        Return the contraciton paramter.
        If regional, this will return a piecewise
        constant function (DG_0)
        """
        return self.active.activation_field

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
        I1 = self.active.I1(F)
        I4f = self.active.I4(F)

        # Active stress
        Wactive = self.active.Wactive(F, diff=0)

        dim = get_dimesion(F)
        W1 = self.W_1(I1, diff=0, dim=dim)
        W4f = self.W_4(I4f, diff=0)

        W = W1 + W4f + Wactive

        return W

    def W_1(self, *args, **kwargs):
        return 0

    def W_4(self, *args, **kwargs):
        return 0

    def CauchyStress(self, F, p=None, deviatoric=False):

        I = kinematics.SecondOrderIdentity(F)

        F = dolfin.variable(F)
        # J = dolfin.variable(det(F))

        # First Piola Kirchoff
        if deviatoric:
            P = self.FirstPiolaStress(F, None)
        else:
            P = self.FirstPiolaStress(F, p)

        # Cauchy stress
        T = kinematics.InversePiolaTransform(P, F)

        return T

    def SecondPiolaStress(self, F, p=None, deviatoric=False, *args, **kwargs):

        I = kinematics.SecondOrderIdentity(F)

        f0 = self.f0
        f0f0 = dolfin.outer(f0, f0)
        
        I1 = dolfin.variable(self.active.I1(F))
        I4f = dolfin.variable(self.active.I4(F))

        Fe = self.active.Fe(F)
        Fa = self.active.Fa()
        Ce = Fe.T*Fe

        # fe = Fe*f0
        # fefe = dolfin.outer(fe, fe)

        # Elastic volume ratio
        J = dolfin.variable(dolfin.det(Fe))
        # Active volume ration
        Ja = dolfin.det(Fa)

        dim = get_dimesion(F)
        Ce_bar = pow(J, -2.0/float(dim))*Ce

        w1 = self.W_1(I1, diff=1, dim=dim)
        w4f = self.W_4(I4f, diff=1)

        # Total Stress
        S_bar = Ja * (2 * w1*I + 2 * w4f * f0f0) * dolfin.inv(Fa).T

        if self.is_isochoric:

            # Deviatoric
            Dev_S_bar = S_bar - (1.0/3.0) * dolfin.inner(S_bar, Ce_bar) \
                        * dolfin.inv(Ce_bar)

            S_mat = J**(-2.0/3.0)*Dev_S_bar
        else:
            S_mat = S_bar

        # Volumetric
        if p is None or deviatoric:
            S_vol = dolfin.zero((dim, dim))
        else:
            psi_vol = self.compressibility(p, J)
            S_vol = J * dolfin.diff(psi_vol, J) * dolfin.inv(Ce)

        # Active stress
        wactive = self.active.Wactive(F, diff=1)
        eta = self.active.eta()

        S_active = wactive * (f0f0 + eta * (I - f0f0))

        S = S_mat + S_vol + S_active

        return S

    def FirstPiolaStress(self, F, p=None, *args, **kwargs):

        I = kinematics.SecondOrderIdentity(F)
        F = dolfin.variable(F)

        # First Piola Kirchoff
        psi_iso = self.strain_energy(F)
        P_iso = dolfin.diff(psi_iso, F)

        if p is None:
            return P_iso
        else:
            J = dolfin.variable(dolfin.det(F))
            psi_vol = self.compressibility(p, J)
            P_vol = J * dolfin.diff(psi_vol, J) * dolfin.inv(F).T

            P = P_iso + P_vol

            return P
