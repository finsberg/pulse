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
from abc import ABC
from abc import abstractmethod
from abc import abstractstaticmethod
from enum import Enum
from typing import Optional

import dolfin
import ufl

try:
    from dolfin_adjoint import Constant, Function, project
except ImportError:
    from dolfin import Constant, Function, project

from .. import kinematics, numpy_mpi
from ..dolfin_utils import RegionalParameter, update_function, get_dimesion
from . import active_stress


def compressibility(model, *args, **kwargs):

    if model == "incompressible":
        return incompressible(*args, **kwargs)


def incompressible(p, J):
    return -p * (J - 1.0)


class ActiveModel(str, Enum):
    active_stress = "active_stress"
    active_strain = "active_strain"


class Material(ABC):
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

    def __init__(
        self,
        activation: Optional[ufl.Coefficient] = None,
        parameters=None,
        active_model: ActiveModel = ActiveModel.active_strain,
        T_ref: Optional[float] = None,
        eta: Optional[float] = 0.0,
        isochoric: bool = True,
        compressible_model="incompressible",
        geometry=None,
        f0=None,
        s0=None,
        n0=None,
        *args,
        **kwargs
    ):

        # Parameters
        self.parameters = self.__class__.default_parameters()
        if parameters is not None:
            self.parameters.update(parameters)
        self._set_parameter_attrs(geometry)

        self._activation = activation if activation is not None else Constant(0.0)
        self._f0 = f0
        self._s0 = s0
        self._n0 = n0
        self._T_ref = self.T_ref = (
            Constant(T_ref, name="T_ref") if T_ref else Constant(1.0, name="T_ref")
        )
        self._eta = Constant(eta, name="eta")
        self._active_model = active_model
        self.active_isotropy = kwargs.pop("active_isotropy", "transversally")
        self._isochoric = isochoric
        self.compressible_model = compressible_model
        self._set_dimension()

    def _set_dimension(self):
        try:
            self._dim = get_dimesion(self.f0)
        except Exception:
            # just assume three dimensions
            self._dim = 3

    def _set_parameter_attrs(self, geometry=None):
        for k, v in self.parameters.items():

            if isinstance(v, (float, int)):
                setattr(self, k, Constant(v, name=k))

            elif isinstance(v, RegionalParameter):

                if geometry is not None:

                    v_new = RegionalParameter(geometry.sfun)
                    numpy_mpi.assign_to_vector(
                        v_new.vector(),
                        numpy_mpi.gather_vector(v.vector()),
                    )
                    v = v_new

                ind_space = v.proj_space
                setattr(self, k, Function(ind_space, name=k))
                mat = getattr(self, k)
                matfun = v.function
                mat.assign(project(matfun, ind_space))

            else:

                if geometry is not None and v.ufl_element().cell() is not None:
                    v_new = update_function(geometry.mesh, v)
                    v = v_new

                setattr(self, k, v)

    @abstractmethod
    def strain_energy(self, F):
        pass

    @staticmethod
    @abstractstaticmethod
    def default_parameters():
        pass

    def update_geometry(self, geometry):

        # Loop over the possible attributes containing
        # a domain and update it
        for m in ("f0", "s0", "n0"):
            setattr(self, m, getattr(geometry, m))

        self._set_parameter_attrs(geometry)

        activation_element = self.activation.ufl_element()
        if activation_element.cell() is not None:

            self.activation = update_function(geometry.mesh, self.activation)

    def copy(self, geometry=None):

        if geometry is not None:
            f0, s0, n0 = geometry.f0, geometry.s0, geometry.n0
        else:
            f0, s0, n0 = self.f0, self.s0, self.n0

        return self.__class__(
            activation=self.activation,
            parameters=self.parameters,
            active_model=self.active_model,
            geometry=geometry,
            f0=f0,
            s0=s0,
            n0=n0,
            T_ref=float(self._T_ref),
        )

    @property
    def name(self):
        return "generic_material"

    def __repr__(self):
        return (
            "{self.__class__.__name__}({self.parameters}, "
            "{self.active_model}, "
            "{self.compressible_model})"
        ).format(self=self)

    def compressibility(self, p, J):

        return compressibility(self.compressible_model, p, J)

    @property
    def Fa(self):

        if self.active_model == ActiveModel.active_stress:
            return dolfin.Identity(self._dim)

        f0 = self.f0
        f0f0 = dolfin.outer(f0, f0)
        Id = kinematics.SecondOrderIdentity(f0f0)

        mgamma = 1 - self.activation_field
        Fa = mgamma * f0f0 + pow(mgamma, -1.0 / float(self._dim - 1)) * (Id - f0f0)

        return Fa

    def Fe(self, F):
        if self.active_model == ActiveModel.active_stress:
            return F
        Fa = self.Fa
        Fe = F * dolfin.inv(Fa)

        return Fe

    def Wactive(self, F=None, diff=0):
        """Active stress energy"""
        if self.active_model == ActiveModel.active_strain:
            return 0

        C = F.T * F

        if diff == 0:

            if self.active_isotropy == "transversally":
                return active_stress.Wactive_transversally(
                    Ta=self.activation_field,
                    C=C,
                    f0=self.f0,
                    eta=self.eta,
                )

            elif self.active_isotropy == "orthotropic":
                return active_stress.Wactive_orthotropic(
                    Ta=self.activation_field,
                    C=C,
                    f0=self.f0,
                    s0=self.s0,
                    n0=self.n0,
                )

            elif self.active_isotropy == "fully_anisotropic":

                return active_stress.Wactive_anisotropic(
                    Ta=self.activation_field,
                    C=C,
                    f0=self.f0,
                    s0=self.s0,
                    n0=self.n0,
                )
            else:
                msg = ("Unknown acitve isotropy " "{}").format(self.active_isotropy)
                raise ValueError(msg)

        elif diff == 1:
            return self.activation_field

    @property
    def isochoric(self):
        return self._isochoric

    @property
    def active_model(self):
        return self._active_model

    @property
    def material_model(self):
        return self._model

    @property
    def eta(self):
        return self._eta

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

    def CauchyStress(self, F, p=None, deviatoric=False):

        F = dolfin.variable(F)

        # First Piola Kirchoff
        if deviatoric:
            p = None

        P = self.FirstPiolaStress(F, p)

        # Cauchy stress
        T = kinematics.InversePiolaTransform(P, F)

        return T

    def SecondPiolaStress(self, F, p=None, deviatoric=False, *args, **kwargs):

        # First Piola Kirchoff
        if deviatoric:
            p = None

        P = self.FirstPiolaStress(F, p)

        S = dolfin.inv(F) * P * dolfin.inv(F).T

        return S

    def FirstPiolaStress(self, F, p=None, *args, **kwargs):

        F = dolfin.variable(F)

        # First Piola Kirchoff
        psi_iso = self.strain_energy(F)
        P = dolfin.diff(psi_iso, F)

        if p is not None:
            J = dolfin.variable(kinematics.Jacobian(F))
            psi_vol = self.compressibility(p, J)
            # PiolaTransform
            P_vol = J * dolfin.diff(psi_vol, J) * dolfin.inv(F).T

            P += P_vol

        return P
