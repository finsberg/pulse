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
from typing import Optional
from typing import Union

import dolfin
import ufl

try:
    from dolfin_adjoint import Constant, Function
except ImportError:
    from dolfin import Constant, Function

from .. import kinematics, numpy_mpi
from ..dolfin_utils import RegionalParameter, update_function
from .active_model import ActiveModel
from .active_strain import ActiveStrain
from .active_stress import ActiveStress


def compressibility(model, *args, **kwargs):

    if model == "incompressible":
        return incompressible(*args, **kwargs)


def incompressible(p, J):
    return -p * (J - 1.0)


def handle_active_model(
    active_model, activation, f0, s0, n0, T_ref, isochoric, eta, **kwargs
):
    if isinstance(active_model, str):
        assert active_model in [
            "active_stress",
            "active_strain",
        ], "The active model '{}' is not implemented.".format(active_model)

        active_args = (activation, f0, s0, n0, T_ref, isochoric)
        # Activation
        if active_model == "active_stress":
            return ActiveStress(*active_args, eta=eta, **kwargs)
        else:
            return ActiveStrain(*active_args)
    else:
        assert isinstance(active_model, ActiveModel)
        return active_model


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
        active_model: Union[ActiveModel, str] = "active_strain",
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

        # Active model
        self.active = handle_active_model(
            active_model, activation, f0, s0, n0, T_ref, isochoric, eta, **kwargs
        )

        self.compressible_model = compressible_model
        self._update_activation(activation, geometry)

    def _update_activation(self, activation, geometry):
        # FIXME: Do we need this function?
        try:
            activation_element = self.activation.ufl_element()
            cell = activation_element.cell()
        except AttributeError:
            if not hasattr(activation, "cell"):
                cell = None
            else:
                cell = activation.cell()

        if geometry is not None and cell is not None:
            self.activation = update_function(geometry.mesh, self.activation)

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
                mat.assign(dolfin.project(matfun, ind_space))

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
            T_ref=float(self.active.T_ref),
        )

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
        return (
            "{self.__class__.__name__}({self.parameters}, "
            "{self.active._model}, "
            "{self.compressible_model})"
        ).format(self=self)

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

    @activation.setter
    def activation(self, f):
        self.active.activation = f

    @property
    def activation_field(self):
        """
        Return the contraciton paramter.
        If regional, this will return a piecewise
        constant function (DG_0)
        """
        return self.active.activation_field

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
