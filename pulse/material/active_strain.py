import dolfin
from .. import kinematics
from ..dolfin_utils import get_dimesion
from .active_model import ActiveModel, check_component


class ActiveStrain(ActiveModel):
    """
    This class implements the elastic invariants within
    the active strain framework

    Assuming transversally isotropic material for now

    """

    _model = "active_strain"

    @property
    def _mgamma(self):
        gamma = self.activation_field

        # FIXME: should allow for different active strain models
        if 1:
            mgamma = 1 - gamma
        elif self._model == "rossi":
            mgamma = 1 + gamma

        return mgamma

    def I1(self, F):

        I1 = self._I1(F)
        f0 = self.f0
        I4f = self._I4(F, f0)

        d = get_dimesion(F)
        mgamma = self._mgamma

        I1e = pow(mgamma, 4 - d) * I1 + (1 / mgamma ** 2 - pow(mgamma, 4 - d)) * I4f

        return I1e

    def I4(self, F, component="f0"):
        r"""
        Quasi-invariant in the elastic configuration
        Let :math:`d` be the geometric dimension.
        If

        .. math::

           \mathbf{F}_a = (1 - \gamma) \mathbf{f}_0 \otimes \mathbf{f}_0  +
           \frac{1}{\sqrt{1 - \gamma}}
           (\mathbf{I} - \mathbf{f}_0 \otimes \mathbf{f}_0)

        then

        .. math::

           I_{4f_0}^E = I_{4f_0} \frac{1}{(1+\gamma)^2}

        If

        .. math::

           \mathbf{F}_a = (1 + \gamma) \mathbf{f}_0 \otimes \mathbf{f}_0  +
           \frac{1}{\sqrt{1 + \gamma}}
           (\mathbf{I} - \mathbf{f}_0 \otimes \mathbf{f}_0)

        then

        .. math::

           I_{4f_0}^E = I_{4f_0} \frac{1}{(1+\gamma)^2}


        """

        check_component(component)
        a0 = getattr(self, component)
        I4f = self._I4(F, a0)
        mgamma = self._mgamma

        I4a0 = 1 / mgamma ** 2 * I4f

        return I4a0

    @property
    def Fa(self):

        f0 = self.f0
        d = get_dimesion(f0)
        f0f0 = dolfin.outer(f0, f0)
        I = kinematics.SecondOrderIdentity(f0f0)

        mgamma = self._mgamma
        Fa = mgamma * f0f0 + pow(mgamma, -1.0 / float(d - 1)) * (I - f0f0)

        return Fa

    def Fe(self, F):

        Fa = self.Fa
        Fe = F * dolfin.inv(Fa)

        return Fe
