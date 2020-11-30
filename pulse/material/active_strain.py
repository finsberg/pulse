import dolfin

from .. import kinematics
from ..dolfin_utils import get_dimesion
from .active_model import ActiveModel


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

    @property
    def Fa(self):

        f0 = self.f0
        d = get_dimesion(f0)
        f0f0 = dolfin.outer(f0, f0)
        Id = kinematics.SecondOrderIdentity(f0f0)

        mgamma = self._mgamma
        Fa = mgamma * f0f0 + pow(mgamma, -1.0 / float(d - 1)) * (Id - f0f0)

        return Fa

    def Fe(self, F):

        Fa = self.Fa
        Fe = F * dolfin.inv(Fa)

        return Fe
