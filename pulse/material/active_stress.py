import dolfin
from .. import kinematics
from .active_model import ActiveModel, check_component


class ActiveStress(ActiveModel):
    """
    Active stress model
    """
    _model = "active_stress"

    def __init__(self, *args, **kwargs):

        # Fraction of transverse stress
        # (0 = active only along fiber, 1 = equal
        # amout of tension in all directions)
        self._eta = dolfin.Constant(kwargs.pop("eta", 0.0))

        ActiveModel.__init__(self, *args, **kwargs)

    @property
    def eta(self):
        return self._eta

    def Wactive(self, F, diff=0):
        """Active stress energy
        """

        C = F.T*F
        f0 = self.f0
        I4f = dolfin.inner(C*f0, f0)
        I1 = dolfin.tr(C)
        gamma = self.activation_field
        eta = self.eta

        if diff == 0:
            return dolfin.Constant(0.5) * gamma * \
                ((I4f-1) + eta * ((I1 - 3) - (I4f - 1)))

        elif diff == 1:
            return gamma

    @property
    def type(self):
        return "ActiveStress"

    def I1(self, F, *args):
        return self._I1(F)

    def I4(self, F, component="f0", *args):

        check_component(component)
        a0 = getattr(self, component)
        return self._I4(F, a0)

    @property
    def Fa(self):
        return kinematics.SecondOrderIdentity(self.f0)

    def Fe(self, F):
        return F




