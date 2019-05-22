from .material_model import Material


class NeoHookean(Material):
    """
    Class for Neo Hookean material
    """

    name = "neo_hookean"

    @staticmethod
    def default_parameters():
        return {"mu": 15.0}

    def W_1(self, I_1, diff=0, dim=3, *args, **kwargs):

        mu = self.mu

        if diff == 0:
            return 0.5 * mu * (I_1 - dim)
        elif diff == 1:
            return 0.5 * mu
        elif diff == 2:
            return 0
