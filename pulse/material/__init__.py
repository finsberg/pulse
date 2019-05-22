from .material_model import Material

# Material models
from .holzapfelogden import HolzapfelOgden
from .guccione import Guccione
from .linearelastic import LinearElastic
from .neohookean import NeoHookean
from .stvenantkirchhoff import StVenantKirchhoff

material_models = (
    HolzapfelOgden,
    Guccione,
    NeoHookean,
    LinearElastic,
    StVenantKirchhoff,
)

material_model_names = [m.name for m in material_models]


def get_material_model(material_model):

    for model in material_models:

        if model.name == material_model:
            return model

    raise ValueError(
        ("Material model {} does not exist. " "Please use one of {}").format(
            material_model, material_model_names
        )
    )


# Active models
from .active_strain import ActiveStrain
from .active_stress import ActiveStress
