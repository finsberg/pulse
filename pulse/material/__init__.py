# Material models
# Active models
from .active_model import ActiveModel
from .active_model import ActiveModels
from .guccione import Guccione
from .holzapfelogden import HolzapfelOgden
from .linearelastic import LinearElastic
from .material_model import Material
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
            material_model,
            material_model_names,
        ),
    )


__all__ = [
    "Material",
    "HolzapfelOgden",
    "Guccione",
    "NeoHookean",
    "LinearElastic",
    "StVenantKirchhoff",
    "ActiveModel",
    "ActiveModels",
]
