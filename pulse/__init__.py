# flake8: noqa
import warnings as _warnings
from collections import namedtuple

from pulse.mechanicsproblem import (
    BoundaryConditions,
    MechanicsProblem,
    NeumannBC,
    RobinBC,
)

# Subpackages
from . import (
    dolfin_utils,
    io_utils,
    iterate,
    kinematics,
    material,
    mechanicsproblem,
    numpy_mpi,
    setup_parameters,
    unloader,
    utils,
)
from .__version__ import __version__
from .dolfin_utils import MixedParameter, QuadratureSpace, RegionalParameter
from .example_meshes import mesh_paths
from .geometry import (
    CRLBasis,
    Geometry,
    HeartGeometry,
    Marker,
    MarkerFunctions,
    Microstructure,
)
from .kinematics import (
    DeformationGradient,
    EulerAlmansiStrain,
    GreenLagrangeStrain,
    Invariants,
    InversePiolaTransform,
    Jacobian,
    LeftCauchyGreen,
    PiolaTransform,
    RightCauchyGreen,
    SecondOrderIdentity,
)
from .material import (
    ActiveStrain,
    ActiveStress,
    Guccione,
    HolzapfelOgden,
    LinearElastic,
    Material,
    NeoHookean,
    StVenantKirchhoff,
)
from .setup_parameters import parameters
from .unloader import FixedPointUnloader, MeshUnloader, RaghavanUnloader
from .utils import annotation

try:
    from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

    _warnings.filterwarnings(
        "ignore", category=QuadratureRepresentationDeprecationWarning
    )
    _warnings.filterwarnings("ignore", category=DeprecationWarning)
except Exception:
    pass

_warnings.filterwarnings("ignore", category=FutureWarning)
_warnings.filterwarnings("ignore", category=UserWarning)


setup_parameters.setup_general_parameters()


Patient = namedtuple("Patient", ["geometry", "data"])


__all__ = [
    "parameters",
    "setup_parameters",
    "annotation",
    "utils",
    "dolfin_utils",
    "io_utils",
    "numpy_mpi",
    "kinematics",
    "mechanicsproblem",
    "iterate",
    "unloader",
    "Geometry",
    "Marker",
    "CRLBasis",
    "HeartGeometry",
    "Microstructure",
    "MarkerFunctions",
    "QuadratureSpace",
    "MixedParameter",
    "RegionalParameter",
    "material",
    "MeshUnloader",
    "RaghavanUnloader",
    "FixedPointUnloader",
    "BoundaryConditions",
    "MechanicsProblem",
    "NeumannBC",
    "RobinBC",
    "ActiveStrain",
    "ActiveStress",
    "Material",
    "HolzapfelOgden",
    "Guccione",
    "LinearElastic",
    "NeoHookean",
    "StVenantKirchhoff",
    "SecondOrderIdentity",
    "DeformationGradient",
    "Jacobian",
    "GreenLagrangeStrain",
    "LeftCauchyGreen",
    "RightCauchyGreen",
    "EulerAlmansiStrain",
    "Invariants",
    "PiolaTransform",
    "InversePiolaTransform",
    "__version__",
]

__author__ = "Henrik Finsberg"
__credits__ = ["Henrik Finsberg"]
__license__ = "LGPL-3"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"
