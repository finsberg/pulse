# flake8: noqa
import warnings as _warnings

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

from . import setup_parameters
from .setup_parameters import parameters
from .utils import annotation

setup_parameters.setup_general_parameters()

from collections import namedtuple

Patient = namedtuple("Patient", ["geometry", "data"])

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
from .unloader import FixedPointUnloader, MeshUnloader, RaghavanUnloader

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
    "__version__",
]

__author__ = "Henrik Finsberg"
__credits__ = ["Henrik Finsberg"]
__license__ = "LGPL-3"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"
