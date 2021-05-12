import warnings as _warnings

try:
    from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

    _warnings.filterwarnings(
        "ignore", category=QuadratureRepresentationDeprecationWarning
    )
    _warnings.filterwarnings("ignore", category=DeprecationWarning)
except:
    pass

_warnings.filterwarnings("ignore", category=FutureWarning)
_warnings.filterwarnings("ignore", category=UserWarning)

from . import setup_parameters
from .setup_parameters import parameters
from .utils import annotation

try:
    setup_parameters.setup_general_parameters()
except Exception:
    pass

from collections import namedtuple

Patient = namedtuple("Patient", ["geometry", "data"])

import logging as _logging

import daiquiri as _daiquiri
import dolfin as _dolfin
from daiquiri import set_default_log_levels as set_log_level

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
from .dolfin_utils import MixedParameter, QuadratureSpace, RegionalParameter
from .example_meshes import mesh_paths
from .geometry import CRLBasis, Geometry, HeartGeometry, MarkerFunctions, Microstructure
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
from .material import *
from .unloader import FixedPointUnloader, MeshUnloader, RaghavanUnloader

_daiquiri.setup(level=_logging.INFO)


_dolfin.set_log_level(_logging.WARNING)

ffc_logger = _logging.getLogger("FFC")
ffc_logger.setLevel(_logging.WARNING)
ffc_logger.addFilter(utils.mpi_filt)

ufl_logger = _logging.getLogger("UFL")
ufl_logger.setLevel(_logging.WARNING)
ufl_logger.addFilter(utils.mpi_filt)


from .__version__ import __version__

__author__ = "Henrik Finsberg"
__credits__ = ["Henrik Finsberg"]
__license__ = "LGPL-3.0-or-later"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"

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
    "set_log_level",
    "__version__",
]

__author__ = "Henrik Finsberg"
__credits__ = ["Henrik Finsberg"]
__license__ = "LGPL-3"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"
