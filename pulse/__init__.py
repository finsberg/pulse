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
import dolfin as _dolfin

flags = ["-O3", "-ffast-math", "-march=native"]
_dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
_dolfin.parameters["form_compiler"]["representation"] = "uflacs"
_dolfin.parameters["form_compiler"]["cpp_optimize"] = True
_dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)


from .utils import annotation

# try:
#     setup_parameters.setup_general_parameters()
# except Exception:
#     pass

from collections import namedtuple

Patient = namedtuple("Patient", ["geometry", "data"])

from . import utils
from . import dolfin_utils
from . import io_utils
from . import numpy_mpi
from . import kinematics
from . import solver
from . import mechanicsproblem
from . import iterate
from . import unloader
from . import geometry
from . import geometry_utils


# Subpackages
from . import material
from .material import *


from .unloader import MeshUnloader, RaghavanUnloader, FixedPointUnloader
from .geometry import (
    Geometry,
    CRLBasis,
    HeartGeometry,
    Microstructure,
    MarkerFunctions,
)
from .example_meshes import mesh_paths
from .solver import NonlinearProblem, NonlinearSolver
from .mechanicsproblem import (
    MechanicsProblem,
    BoundaryConditions,
    NeumannBC,
    RobinBC,
)

from .dolfin_utils import QuadratureSpace, MixedParameter, RegionalParameter

from .kinematics import (
    SecondOrderIdentity,
    DeformationGradient,
    Jacobian,
    GreenLagrangeStrain,
    LeftCauchyGreen,
    RightCauchyGreen,
    EulerAlmansiStrain,
    Invariants,
    PiolaTransform,
    InversePiolaTransform,
)


import logging as _logging

import daiquiri as _daiquiri


def set_log_level(level):
    from daiquiri import set_default_log_levels

    for logger in [
        utils.logger,
        dolfin_utils.logger,
        io_utils.logger,
        numpy_mpi.logger,
        solver.logger,
        mechanicsproblem.logger,
        iterate.logger,
        unloader.logger,
        geometry.logger,
        geometry_utils.logger,
    ]:
        pass


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
