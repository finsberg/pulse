from .__version__ import __version__
from .utils import annotation

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
from . import material


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
from .material import (
    LinearElastic,
    NeoHookean,
    HolzapfelOgden,
    Guccione,
    StVenantKirchhoff,
    ActiveModel,
    ActiveStrain,
    ActiveStress,
    Material,
)


import logging as _logging

import daiquiri as _daiquiri


def set_log_level(level):
    from daiquiri import set_default_log_levels

    loggers = [
        "pulse.utils.logger",
        "pulse.dolfin_utils.logger",
        "pulse.io_utils.logger",
        "pulse.solver.logger",
        "pulse.mechanicsproblem.logger",
        "pulse.iterate.logger",
        "pulse.unloader.logger",
        "pulse.geometry.logger",
        "pulse.geometry_utils.logger",
    ]
    set_default_log_levels((logger, level) for logger in loggers)


_daiquiri.setup(level=_logging.INFO)


ffc_logger = _logging.getLogger("FFC")
ffc_logger.setLevel(_logging.WARNING)
ffc_logger.addFilter(utils.mpi_filt)

ufl_logger = _logging.getLogger("UFL")
ufl_logger.setLevel(_logging.WARNING)
ufl_logger.addFilter(utils.mpi_filt)

__author__ = "Henrik Finsberg"
__credits__ = ["Henrik Finsberg"]
__license__ = "LGPL-3.0-or-later"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"

__all__ = [
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
    "ActiveModel",
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
    "solver",
    "mesh_paths",
    "NonlinearProblem",
    "NonlinearSolver",
]

__author__ = "Henrik Finsberg"
__credits__ = ["Henrik Finsberg"]
__license__ = "LGPL-3"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"
