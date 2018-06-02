# from .parameters import setup_adjoint_contraction_parameters
# parameters = setup_adjoint_contraction_parameters()

from collections import namedtuple
Patient = namedtuple('Patient', ['geometry', 'data'])

from . import utils
from . import dolfin_utils
from . import io_utils
from . import numpy_mpi
from . import mechanicsproblem
from .iterate import iterate
# Subpackages
from . import unloading
from . import material
from .material import *



# from .utils import logger
# from .dolfin_utils import RegionalParameter

from .kinematics import (SecondOrderIdentity,
                         DeformationGradient,
                         Jacobian,
                         GreenLagrangeStrain,
                         LeftCauchyGreen,
                         RightCauchyGreen,
                         EulerAlmansiStrain,
                         Invariants,
                         PiolaTransform,
                         InversePiolaTransform)

__version__ = '0.1'
__author__ = 'Henrik Finsberg'
__credits__ = ['Henrik Finsberg']
__license__ = 'LGPL-3'
__maintainer__ = 'Henrik Finsberg'
__email__ = 'henriknf@simula.no'
