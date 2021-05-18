__version__ = "2020.2"
import logging as _logging
import warnings as _warnings

import dolfin as _dolfin

try:
    from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

    _warnings.filterwarnings(
        "ignore", category=QuadratureRepresentationDeprecationWarning
    )
    _warnings.filterwarnings("ignore", category=DeprecationWarning)
except ImportError:
    pass

_warnings.filterwarnings("ignore", category=FutureWarning)
_warnings.filterwarnings("ignore", category=UserWarning)


flags = ["-O3", "-ffast-math", "-march=native"]
_dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
_dolfin.parameters["form_compiler"]["representation"] = "uflacs"
_dolfin.parameters["form_compiler"]["cpp_optimize"] = True
_dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
_dolfin.set_log_level(_logging.WARNING)
