__version__ = "2021.2.0"
import logging as _logging
import warnings as _warnings

try:
    from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

    _warnings.filterwarnings(
        "ignore",
        category=QuadratureRepresentationDeprecationWarning,
    )
except ImportError:
    pass

try:
    import dolfin as _dolfin
except ImportError:
    pass
else:
    flags = ["-O3", "-ffast-math", "-march=native"]
    _dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
    _dolfin.parameters["form_compiler"]["representation"] = "uflacs"
    _dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    _dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    _dolfin.set_log_level(_logging.WARNING)
