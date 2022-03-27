__version__ = "2022.1.1"
import logging as _logging

try:
    import dolfin as _dolfin
except ImportError:
    _logging.warning("Cannot find 'dolfin' - 'pulse' doest not work without it")
else:
    flags = ["-O3", "-ffast-math", "-march=native"]
    _dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
    _dolfin.parameters["form_compiler"]["representation"] = "uflacs"
    _dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    _dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    _dolfin.set_log_level(_logging.WARNING)
