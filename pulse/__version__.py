__version__ = "2023.0.0"
import logging as _logging

try:
    import dolfin as _dolfin  # noqa: F401
except ImportError:
    _logging.warning("Cannot find 'dolfin' - 'pulse' doest not work without it")
