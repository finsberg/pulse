import os

import dolfin
import h5py

from .utils import getLogger
from .utils import mpi_comm_world

logger = getLogger(__name__)


parallel_h5py = h5py.h5.get_config().mpi

try:
    import mpi4py

    has_mpi4py = True
except ImportError:
    has_mpi4py = False
    if parallel_h5py:
        raise ImportError

try:
    import petsc4py

    has_petsc4py = True
except ImportError:
    has_petsc4py = False


def open_h5py(h5name, file_mode="a", comm=mpi_comm_world()):

    if parallel_h5py:
        if has_petsc4py:
            petsc4py.init()
        if has_mpi4py and has_petsc4py:
            assert isinstance(comm, (petsc4py.PETSc.Comm, mpi4py.MPI.Intracomm))

        if has_petsc4py and isinstance(comm, petsc4py.PETSc.Comm):
            comm = comm.tompi4py()

        return h5py.File(h5name, file_mode, driver="mpio", comm=comm)
    else:
        return h5py.File(h5name, file_mode)


def check_h5group(h5name, h5group, delete=False, comm=mpi_comm_world()):

    h5group_in_h5file = False
    if not os.path.isfile(h5name):
        return False

    filemode = "a" if delete else "r"
    if not os.access(h5name, os.W_OK):
        filemode = "r"
        if delete:
            logger.warning(
                ("You do not have write access to file " "{}").format(h5name),
            )
            delete = False

    with open_h5py(h5name, filemode, comm) as h5file:
        if h5group in h5file:
            h5group_in_h5file = True
            if delete:
                if parallel_h5py:
                    logger.debug(("Deleting existing group: " "'{}'").format(h5group))
                    del h5file[h5group]

                else:
                    if dolfin.MPI.rank(comm) == 0:
                        logger.debug(
                            ("Deleting existing group: " "'{}'").format(h5group),
                        )
                        del h5file[h5group]

    return h5group_in_h5file


def read_h5file(h5file, obj, group, *args, **kwargs):

    # Hack in order to work with fenics-adjoint
    # if not hasattr(obj, "create_block_variable"):
    #     obj.create_block_variable = lambda: None

    h5file.read(obj, group, *args, **kwargs)
