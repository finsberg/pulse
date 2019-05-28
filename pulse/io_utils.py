import os
import h5py
import dolfin

from .utils import make_logger, mpi_comm_world
from . import parameters

logger = make_logger(__name__, parameters["log_level"])

parallel_h5py = h5py.h5.get_config().mpi

try:
    import mpi4py

    has_mpi4py = True
except ImportError:
    has_mpi4py = False
    if parallel_h5py:
        raise ImportError
else:
    from mpi4py import MPI as mpi4py_MPI

try:
    import petsc4py

    has_petsc4py = True
except ImportError:
    has_petsc4py = False


def check_group_exists(h5name, h5group, comm=None):

    if not os.path.exists(h5name):
        return False

    try:
        h5file = h5py.File(h5name)
    except Exception as ex:
        logger.info(ex)
        return False

    group_exists = False
    if h5group in h5file:
        group_exists = True

    h5file.close()
    if comm is None:
        comm = mpi_comm_world()
    dolfin.MPI.barrier(comm)
    return group_exists


def copy_h5group(h5name, src, dst, comm=None, overwrite=False):

    if comm is None:
        comm = mpi_comm_world()

    if comm.rank == 0:
        with h5py.File(h5name, "a") as h5file:

            if dst in h5file and overwrite:
                del h5file[dst]

            if dst not in h5file:
                h5file.copy(src, dst)

    dolfin.MPI.barrier(comm)


def open_h5py(h5name, file_mode="a", comm=mpi_comm_world()):

    if parallel_h5py:
        if has_mpi4py and has_petsc4py:
            assert isinstance(comm, (petsc4py.PETSc.Comm, mpi4py.MPI.Intracomm))

        if isinstance(comm, petsc4py.PETSc.Comm):
            comm = comm.tompi4py()

        return h5py.File(h5name, file_mode, comm=comm)
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
                ("You do not have write access to file " "{}").format(h5name)
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
                            ("Deleting existing group: " "'{}'").format(h5group)
                        )
                        del h5file[h5group]

    return h5group_in_h5file


def check_and_delete(h5name, h5group, comm=mpi_comm_world()):

    with open_h5py(h5name, "a", comm) as h5file:
        if h5group in h5file:

            if parallel_h5py:

                logger.debug("Deleting existing group: '{}'".format(h5group))
                del h5file[h5group]

            else:
                if comm.rank == 0:

                    logger.debug("Deleting existing group: '{}'".format(h5group))
                    del h5file[h5group]


def read_h5file(h5file, obj, group, *args, **kwargs):

    # Hack in order to work with fenics-adjoint
    # if not hasattr(obj, "create_block_variable"):
    #     obj.create_block_variable = lambda: None

    h5file.read(obj, group, *args, **kwargs)
