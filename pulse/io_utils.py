import os
import numpy as np
import h5py

import dolfin
import dolfin_adjoint

from . import numpy_mpi

from .utils import logger, Text

parallel_h5py = h5py.h5.get_config().mpi

try:
    import mpi4py
    has_mpi4py = True
except ImportError:
    logger.warning("mpi4py not found. Install via 'pip install mpi4py'")
    has_mpi4py = False
    if parallel_h5py:
        raise ImportError
else:
    from mpi4py import MPI as mpi4py_MPI

try:
    import petsc4py
    has_petsc4py = True
except ImportError:
    logger.warning("petsc4py not found. Install via 'pip install petsc4py'")
    has_petsc4py = False


def passive_inflation_exists(params):

    from .config import PASSIVE_INFLATION_GROUP

    if not os.path.exists(params["sim_file"]):
        logger.info(Text.blue(("Passive inflation, Run Optimization")))
        return False

    key = PASSIVE_INFLATION_GROUP

    with h5py.File(params["sim_file"]) as h5file:
        # Check if pv point is already computed
        if key in h5file.keys():
            logger.info(Text.green(("Passive inflation, "
                                    "fetched from database")))
            ret = True
        else:
            logger.info(Text.blue(("Passive inflation, Run Optimization")))
            ret = False

    return ret


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
        comm = dolfin.mpi_comm_world()
    dolfin.MPI.barrier(comm)
    return group_exists


def copy_h5group(h5name, src, dst, comm=None, overwrite=False):

    if comm is None:
        comm = dolfin.mpi_comm_world()

    if comm.rank == 0:
        with h5py.File(h5name, "a") as h5file:

            if dst in h5file and overwrite:
                del h5file[dst]

            if dst not in h5file:
                h5file.copy(src, dst)

    dolfin.MPI.barrier(comm)


def contract_point_exists(params):

    import numpy as np

    if not os.path.exists(params["sim_file"]):
        logger.info(Text.red("Run passive inflation before systole"))
        raise IOError("Need state from passive inflation")
        return False

    h5file = h5py.File(params["sim_file"])
    key1 = config.ACTIVE_CONTRACTION
    key2 = config.CONTRACTION_POINT\
                 .format(params["active_contraction_iteration_number"])
    key3 = config.PASSIVE_INFLATION_GROUP

    if key3 not in h5file.keys():
        logger.info(Text.red("Run passive inflation before systole"))
        raise IOError("Need state from passive inflation")

    if params["phase"] == config.PHASES[0]:
        h5file.close()
        return False

    if key1 not in h5file.keys():
        h5file.close()
        return False

    try:

        # Check if pv point is already computed
        if key1 in h5file.keys() and key2 in h5file[key1].keys():
            pressure = np.array(h5file[key1][key2]["bcs"]["pressure"])[-1]
            logger.info(Text.green(("Contract point {}, pressure = {:.3f} "
                                    "{}").format(
                                        params["active_contraction_iteration_number"],
                                        pressure, "fetched from database")))
            h5file.close()
            return True
        logger.info(Text.blue(("Contract point {}, "
                               "{}").format(params["active_contraction_iteration_number"],
                                            "Run Optimization")))
        h5file.close()
        return False
    except KeyError:
        return False


def get_simulated_pressure(params):
    """
    Get the last simulated pressure stored in
    the result file specified by given parameters

    :param dict params: adjoint contracion parameters
    :returns: The final pressure
    :rtype: float

    """

    key1 = config.ACTIVE_CONTRACTION
    key2 = config.CONTRACTION_POINT.format(
        params["active_contraction_iteration_number"])

    with h5py.File(params["sim_file"], "r") as h5file:
        try:
            pressure = np.array(h5file[key1][key2]["bcs"]["pressure"])[-1]
        except Exception as ex:
            logger.info(ex)
            pressure = None

    return pressure


def gather_dictionary(d):

    def gather_dict(a):
        v = {}
        for key, val in a.iteritems():

            if isinstance(val, dict):
                v[key] = gather_dict(val)

            elif isinstance(val, (list,  np.ndarray, tuple)):
                    if len(val) == 0:
                        # If the list is empty we do nothing
                        pass

                    elif isinstance(val[0], (dolfin.Vector,
                                             dolfin.GenericVector)):
                        v[key] = {}
                        for i, f in enumerate(val):
                            v[key][i] = numpy_mpi.gather_broadcast(f.array())

                    elif isinstance(val[0], (dolfin.Function,
                                             dolfin_adjoint.Function)):
                        v[key] = {}
                        for i, f in enumerate(val):
                            v[key][i] = numpy_mpi. \
                                        gather_broadcast(f.vector().array())

                    elif isinstance(val[0], (float, int)):
                        v[key] = np.array(val, dtype=float)

                    elif isinstance(val[0], (list, np.ndarray, dict)):

                        # Make this list of lists into a dictionary
                        f = {str(i): v for i, v in enumerate(val)}
                        v[key] = gather_dict(f)

                    else:
                        raise ValueError("Unknown type {}".
                                         format(type(val[0])))

            elif isinstance(val, (float, int)):
                v[key] = np.array([float(val)], dtype=float)

            elif isinstance(val, (dolfin.Vector, dolfin.GenericVector)):
                v[key] = numpy_mpi.gather_broadcast(val.array())

            elif isinstance(val, (dolfin.Function, dolfin_adjoint.Function)):
                v[key] = numpy_mpi.gather_broadcast(val.vector().array())

            else:
                raise ValueError("Unknown type {}".format(type(val)))

        return v

    return gather_dict(d)


def open_h5py(h5name, file_mode="a", comm=dolfin.mpi_comm_world()):

    if has_mpi4py and has_petsc4py:
        assert isinstance(comm, (petsc4py.PETSc.Comm, mpi4py.MPI.Intracomm))

        if parallel_h5py:
            if isinstance(comm, petsc4py.PETSc.Comm):
                comm = comm.tompi4py()

        return h5py.File(h5name, file_mode, driver='mpio', comm=comm)
    else:
        return h5py.File(h5name, file_mode)


def get_ed_state_group(h5name, h5group):

    group = "/".join([h5group,
                      config.PASSIVE_INFLATION_GROUP,
                      'states'])
    with h5py.File(h5name, 'r') as h5file:

        if group in h5file:
            nstates = len(h5file[group])
        else:
            nstates = 0

    return "/".join([group, str(nstates-1)])


def check_h5group(h5name, h5group, delete=False, comm=dolfin.mpi_comm_world()):

    h5group_in_h5file = False
    if not os.path.isfile(h5name):
        return False

    filemode = "a" if delete else "r"
    if not os.access(h5name, os.W_OK):
        filemode = "r"
        if delete:
            logger.warning(("You do not have write access to file "
                            "{}").format(h5name))
            delete = False

    with open_h5py(h5name, filemode, comm) as h5file:
        if h5group in h5file:
            h5group_in_h5file = True
            if delete:
                if parallel_h5py:
                    logger.debug(("Deleting existing group: "
                                 "'{}'").format(h5group))
                    del h5file[h5group]

                else:
                    if dolfin.MPI.rank(comm) == 0:
                        logger.debug(("Deleting existing group: "
                                      "'{}'").format(h5group))
                        del h5file[h5group]

    return h5group_in_h5file


def check_and_delete(h5name, h5group, comm=dolfin.mpi_comm_world()):

    with open_h5py(h5name, "a", comm) as h5file:
        if h5group in h5file:

            if parallel_h5py:

                logger.debug("Deleting existing group: '{}'".format(h5group))
                del h5file[h5group]

            else:
                if comm.rank == 0:

                    logger.debug("Deleting existing group: '{}'".format(h5group))
                    del h5file[h5group]

def dict2h5_hpc(d, h5name, h5group="",
                comm=dolfin.mpi_comm_world(),
                overwrite_file=True, overwrite_group=True):
    """Create a HDF5 file and put the
    data in the dictionary in the
    same hiearcy in the HDF5 file

    Assume leaf of dictionary is either
    float, numpy.ndrray, list or
    dolfin.GenericVector.

    :param d: Dictionary to be saved
    :param h5fname: Name of the file where you want to save

    """
    if overwrite_file:
        if os.path.isfile(h5name):
            os.remove(h5name)

    file_mode = "a" if os.path.isfile(h5name) and not overwrite_file else "w"

    # IF we should append the file but overwrite the group we need to
    # check that the group does not exist. If so we need to open it in
    # h5py and delete it.
    if file_mode == "a" and overwrite_group and h5group!="":
        check_and_delete(h5name, h5group, comm)
                    

    # First gather the whole dictionary as numpy arrays


    
        
    with open_h5py(h5name, file_mode, comm) as h5file:

        def dict2h5(a, group):
            
            for key, val in a.iteritems():
               
                subgroup = "/".join([group, str(key)])
                    
                if isinstance(val, dict):
                    dict2h5(val, subgroup)
                    
                elif isinstance(val, (list,  np.ndarray, tuple)):
                                        
                    if len(val) == 0:
                        # If the list is empty we do nothing
                        pass
                    
                    elif isinstance(val[0], (dolfin.Vector, dolfin.GenericVector)):
                        for i, f in enumerate(val):

                            if parallel_h5py:
                                v = f.array()
                                h5file.create_dataset(subgroup + "/{}".format(i),data = v)
                            else:
                                v = gather_broadcast(f.array())
                                
                                if comm.rank == 0:
                                    h5file.create_dataset(subgroup + "/{}".format(i),data = v)
                                         
                            
                    elif isinstance(val[0], (dolfin.Function, dolfin_adjoint.Function)):
                        for i, f in enumerate(val):
                            
                            if parallel_h5py:
                                v = f.vector().array()
                                h5file.create_dataset(subgroup + "/{}".format(i),data=v)
                            else:
                                v = gather_broadcast(f.vector().array())
                                if comm.rank == 0:
                                    h5file.create_dataset(subgroup + "/{}".format(i),data=v)
                            
                                                      
                                         
                            
                    elif isinstance(val[0], (float, int)):
                       
                        v = np.array(val, dtype=float)
                        if parallel_h5py:
                            h5file.create_dataset(subgroup, data=v)
                        else:
                            if comm.rank == 0:
                                h5file.create_dataset(subgroup, data=v)
                            
                        
                    elif isinstance(val[0], list) or isinstance(val[0], np.ndarray) \
                         or  isinstance(val[0], dict):
                        # Make this list of lists into a dictionary
                        f = {str(i):v for i,v in enumerate(val)}
                        dict2h5(f, subgroup)                
                    
                    else:
                        raise ValueError("Unknown type {}".format(type(val[0])))
                    
                elif isinstance(val, (float, int)):
                    v = np.array([float(val)], dtype=float)
                    
                    if parallel_h5py:
                        h5file.create_dataset(subgroup, data = v)
                    else:
                        if comm.rank == 0:
                            h5file.create_dataset(subgroup, data = v)
    
                elif isinstance(val, (dolfin.Vector, dolfin.GenericVector)):
                    
                    if parallel_h5py:
                        v = val.array()
                        h5file.create_dataset(subgroup, data = v)
                    else:
                        v = gather_broadcast(val.array())
                        if comm.rank == 0:
                            h5file.create_dataset(subgroup, data = v)
                    

                elif isinstance(val, (dolfin.Function, dolfin_adjoint.Function)):
                    
                    if parallel_h5py:
                        v = val.vector().array()
                        h5file.create_dataset(subgroup,data= v)
                    else:
                        v= gather_broadcast(val.vector().array())
                        if comm.rank == 0:
                            h5file.create_dataset(subgroup,data= v)
                    
                else:
                    raise ValueError("Unknown type {}".format(type(val)))

        
        dict2h5(d, h5group)
        comm.Barrier()



def numpy_dict_to_h5(d, h5name, h5group = "", comm = dolfin.mpi_comm_world(),
                overwrite_file = True, overwrite_group=True):
    """Create a HDF5 file and put the
    data in the dictionary in the 
    same hiearcy in the HDF5 file
    
    Assume leaf of dictionary is either
    float, numpy.ndrray, list or 
    dolfin.GenericVector.

    :param d: Dictionary to be saved
    :param h5fname: Name of the file where you want to save
    
    
    """
    if overwrite_file:
        if os.path.isfile(h5name):
            os.remove(h5name)

    
    file_mode = "a" if os.path.isfile(h5name) and not overwrite_file else "w"

    # IF we should append the file but overwrite the group we need to
    # check that the group does not exist. If so we need to open it in
    # h5py and delete it.
    if file_mode == "a" and overwrite_group and h5group!="":
        check_and_delete(h5name, h5group, comm)
                    
    if comm.rank == 0:
        with h5py.File(h5name, file_mode) as h5file:

            def dict2h5(a, group):
                
                for key, val in a.iteritems():
                
                    subgroup = "/".join([group, str(key)])
                    
                    if isinstance(val, dict):
                        dict2h5(val, subgroup)

                    else:
                        assert isinstance(val, np.ndarray)
                        assert val.dtype == np.float
                    
                        h5file.create_dataset(subgroup,data= val)

            dict2h5(d, h5group)

    dolfin.MPI.barrier(comm)
    
