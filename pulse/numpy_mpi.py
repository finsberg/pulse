#!/usr/bin/env python
"""These functions are copied from
cbcpost https://bitbucket.org/simula_cbc/cbcpost
"""
import dolfin
import numpy as np
from dolfin import MPI

from .utils import DOLFIN_VERSION_MAJOR
from .utils import mpi_comm_world


def gather_vector(u, size=None):
    comm = mpi_comm_world()

    if size is None:
        # size = int(MPI.size(comm) * MPI.sum(comm, u.size()))
        size = int(MPI.sum(comm, u.size()))

    # From this post: https://fenicsproject.discourse.group/t/gather-function-in-parallel-error/1114/4
    u_vec = dolfin.Vector(comm, size)
    # Values from everywhere on 0
    u_vec = u.gather_on_zero()
    # To everywhere from 0
    try:
        mine = comm.bcast(u_vec)
    except AttributeError:
        comm = comm.tompi4py()
        mine = comm.bcast(u_vec)

    # Reconstruct
    if comm.rank == 0:
        x = u_vec
    else:
        v = dolfin.Vector(MPI.comm_self, size)
        v.set_local(mine)
        x = v.get_local()

    return x


def compile_extension_module(cpp_code, **kwargs):
    if DOLFIN_VERSION_MAJOR >= 2018:
        headers = kwargs.get("additional_system_headers", [])
        headers = [
            "#include <Eigen/Core>",
            "#include <pybind11/pybind11.h>",
            # "using Array = Eigen::Ref<const Eigen::ArrayXi>;;"] +\
            "using Array = Eigen::ArrayXi;",
        ] + ["#include <" + h + ">" for h in headers if h != ""]
        cpp_code = "\n".join(headers) + cpp_code
        return dolfin.compile_cpp_code(cpp_code)
    else:
        return dolfin.compile_extension_module(cpp_code, **kwargs)


def broadcast(array, from_process):
    "Broadcast array to all processes"
    if not hasattr(broadcast, "cpp_module"):
        cpp_code = """

        namespace dolfin {
            std::vector<double> broadcast(const MPI_Comm mpi_comm, const Array<double>& inarray, int from_process)
            {
                int this_process = dolfin::MPI::rank(mpi_comm);
                std::vector<double> outvector(inarray.size());

                if(this_process == from_process) {
                    for(int i=0; i<inarray.size(); i++)
                    {
                        outvector[i] = inarray[i];
                    }
                }
                dolfin::MPI::barrier(mpi_comm);
                dolfin::MPI::broadcast(mpi_comm, outvector, from_process);

                return outvector;
            }
        }
        """
        if DOLFIN_VERSION_MAJOR >= 2018:
            cpp_code += """

            PYBIND11_MODULE(SIGNATURE, m)
            {
            m.def("broadcast", &dolfin::broadcast);
            }
            """
        cpp_module = compile_extension_module(
            cpp_code,
            additional_system_headers=["dolfin/common/MPI.h"],
        )

        broadcast.cpp_module = cpp_module

    cpp_module = broadcast.cpp_module

    if MPI.rank(mpi_comm_world()) == from_process:
        array = np.array(array, dtype=np.float)
        shape = array.shape
        shape = np.array(shape, dtype=np.float_)
    else:
        array = np.array([], dtype=np.float)
        shape = np.array([], dtype=np.float_)

    shape = cpp_module.broadcast(mpi_comm_world(), shape, from_process)
    array = array.flatten()

    out_array = cpp_module.broadcast(mpi_comm_world(), array, from_process)

    if len(shape) > 1:
        out_array = out_array.reshape(*shape)

    return out_array


def gather(array, on_process=0, flatten=False):
    "Gather array from all processes on a single process"
    if not hasattr(gather, "cpp_module"):
        cpp_code = """
        namespace dolfin {
            std::vector<double> gather(const MPI_Comm mpi_comm, const Array<double>& inarray, int on_process)
            {
                int this_process = dolfin::MPI::rank(mpi_comm);

                std::vector<double> outvector(dolfin::MPI::size(mpi_comm)*dolfin::MPI::sum(mpi_comm, inarray.size()));
                std::vector<double> invector(inarray.size());

                for(int i=0; i<inarray.size(); i++)
                {
                    invector[i] = inarray[i];
                }

                dolfin::MPI::gather(mpi_comm, invector, outvector, on_process);
                return outvector;
            }
        }
        """
        if DOLFIN_VERSION_MAJOR >= 2018:
            cpp_code += """

            PYBIND11_MODULE(SIGNATURE, m)
            {
            m.def("gather", &dolfin::gather);
            }
            """
        gather.cpp_module = compile_extension_module(
            cpp_code,
            additional_system_headers=["dolfin/common/MPI.h"],
        )

    cpp_module = gather.cpp_module
    array = np.array(array, dtype=np.float)
    out_array = cpp_module.gather(mpi_comm_world(), array, on_process)

    if flatten:
        return out_array

    dist = distribution(len(array))
    cumsum = [0] + [sum(dist[: i + 1]) for i in range(len(dist))]
    out_array = [[out_array[cumsum[i] : cumsum[i + 1]]] for i in range(len(cumsum) - 1)]

    return out_array


def distribution(number):
    "Get distribution of number on all processes"
    if not hasattr(distribution, "cpp_module"):
        cpp_code = """
        namespace dolfin {
            std::vector<unsigned int> distribution(const MPI_Comm mpi_comm, int number)
            {
                // Variables to help in synchronization
                int num_processes = dolfin::MPI::size(mpi_comm);
                int this_process = dolfin::MPI::rank(mpi_comm);

                std::vector<uint> distribution(num_processes);

                for(uint i=0; i<num_processes; i++) {
                    if(i==this_process) {
                        distribution[i] = number;
                    }
                    dolfin::MPI::barrier(mpi_comm);
                    dolfin::MPI::broadcast(mpi_comm, distribution, i);
                }
                return distribution;
          }
        }
        """
        if DOLFIN_VERSION_MAJOR >= 2018:
            cpp_code += """

            PYBIND11_MODULE(SIGNATURE, m)
            {
            m.def("distribution", &dolfin::distribution);
            }
            """
        distribution.cpp_module = compile_extension_module(
            cpp_code,
            additional_system_headers=["dolfin/common/MPI.h"],
        )

    cpp_module = distribution.cpp_module
    return cpp_module.distribution(mpi_comm_world(), number)


def gather_broadcast(arr):
    # try:
    #     dtype = arr.dtype
    # except AttributeError:
    #     dtype = np.float

    # arr = gather(arr, flatten=True)
    # arr = broadcast(arr, 0)

    # return arr.astype(dtype)
    return arr


def assign_to_vector(v, a):
    """
    Assign the value of the array a to the dolfin vector v
    """
    lr = v.local_range()
    v[:] = a[lr[0] : lr[1]]


def mpi_print(mess):
    if mpi_comm_world().rank == 0:
        print(mess)
