#!/usr/bin/env python
from dolfin import MPI, mpi_comm_world, compile_extension_module
import numpy as np

#These functions are copied from cbcpost https://bitbucket.org/simula_cbc/cbcpost
def broadcast(array, from_process):
    "Broadcast array to all processes"
    if not hasattr(broadcast, "cpp_module"):
        cpp_code = '''
    
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
        '''
        cpp_module = compile_extension_module(cpp_code, additional_system_headers=["dolfin/common/MPI.h"])
        
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
        cpp_code = '''
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
        '''
        gather.cpp_module = compile_extension_module(cpp_code, additional_system_headers=["dolfin/common/MPI.h"])

    cpp_module = gather.cpp_module
    array = np.array(array, dtype=np.float)
    out_array = cpp_module.gather(mpi_comm_world(), array, on_process)

    if flatten:
        return out_array

    dist = distribution(len(array))
    cumsum = [0]+[sum(dist[:i+1]) for i in range(len(dist))]
    out_array = [[out_array[cumsum[i]:cumsum[i+1]]] for i in range(len(cumsum)-1)]

    return out_array

def distribution(number):
    "Get distribution of number on all processes"
    if not hasattr(distribution, "cpp_module"):
        cpp_code = '''
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
        '''
        distribution.cpp_module = compile_extension_module(cpp_code, additional_system_headers=["dolfin/common/MPI.h"])

    cpp_module = distribution.cpp_module
    return cpp_module.distribution(mpi_comm_world(), number)

def gather_broadcast(arr):
    try:
        dtype = arr.dtype
    except AttributeError:
        dtype = np.float

    arr = gather(arr, flatten = True)
    arr = broadcast(arr, 0)

    return arr.astype(dtype)


def assign_to_vector(v, a):
    """
    Assign the value of the array a to the dolfin vector v
    """
    lr = v.local_range()
    v[:] = a[lr[0]:lr[1]]
  
    
def mpi_print(mess):
    if mpi_comm_world().rank == 0:
        print(mess)
