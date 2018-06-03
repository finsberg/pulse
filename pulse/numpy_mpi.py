#!/usr/bin/env python
# c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED
# Authors: Henrik Finsberg
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS:
# post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
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
    arr = gather(arr, flatten = True)
    arr = broadcast(arr, 0)
    return arr

def assign_to_vector(v, a):
    """
    Assign the value of the array a to the dolfin vector v
    """
    lr = v.local_range()
    v[:] = a[lr[0]:lr[1]]
  
    
def mpi_print(mess):
    if mpi_comm_world().rank == 0:
        print(mess)
