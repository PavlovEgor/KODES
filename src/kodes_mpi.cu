#include "kodes_mpi.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace kodes
{

int mpiSelectDevice(MPI_Comm comm)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        throw std::runtime_error(
            "kodes::mpiSelectDevice: MPI_Init has not been called. kodes "
            "never calls MPI_Init/MPI_Finalize itself; the host application "
            "must manage the MPI lifecycle."
        );
    }

    int worldRank = 0;
    MPI_Comm_rank(comm, &worldRank);

    MPI_Comm nodeComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);

    int nodeRank = 0;
    int nodeSize = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    const cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0)
    {
        throw std::runtime_error(
            "kodes::mpiSelectDevice: no CUDA device visible to rank " +
            std::to_string(worldRank)
        );
    }

    const int device = nodeRank % deviceCount;
    cudaSetDevice(device);

    if (nodeSize > deviceCount)
    {
        fprintf(
            stderr,
            "kodes::mpiSelectDevice: rank %d shares CUDA device %d with "
            "other ranks on its node (%d ranks, %d visible device(s))\n",
            worldRank, device, nodeSize, deviceCount
        );
    }

    return device;
}

} // namespace kodes
