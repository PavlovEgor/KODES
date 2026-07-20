#include "kodes_mpi.cuh"

#include <cstdio>
#include <cuda_runtime.h>

// Minimal demonstration of kodes' MPI device-selection contract: an
// arbitrary number of ranks binding to an arbitrary number of visible GPUs.
//
// This example plays the role of the "host application" (MPI_Init/Finalize),
// which in a real deployment (e.g. the OpenFOAM chemistry-model plugin) is
// owned by the surrounding code, not by kodes - kodes only ever provides
// kodes::mpiSelectDevice().
//
// After mpiSelectDevice() returns, the rest of the kodes API
// (HostResources/SeulexDeviceResources/Operator/Seulex) is used exactly as
// in a single-GPU program: each rank already owns its own local slice of
// systems, and every subsequent CUDA call this thread makes lands on the
// bound device.
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const int device = kodes::mpiSelectDevice();

    char hostName[MPI_MAX_PROCESSOR_NAME];
    int hostNameLen = 0;
    MPI_Get_processor_name(hostName, &hostNameLen);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    printf(
        "[rank %d/%d] host=%s -> cuda device %d (%s)\n",
        worldRank, worldSize, hostName, device, prop.name
    );

    MPI_Finalize();
    return 0;
}
