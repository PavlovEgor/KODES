#pragma once

// Avoid pulling in the deprecated MPI C++ bindings on Open MPI; harmless on
// other MPI implementations.
#define OMPI_SKIP_MPICXX
#include <mpi.h>

namespace kodes
{

// Binds the calling process to a CUDA device based on its MPI rank, so an
// arbitrary number of ranks can share an arbitrary number of GPUs - including
// across multiple nodes with different GPU counts per node - without every
// rank defaulting onto device 0.
//
// KODES never calls MPI_Init/MPI_Finalize itself: the host application (e.g.
// OpenFOAM) owns the MPI lifecycle, and each rank is assumed to already own
// its own local slice of systems (mesh cells) before it touches kodes at all
// - there is no scatter/gather here, only device selection.
//
// Call this exactly once per rank, right after MPI_Init and before creating
// any device-side kodes object (DeviceResources, SeulexDeviceResources,
// Integrator, ...). Every kodes CUDA call the calling host thread makes
// afterwards (cudaMalloc, kernel launches, cudaMemcpy) targets the bound
// device, since that's ordinary per-thread CUDA runtime state - if the host
// application spawns additional threads that also call into kodes, each such
// thread must bind its own device the same way.
//
// Ranks are grouped by physical node (via a shared-memory communicator
// split), then assigned device = (rank within node) % (GPUs visible to this
// rank). More ranks than GPUs on a node round-robins the GPUs (a rank that
// ends up sharing a GPU is reported on stderr); fewer ranks than GPUs simply
// leaves some idle. Also compatible with launchers that already restrict
// each rank's visible devices (e.g. SLURM's CUDA_VISIBLE_DEVICES pinning),
// since deviceCount is then just 1.
//
// Throws std::runtime_error if MPI isn't initialized or no CUDA device is
// visible to this rank.
int mpiSelectDevice(MPI_Comm comm = MPI_COMM_WORLD);

} // namespace kodes
