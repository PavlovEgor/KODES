#ifndef KODES_LAUNCH_CONFIG
#define KODES_LAUNCH_CONFIG

#pragma once

#include <cuda_runtime.h>

#include "basic_types.cuh"

namespace kodes
{

// How a solve is mapped onto the device.
//
//  * `threads`*`blocks` == `scratchSize` threads are launched. That is the
//    number of systems being integrated *at the same time*, so it is also the
//    number of slots allocated for the per thread temporaries (Jacobian, LU
//    matrix, ...), whose size grows as systemSize^2.
//
//  * `batchSize` systems are shipped to the device per cudaMemcpy round. Only
//    the state (systemSize + parameterSize scalars plus the step bookkeeping)
//    is stored per system, so the batch can be far larger than scratchSize and
//    fill the free VRAM, which keeps the number of host<->device transfers low.
//
// Each thread walks its share of the batch in a grid-stride loop.
struct LaunchConfig
{
    label  threads       = KODES_BLOCK_SIZE; // threads per block
    label  blocks        = 0;                // blocks launched
    label  scratchSize   = 0;                // threads*blocks resident slots
    label  batchSize     = 0;                // systems per host->device batch
    size_t sharedMemSize = 0;                // dynamic shared memory per block

    label numOfBatches(const label ensembleSize) const
    {
        return (ensembleSize + batchSize - 1) / batchSize;
    }

    void print(const char* name = "kodes") const
    {
        printf
        (
            "%s launch config: %d blocks x %d threads = %d concurrent systems, "
            "batch of %d systems, %zu B shared memory per block\n",
            name, blocks, threads, scratchSize, batchSize, sharedMemSize
        );
    }
};

// Number of threads of `kernel` that the current device can keep resident at
// the same time. Depends on the kernel's register and shared memory footprint,
// hence on the mechanism being integrated, so it has to be queried at run time.
__host__ inline label maxConcurrentThreads
(
    const void* kernel,
    const label threads = KODES_BLOCK_SIZE,
    const size_t sharedMemSize = 0
)
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int blocksPerSM = 0;
    CUDA_CHECK
    (
        cudaOccupancyMaxActiveBlocksPerMultiprocessor
        (
            &blocksPerSM, kernel, threads, sharedMemSize
        )
    );

    if (blocksPerSM < 1)
    {
        fprintf
        (
            stderr,
            "kodes::maxConcurrentThreads error at %s:%d: no block of %d threads "
            "fits on a multiprocessor (%zu B of shared memory requested)\n",
            __FILE__, __LINE__, threads, sharedMemSize
        );
        std::exit(EXIT_FAILURE);
    }

    return label(blocksPerSM) * label(prop.multiProcessorCount) * threads;
}

__host__ inline size_t freeDeviceMemory()
{
    size_t freeMem = 0;
    size_t totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    return freeMem;
}

}

#endif
