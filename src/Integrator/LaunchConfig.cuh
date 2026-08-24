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

// Turn the device's numbers into a plan. Kept free of any CUDA call so that it
// can be exercised on its own; planLaunch() below feeds it the queried values.
//
//  concurrentThreads - threads the device can keep resident at the same time
//  scratchPerThread  - device memory one resident thread needs for temporaries
//  statePerSystem    - device memory one system of the batch needs
//  budget            - device memory this run may use
__host__ inline LaunchConfig makePlan
(
    const label ensembleSize,
    const label threads,
    const label concurrentThreads,
    const size_t scratchPerThread,
    const size_t statePerSystem,
    const size_t budget
)
{
    if (ensembleSize <= 0 || threads <= 0 || concurrentThreads < threads
     || scratchPerThread == 0 || statePerSystem == 0)
    {
        fprintf(stderr, "kodes::makePlan error at %s:%d: invalid arguments\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    LaunchConfig config;
    config.threads = threads;
    config.sharedMemSize = sharedMemorySize(threads);

    // never launch more threads than there are systems to integrate
    label blocks = concurrentThreads / threads;
    const label neededBlocks = (ensembleSize + threads - 1) / threads;
    if (blocks > neededBlocks)
    {
        blocks = neededBlocks;
    }

    // A resident thread costs its scratch plus the state of at least one system
    // of the batch, since a batch shorter than the grid would leave threads
    // idle. That is the price of a block, and it fixes how many blocks the
    // memory budget can afford.
    const size_t perBlock = size_t(threads) * (scratchPerThread + statePerSystem);

    if (perBlock > budget)
    {
        fprintf
        (
            stderr,
            "kodes::makePlan error at %s:%d: a single block of %d threads needs "
            "%zu MiB, more than the %zu MiB budget - launch smaller blocks\n",
            __FILE__, __LINE__, threads, perBlock >> 20, budget >> 20
        );
        std::exit(EXIT_FAILURE);
    }

    const label affordableBlocks = label(budget / perBlock);
    if (blocks > affordableBlocks)
    {
        blocks = affordableBlocks;
    }

    config.blocks = blocks;
    config.scratchSize = blocks * threads;

    const size_t scratchBytes = size_t(config.scratchSize) * scratchPerThread;

    // Spend what is left of the budget on the batch: the state of one system is
    // tiny, so this is what fills the VRAM and keeps the transfer count low.
    // By construction this is at least scratchSize systems.
    size_t batchSize = (budget - scratchBytes) / statePerSystem;

    if (batchSize > size_t(ensembleSize))
    {
        batchSize = size_t(ensembleSize);
    }

    // keep the batch a whole number of blocks, so the state loads of a warp
    // stay contiguous - unless the whole ensemble already fits in one batch,
    // where rounding down would only add a short trailing batch
    if (batchSize > size_t(threads) && batchSize < size_t(ensembleSize))
    {
        batchSize -= batchSize % size_t(threads);
    }

    config.batchSize = label(batchSize);

    return config;
}

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
