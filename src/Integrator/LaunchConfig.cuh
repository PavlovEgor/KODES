#ifndef KODES_LAUNCH_CONFIG
#define KODES_LAUNCH_CONFIG

#pragma once

#include <cuda_runtime.h>
#include <string.h>

#include "basic_types.cuh"

// Fraction of the free VRAM a plan may claim, whatever share is asked for
#define KODES_MEMORY_HEADROOM 0.8

namespace kodes
{

// Named shares of the device. Extend this table to add a name.
struct DeviceShare
{
    const char* name;
    scalar      value;
};

inline constexpr DeviceShare deviceShares[] =
{
    {"best", 1.0},   // everything the device offers
    {"half", 0.5}    // one half of it, to leave room for another process
};

// How a solve is mapped onto the device.
//
//  * threads*blocks == scratchSize threads are launched. That is the number of
//    systems integrated at the same time, hence the number of slots allocated
//    for the per thread temporaries, whose size grows as systemSize^2.
//
//  * batchSize systems are shipped to the device per transfer. Only the state
//    is stored per system, so the batch can be far larger than scratchSize and
//    fill the free VRAM, which keeps the number of transfers low.
//
// Constructed either from a share name ("best", "half", ... see deviceShares)
// or from explicit sizes; planLaunch() turns it into the final plan.
class LaunchConfig
{
public:

    label  threads       = KODES_BLOCK_SIZE;
    label  blocks        = 0;
    label  scratchSize   = 0;
    label  batchSize     = 0;
    size_t sharedMemSize = 0;

    __host__ explicit LaunchConfig
    (
        const char* shareName = "best",
        const label threadsPerBlock = KODES_BLOCK_SIZE
    )
    :
    threads(threadsPerBlock)
    {
        for (const DeviceShare& share : deviceShares)
        {
            if (strcmp(share.name, shareName) == 0)
            {
                share_ = share.value;
                return;
            }
        }

        fprintf(stderr, "LaunchConfig error at %s:%d: unknown share \"%s\", known are", __FILE__, __LINE__, shareName);
        for (const DeviceShare& share : deviceShares)
        {
            fprintf(stderr, " \"%s\"", share.name);
        }
        fprintf(stderr, "\n");
        std::exit(EXIT_FAILURE);
    }

    __host__ LaunchConfig
    (
        const label concurrentSystems,
        const label systemsPerBatch,
        const label threadsPerBlock = KODES_BLOCK_SIZE
    )
    :
    threads(threadsPerBlock),
    scratchSize(concurrentSystems),
    batchSize(systemsPerBatch),
    byHand_(true)
    {}

    __host__ scalar share() const { return share_; }

    __host__ bool setByHand() const { return byHand_; }

    __host__ label numOfBatches(const label ensembleSize) const
    {
        return (ensembleSize + batchSize - 1) / batchSize;
    }

    __host__ void print(const char* name = "kodes") const
    {
        printf
        (
            "%s launch config: %d blocks x %d threads = %d concurrent systems, "
            "batch of %d systems, %zu B shared memory per block\n",
            name, blocks, threads, scratchSize, batchSize, sharedMemSize
        );
    }

private:

    scalar share_ = 1.0;
    bool   byHand_ = false;
};

// Turn the device's numbers into a plan. Free of any CUDA call, so it can be
// exercised on its own; planLaunch() feeds it the queried values.
//
//  concurrentThreads - threads the device can keep resident at the same time
//  scratchPerThread  - device memory one resident thread needs for temporaries
//  statePerSystem    - device memory one system of the batch needs
__host__ inline LaunchConfig makePlan
(
    const LaunchConfig& request,
    const label ensembleSize,
    const label concurrentThreads,
    const size_t scratchPerThread,
    const size_t statePerSystem,
    const size_t freeMemory
)
{
    const label threads = request.threads;

    if (ensembleSize <= 0 || threads <= 0 || concurrentThreads < threads
     || scratchPerThread == 0 || statePerSystem == 0)
    {
        fprintf(stderr, "kodes::makePlan error at %s:%d: invalid arguments\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    LaunchConfig config = request;
    config.sharedMemSize = sharedMemorySize(threads);

    if (request.setByHand())
    {
        if (config.scratchSize <= 0 || config.scratchSize % threads != 0 || config.batchSize <= 0)
        {
            fprintf
            (
                stderr,
                "kodes::makePlan error at %s:%d: scratchSize %d must be a positive "
                "multiple of %d threads and batchSize %d must be positive\n",
                __FILE__, __LINE__, config.scratchSize, threads, config.batchSize
            );
            std::exit(EXIT_FAILURE);
        }

        const size_t asked = size_t(config.scratchSize) * scratchPerThread
                           + size_t(config.batchSize) * statePerSystem;

        if (asked > freeMemory)
        {
            fprintf
            (
                stderr,
                "kodes::makePlan error at %s:%d: %zu MiB asked for by hand, only "
                "%zu MiB free on the device\n",
                __FILE__, __LINE__, asked >> 20, freeMemory >> 20
            );
            std::exit(EXIT_FAILURE);
        }

        config.blocks = config.scratchSize / threads;

        return config;
    }

    const size_t budget = size_t(double(freeMemory) * KODES_MEMORY_HEADROOM * request.share());

    label concurrency = label(concurrentThreads * request.share());
    if (concurrency < threads)
    {
        concurrency = threads;
    }

    // never launch more threads than there are systems to integrate
    label blocks = concurrency / threads;
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

    // Whatever is left of the budget goes to the batch: the state of one system
    // is tiny, so this is what fills the VRAM and keeps the transfer count low.
    // By construction it is at least scratchSize systems.
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

// Threads of `kernel` the current device can keep resident at the same time.
// Depends on the kernel's register and shared memory footprint, hence on the
// mechanism being integrated, so it has to be queried at run time.
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
