#ifndef KODES_LAUNCH_CONFIG
#define KODES_LAUNCH_CONFIG

#pragma once

#include <cuda_runtime.h>
#include <string.h>

#include "basic_types.cuh"

// Fraction of the free VRAM a plan may claim, whatever share is asked for
#define KODES_MEMORY_HEADROOM 0.8

// Bytes of call stack one device thread gets, or 0 to leave the driver's own
// value alone. 0 is the default, and should stay the default.
//
// Normally the driver sizes the stack itself: ptxas records a frame size per
// function, the driver walks the kernel's call graph and reserves what it
// finds. `cuobjdump -res-usage` prints what it found - for a 159 species
// mechanism solveKernel comes out at STACK:15600 - and cudaLimitStackSize is
// only the floor under that, for the part of the graph the compiler could not
// see (an indirect call, which a virtual call on the device is).
//
// Raising the floor is expensive out of all proportion to what it buys. The
// driver reserves it for every thread the device *could* hold resident -
// multiProcessorCount * maxThreadsPerMultiProcessor - not for the ones a launch
// uses, so on a large card 128 KiB per thread is over 30 GiB, more than the card
// has, and what survives is what the batch gets to live in. A run that asked for
// it came back with 8 blocks x 64 threads for a 40000 cell case.
//
// So: only set this when a run is actually dying for want of stack, and then set
// it to the frame cuobjdump reports plus room for the indirect part, not to a
// round number that sounds safe.
#define KODES_DEVICE_STACK_BYTES 0

// Most shared memory a block may be given for the matrix it factorises. The
// device usually allows more, but shared memory is what limits how many blocks
// an SM keeps resident, and past this the occupancy lost costs more than the
// faster back substitution wins. A mechanism whose matrix does not fit under
// it works in global memory instead, which is correct either way - only
// slower.
#define KODES_SHARED_BUDGET 16384

namespace kodes
{

// Named shares of the device. Extend this table to add a name.
struct DeviceShare
{
    const char* name;
    scalar      value;
};

inline constexpr DeviceShare kDeviceShares[] =
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
//  * blockScratchInShared says where scratch a whole block shares would be
//    worked in. No method of this version shares any - every thread
//    integrates on its own - so it is always false, and kept because the plan
//    and the library interface still report it.
//
// Constructed either from a share name ("best", "half", ... see kDeviceShares)
// or from explicit sizes; planLaunch() turns it into the final plan.
class LaunchConfig
{
public:

    label  threads       = KODES_BLOCK_SIZE;
    label  blocks        = 0;
    label  scratchSize   = 0;
    label  batchSize     = 0;
    size_t sharedMemSize = 0;
    bool   blockScratchInShared = false;

    __host__ explicit LaunchConfig
    (
        const char* shareName = "best",
        const label threadsPerBlock = KODES_BLOCK_SIZE
    )
    :
    threads(threadsPerBlock)
    {
        for (const DeviceShare& share : kDeviceShares)
        {
            if (strcmp(share.name, shareName) == 0)
            {
                share_ = share.value;
                return;
            }
        }

        fprintf(stderr, "LaunchConfig error at %s:%d: unknown share \"%s\", known are", __FILE__, __LINE__, shareName);
        for (const DeviceShare& share : kDeviceShares)
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
            "batch of %d systems, %zu B shared memory per block, block scratch "
            "in %s memory\n",
            name, blocks, threads, scratchSize, batchSize, sharedMemSize,
            blockScratchInShared ? "shared" : "global"
        );
    }

private:

    scalar share_ = 1.0;
    bool   byHand_ = false;
};

// Whether a block's shared scratch is small enough to be worked in shared
// memory, once pyJac's own shared_temp has taken its share of it. Split out of
// makePlan because planLaunch has to settle it *before* asking the device for
// the occupancy of the solve kernel: the answer depends on how much shared
// memory the kernel asks for, so asking with the wrong figure would plan the
// grid against an occupancy the run never has.
__host__ inline bool blockScratchFitsShared
(
    const label threads,
    const size_t scratchPerBlock,
    const size_t maxSharedPerBlock
)
{
    // nothing to put there
    if (scratchPerBlock == 0)
    {
        return false;
    }

    const size_t budget =
        maxSharedPerBlock < size_t(KODES_SHARED_BUDGET)
      ? maxSharedPerBlock
      : size_t(KODES_SHARED_BUDGET);

    return sharedMemorySize(threads, scratchPerBlock) <= budget;
}

// Turn the device's numbers into a plan. Free of any CUDA call, so it can be
// exercised on its own; planLaunch() feeds it the queried values.
//
//  concurrentThreads - threads the device can keep resident at the same time
//  scratchPerThread  - device memory one resident thread needs for temporaries
//  scratchPerBlock   - device memory one block needs for what its threads share
//  statePerSystem    - device memory one system of the batch needs
__host__ inline LaunchConfig makePlan
(
    const LaunchConfig& request,
    const label ensembleSize,
    const label concurrentThreads,
    const size_t scratchPerThread,
    const size_t scratchPerBlock,
    const size_t statePerSystem,
    const size_t freeMemory,
    const size_t maxSharedPerBlock
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

    config.blockScratchInShared =
        blockScratchFitsShared(threads, scratchPerBlock, maxSharedPerBlock);

    config.sharedMemSize = sharedMemorySize
    (
        threads, config.blockScratchInShared ? scratchPerBlock : 0
    );

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

        config.blocks = config.scratchSize / threads;

        const size_t asked = size_t(config.scratchSize) * scratchPerThread
                           + size_t(config.blocks) * scratchPerBlock
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
    // idle. On top of that the block pays for what its threads share. That is
    // the price of a block, and it fixes how many blocks the memory budget can
    // afford.
    const size_t perBlock = size_t(threads) * (scratchPerThread + statePerSystem)
                          + scratchPerBlock;

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

    const size_t scratchBytes = size_t(config.scratchSize) * scratchPerThread
                              + size_t(config.blocks) * scratchPerBlock;

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

// What a device thread's call stack is right now.
__host__ inline size_t deviceStack()
{
    size_t granted = 0;
    CUDA_CHECK(cudaDeviceGetLimit(&granted, cudaLimitStackSize));
    return granted;
}

// Give every device thread `bytes` of call stack, and return what the driver
// actually granted (it rounds up). See KODES_DEVICE_STACK_BYTES for why the
// default will not do.
//
// Whether the driver takes the memory for the backing store here or holds off
// until the first launch is its own business. That is exactly why planLaunch()
// calls this before it counts free memory and not after: either cudaMemGetInfo
// already has the reservation out of it, or it is still to come and comes out
// of KODES_MEMORY_HEADROOM. Counting first would leave neither.
__host__ inline size_t setDeviceStack(const size_t bytes)
{
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, bytes));
    return deviceStack();
}

__host__ inline size_t maxSharedMemoryPerBlock()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    return prop.sharedMemPerBlock;
}

}

#endif
