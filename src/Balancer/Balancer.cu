#include "Balancer.cuh"

// No KODES_DEFINE_DEVICE_OBJECT here: Balancer is the base of a table, never a
// row of one. Only the leaves declare the four statics.

__global__ void
kodes::fillKeysKernel
(
    kodes::Balancer* balancer,
    kodes::DeviceResources* batch,
    const kodes::ODESystem* ode,
    const label realBatchSize
)
{
    const KeyTable keys = balancer->keys();

    const label numOfKeys = balancer->numOfKeys();
    const bool loadSystem = balancer->usesDerivatives();

    unsigned long long lo[kBalancerMaxKeys];
    unsigned long long hi[kBalancerMaxKeys];

    for (label k = 0; k < numOfKeys; ++k)
    {
        lo[k] = 0xFFFFFFFFFFFFFFFFULL;
        hi[k] = 0ULL;
    }

    scalar key[kBalancerMaxKeys];

    for (label system = T_ID; system < realBatchSize; system += GRID_DIM)
    {
        // a key that evaluates the right hand side needs the state where the
        // mechanism expects it: in this thread's scratch slot. The solve
        // loads every system it integrates afresh, so nothing is overwritten
        // that it still needs.
        if (loadSystem)
        {
            batch->loadSystem(system);
        }

        balancer->key(batch, ode, system, key);

        for (label k = 0; k < numOfKeys; ++k)
        {
            keys(system, k) = key[k];

            // a system that has already blown up must not stretch the range
            // over which the finite keys are then binned
            if (isfinite(key[k]))
            {
                const unsigned long long bits = orderedBits(key[k]);

                if (bits < lo[k]) lo[k] = bits;
                if (bits > hi[k]) hi[k] = bits;
            }
        }
    }

    // Each range is a reduction over the whole grid. Folding each warp first
    // leaves one pair of atomics per warp instead of one per thread.
    //
    // The lanes are counted rather than taken from __activemask(), which
    // promises nothing about which threads have reconverged after the loop
    // above. Every thread of the block reaches this point, so they are all of
    // a warp's - fewer than 32 only in the last warp of a block whose size is
    // not a multiple of the warp, and a lane must not read past those.
    const label lane = label(threadIdx.x) % kWarpSize;
    const label warpLanes = min(kWarpSize, label(blockDim.x) - (label(threadIdx.x) - lane));
    const unsigned mask =
        warpLanes == kWarpSize ? 0xFFFFFFFFu : (1u << warpLanes) - 1u;

    const DeviceArray<unsigned long long> keyMin = balancer->keyMin();
    const DeviceArray<unsigned long long> keyMax = balancer->keyMax();

    for (label k = 0; k < numOfKeys; ++k)
    {
        for (label offset = kWarpSize/2; offset > 0; offset /= 2)
        {
            const unsigned long long otherLo = __shfl_down_sync(mask, lo[k], offset);
            const unsigned long long otherHi = __shfl_down_sync(mask, hi[k], offset);

            if (lane + offset < warpLanes)
            {
                if (otherLo < lo[k]) lo[k] = otherLo;
                if (otherHi > hi[k]) hi[k] = otherHi;
            }
        }

        if (lane == 0)
        {
            atomicMin(&keyMin[k], lo[k]);
            atomicMax(&keyMax[k], hi[k]);
        }
    }
}

__global__ void
kodes::fillBucketsKernel(kodes::Balancer* balancer, const label realBatchSize)
{
    const KeyTable keys = balancer->keys();
    const DeviceArray<label> bucket = balancer->bucket();
    const DeviceArray<label> counts = balancer->counts();

    const DeviceArray<unsigned long long> keyMin = balancer->keyMin();
    const DeviceArray<unsigned long long> keyMax = balancer->keyMax();

    const label numOfKeys = balancer->numOfKeys();
    const label numOfBins = balancer->numOfBins();

    for (label system = T_ID; system < realBatchSize; system += GRID_DIM)
    {
        // The keys are mixed in one index, most significant first, so the
        // buckets run in lexicographic order: key 1 only ever reorders systems
        // that already share a bin of key 0.
        label bin = 0;

        for (label k = 0; k < numOfKeys; ++k)
        {
            const scalar lo = unorderedBits(keyMin[k]);
            const scalar hi = unorderedBits(keyMax[k]);

            bin = bin*numOfBins + binOf(keys(system, k), lo, hi, numOfBins);
        }

        bucket[system] = bin;

        atomicAdd(&counts[bin], 1);
    }
}

__global__ void
kodes::scanBucketsKernel(kodes::Balancer* balancer)
{
    __shared__ label buffer[2][kBalancerScanBlock];
    __shared__ label running;

    const DeviceArray<label> counts = balancer->counts();
    const DeviceArray<label> cursor = balancer->cursor();

    const label numOfBuckets = balancer->numOfBuckets();
    const label lane = label(threadIdx.x);

    if (lane == 0)
    {
        running = 0;
    }
    __syncthreads();

    // One block walks the histogram in chunks, scanning each chunk in shared
    // memory and carrying the total of the previous ones in `running`. The
    // histogram is a few thousand entries, so a single block is plenty and a
    // second kernel for the block offsets would cost more than it saves.
    for (label base = 0; base < numOfBuckets; base += kBalancerScanBlock)
    {
        const label bucket = base + lane;
        const label count = bucket < numOfBuckets ? counts[bucket] : 0;

        label source = 0;

        buffer[0][lane] = count;
        __syncthreads();

        for (label offset = 1; offset < kBalancerScanBlock; offset *= 2)
        {
            buffer[1 - source][lane] =
                buffer[source][lane] + (lane >= offset ? buffer[source][lane - offset] : 0);

            source = 1 - source;
            __syncthreads();
        }

        // inclusive scan of the chunk, minus this bucket's own count, is where
        // the bucket starts
        if (bucket < numOfBuckets)
        {
            cursor[bucket] = running + buffer[source][lane] - count;
        }

        const label chunkTotal = buffer[source][kBalancerScanBlock - 1];
        __syncthreads();

        if (lane == 0)
        {
            running += chunkTotal;
        }
        __syncthreads();
    }
}

__global__ void
kodes::scatterOrderKernel(kodes::Balancer* balancer, const label realBatchSize)
{
    const DeviceArray<label> bucket = balancer->bucket();
    const DeviceArray<label> cursor = balancer->cursor();
    const DeviceArray<label> order = balancer->order();

    for (label system = T_ID; system < realBatchSize; system += GRID_DIM)
    {
        // whichever thread gets there first takes the next position of the
        // bucket: the systems of one bucket come out in an arbitrary order,
        // which is exactly the order the balancing does not care about
        order[atomicAdd(&cursor[bucket[system]], 1)] = system;
    }
}

__host__ void
kodes::Balancer::allocate()
{
    if (numOfKeys_ < 1 || numOfKeys_ > kBalancerMaxKeys)
    {
        fprintf
        (
            stderr,
            "kodes::Balancer::allocate error at %s:%d: %d keys, between 1 and %d expected\n",
            __FILE__, __LINE__, numOfKeys_, kBalancerMaxKeys
        );
        std::exit(EXIT_FAILURE);
    }

    keys_.allocate(numOfKeys_, batchSize_);

    bucket_.allocate(batchSize_);
    order_.allocate(batchSize_);

    keyMin_.allocate(numOfKeys_);
    keyMax_.allocate(numOfKeys_);

    counts_.allocate(numOfBuckets_);
    cursor_.allocate(numOfBuckets_);

    if (usesDerivatives_)
    {
        dydt_.allocate(scratchSize_);
    }
}

__host__ void
kodes::Balancer::deallocate()
{
    keys_.deallocate();

    bucket_.deallocate();
    order_.deallocate();

    keyMin_.deallocate();
    keyMax_.deallocate();

    counts_.deallocate();
    cursor_.deallocate();

    // never allocated otherwise, and a view that was not is not null either:
    // it is whatever the stub was constructed with
    if (usesDerivatives_)
    {
        dydt_.deallocate();
    }
}

__host__ void
kodes::Balancer::balance
(
    Balancer* devBalancer,
    DeviceResources* batch,
    const ODESystem* ode,
    const label realBatchSize,
    const LaunchConfig& config
) const
{
    if (realBatchSize <= 0 || realBatchSize > batchSize_)
    {
        fprintf(stderr, "kodes::Balancer::balance error at %s:%d: realBatchSize %d out of 1..%d\n", __FILE__, __LINE__, realBatchSize, batchSize_);
        std::exit(EXIT_FAILURE);
    }

    if (usesDerivatives_ && !ode)
    {
        fprintf(stderr, "kodes::Balancer::balance error at %s:%d: a key needs the right hand side, but no ODE was given\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    // an empty histogram, and a range that any finite key widens
    counts_.setBytes(0);
    keyMin_.setBytes(0xFF);
    keyMax_.setBytes(0x00);

    // The key pass is given the same grid and the same dynamic shared memory
    // as the solve: a key is free to evaluate the right hand side in the
    // scratch slots of the batch, which are laid out for exactly that grid,
    // and a generated mechanism reads both the thread indexing and that
    // shared block.
    kodes::fillKeysKernel<<<config.blocks, config.threads, config.sharedMemSize>>>
    (
        devBalancer, batch, ode, realBatchSize
    );
    CUDA_CHECK_LAST();

    kodes::fillBucketsKernel<<<config.blocks, config.threads>>>(devBalancer, realBatchSize);
    CUDA_CHECK_LAST();

    kodes::scanBucketsKernel<<<1, kBalancerScanBlock>>>(devBalancer);
    CUDA_CHECK_LAST();

    kodes::scatterOrderKernel<<<config.blocks, config.threads>>>(devBalancer, realBatchSize);
    CUDA_CHECK_LAST();
}
