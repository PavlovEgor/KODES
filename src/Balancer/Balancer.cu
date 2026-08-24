#include "Balancer.cuh"

// Lanes of a warp. Fixed on every NVIDIA device, and needed as a constant so
// the shuffle reduction below can be unrolled.
#define KODES_WARP 32

__global__ void
kodes::fillKeys
(
    kodes::Balancer* balancer,
    kodes::DeviceResources* resources,
    const kodes::ODESystem* ode,
    const label realBatchSize
)
{
    scalar* __restrict__ keys = balancer->keys();

    const label numOfKeys = balancer->numOfKeys();
    const label batchSize = balancer->batchSize();
    const bool  loadSystem = balancer->usesDerivatives();

    unsigned long long lo[KODES_MAX_KEYS];
    unsigned long long hi[KODES_MAX_KEYS];

    for (label k = 0; k < numOfKeys; ++k)
    {
        lo[k] = 0xFFFFFFFFFFFFFFFFULL;
        hi[k] = 0ULL;
    }

    scalar key[KODES_MAX_KEYS];

    for (label system = T_ID; system < realBatchSize; system += GRID_DIM)
    {
        // a key that evaluates the right hand side needs the state where the
        // mechanism expects it: in this thread's scratch slot
        if (loadSystem)
        {
            resources->loadSystem(system);
        }

        balancer->key(resources, ode, system, key);

        for (label k = 0; k < numOfKeys; ++k)
        {
            keys[k*batchSize + system] = key[k];

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
    // leaves one pair of atomics per warp instead of one per thread. Every
    // thread reaches this - the loop above is the only branch - so the full
    // mask is the right one.
    for (label k = 0; k < numOfKeys; ++k)
    {
        for (label offset = KODES_WARP/2; offset > 0; offset /= 2)
        {
            const unsigned long long otherLo = __shfl_down_sync(0xFFFFFFFFu, lo[k], offset);
            const unsigned long long otherHi = __shfl_down_sync(0xFFFFFFFFu, hi[k], offset);

            if (otherLo < lo[k]) lo[k] = otherLo;
            if (otherHi > hi[k]) hi[k] = otherHi;
        }

        if ((threadIdx.x & (KODES_WARP - 1)) == 0)
        {
            atomicMin(balancer->keyMin() + k, lo[k]);
            atomicMax(balancer->keyMax() + k, hi[k]);
        }
    }
}

__global__ void
kodes::fillBuckets(kodes::Balancer* balancer, const label realBatchSize)
{
    const scalar* __restrict__ keys = balancer->keys();
    label* __restrict__ bucket = balancer->bucket();
    label* __restrict__ counts = balancer->counts();

    const unsigned long long* __restrict__ keyMin = balancer->keyMin();
    const unsigned long long* __restrict__ keyMax = balancer->keyMax();

    const label numOfKeys = balancer->numOfKeys();
    const label numOfBins = balancer->numOfBins();
    const label batchSize = balancer->batchSize();

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

            bin = bin*numOfBins + binOf(keys[k*batchSize + system], lo, hi, numOfBins);
        }

        bucket[system] = bin;

        atomicAdd(&counts[bin], 1);
    }
}

__global__ void
kodes::scanBuckets(kodes::Balancer* balancer)
{
    __shared__ label buffer[2][KODES_BALANCER_SCAN_BLOCK];
    __shared__ label running;

    const label* __restrict__ counts = balancer->counts();
    label* __restrict__ cursor = balancer->cursor();

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
    for (label base = 0; base < numOfBuckets; base += KODES_BALANCER_SCAN_BLOCK)
    {
        const label bucket = base + lane;
        const label count = bucket < numOfBuckets ? counts[bucket] : 0;

        label source = 0;

        buffer[0][lane] = count;
        __syncthreads();

        for (label offset = 1; offset < KODES_BALANCER_SCAN_BLOCK; offset *= 2)
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

        const label chunkTotal = buffer[source][KODES_BALANCER_SCAN_BLOCK - 1];
        __syncthreads();

        if (lane == 0)
        {
            running += chunkTotal;
        }
        __syncthreads();
    }
}

__global__ void
kodes::scatterOrder(kodes::Balancer* balancer, const label realBatchSize)
{
    const label* __restrict__ bucket = balancer->bucket();
    label* __restrict__ cursor = balancer->cursor();
    label* __restrict__ order = balancer->order();

    for (label system = T_ID; system < realBatchSize; system += GRID_DIM)
    {
        // whichever thread gets there first takes the next slot of the bucket:
        // the systems of one bucket come out in an arbitrary order, which is
        // exactly the order the balancing does not care about
        order[atomicAdd(&cursor[bucket[system]], 1)] = system;
    }
}

__host__ void
kodes::Balancer::allocate()
{
    CUDA_CHECK(cudaMalloc(&keys_, size_t(numOfKeys_) * size_t(batchSize_) * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&bucket_, size_t(batchSize_) * sizeof(label)));
    CUDA_CHECK(cudaMalloc(&order_, size_t(batchSize_) * sizeof(label)));

    CUDA_CHECK(cudaMalloc(&keyMin_, size_t(numOfKeys_) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&keyMax_, size_t(numOfKeys_) * sizeof(unsigned long long)));

    CUDA_CHECK(cudaMalloc(&counts_, size_t(numOfBuckets_) * sizeof(label)));
    CUDA_CHECK(cudaMalloc(&cursor_, size_t(numOfBuckets_) * sizeof(label)));

    dydt_ = nullptr;

    if (usesDerivatives_)
    {
        CUDA_CHECK(cudaMalloc(&dydt_, size_t(scratchSize_) * size_t(systemSize_) * sizeof(scalar)));
    }
}

__host__ void
kodes::Balancer::deallocate()
{
    CUDA_CHECK(cudaFree(keys_));
    CUDA_CHECK(cudaFree(bucket_));
    CUDA_CHECK(cudaFree(order_));

    CUDA_CHECK(cudaFree(keyMin_));
    CUDA_CHECK(cudaFree(keyMax_));

    CUDA_CHECK(cudaFree(counts_));
    CUDA_CHECK(cudaFree(cursor_));

    if (dydt_)
    {
        CUDA_CHECK(cudaFree(dydt_));
    }
}

__host__ void
kodes::Balancer::balance
(
    Balancer* devBalancer,
    DeviceResources* resources,
    const ODESystem* ode,
    const label realBatchSize,
    const LaunchConfig& config
)
{
    if (realBatchSize <= 0 || realBatchSize > batchSize_)
    {
        fprintf(stderr, "Balancer::balance error at %s:%d: realBatchSize out of range\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    if (usesDerivatives_ && !ode)
    {
        fprintf(stderr, "Balancer::balance error at %s:%d: a key needs the right hand side, but no ODE was given\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    // an empty histogram, and a range that any finite key widens
    CUDA_CHECK(cudaMemset(counts_, 0, size_t(numOfBuckets_) * sizeof(label)));
    CUDA_CHECK(cudaMemset(keyMin_, 0xFF, size_t(numOfKeys_) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(keyMax_, 0x00, size_t(numOfKeys_) * sizeof(unsigned long long)));

    // The key kernel is given the same grid and the same dynamic shared memory
    // as the solve: a key is free to evaluate the right hand side, and a
    // generated mechanism reads both the thread indexing and that shared block.
    kodes::fillKeys<<<config.blocks, config.threads, config.sharedMemSize>>>
    (
        devBalancer, resources, ode, realBatchSize
    );
    CUDA_CHECK_LAST();

    kodes::fillBuckets<<<config.blocks, config.threads>>>(devBalancer, realBatchSize);
    CUDA_CHECK_LAST();

    kodes::scanBuckets<<<1, KODES_BALANCER_SCAN_BLOCK>>>(devBalancer);
    CUDA_CHECK_LAST();

    kodes::scatterOrder<<<config.blocks, config.threads>>>(devBalancer, realBatchSize);
    CUDA_CHECK_LAST();
}
