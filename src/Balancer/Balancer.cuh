#ifndef KODES_BALANCER
#define KODES_BALANCER

#pragma once

#include "basic_types.cuh"
#include "system_dimensions.cuh"
#include "device_object.cuh"
#include "DeviceArray.cuh"
#include "DeviceResources.cuh"
#include "ODESystem.cuh"
#include "LaunchConfig.cuh"
#include "thread_vector_functions.cuh"
#include "KeyTable.cuh"

// Buckets the batch is spread over. The order is exact between buckets and
// arbitrary inside one, so this number is the resolution of the balancing: a
// batch of a million systems leaves ~60 of them, two warps, per bucket.
#define KODES_BALANCER_BUCKETS 16384

namespace kodes
{

// Keys one balancer may order by. The bucket count above is shared out between
// them, so every key added costs the others resolution.
constexpr label kBalancerMaxKeys = 4;

// Threads of the single block that scans the bucket histogram. A power of two,
// since the scan doubles its stride.
constexpr label kBalancerScanBlock = 512;

// Lanes of a warp, fixed on every NVIDIA device, for the shuffle that folds the
// key ranges a warp at a time
constexpr label kWarpSize = 32;

// A double mapped to the unsigned integer of the same order, so that the range
// of a batch of keys can be taken with the 64 bit integer atomics - there is no
// atomicMin for double. Setting the sign bit lifts the positives above the
// negatives; flipping every bit of a negative reverses the order its magnitude
// would otherwise impose.
__device__ inline unsigned long long orderedBits(const scalar x)
{
    const unsigned long long u = (unsigned long long)__double_as_longlong(x);

    return (u & 0x8000000000000000ULL) ? ~u : (u | 0x8000000000000000ULL);
}

__device__ inline scalar unorderedBits(const unsigned long long u)
{
    const unsigned long long b = (u & 0x8000000000000000ULL)
                               ? (u & 0x7FFFFFFFFFFFFFFFULL)
                               : ~u;

    return __longlong_as_double((long long)b);
}

// Bin of `x` in the [lo, hi] range cut into `bins` equal parts. A key that is
// not a number - a system that has already blown up - fails every comparison
// and lands in bin 0, where it cannot drag a whole warp along.
__device__ inline label
binOf(const scalar x, const scalar lo, const scalar hi, const label bins)
{
    if (!(hi > lo) || !(x > lo))
    {
        return 0;
    }

    if (x >= hi)
    {
        return bins - 1;
    }

    const label bin = label((x - lo)/(hi - lo)*scalar(bins));

    return bin < bins ? bin : bins - 1;
}

// RMS of the relative rate of change of the components of a system: an inverse
// time scale, and about as direct a measure of how small a step the system will
// need as one right hand side evaluation can give.
//
// Relative, because in a reacting mixture dT/dt is orders of magnitude above
// every dY_i/dt, so an absolute norm would report nothing but the temperature -
// which is already a key of its own. Returned as its decimal logarithm: the
// value itself runs over a dozen decades, and the bins of a bucket sort are
// equal in the key, so it is the logarithm that has to be binned.
__device__ inline scalar relativeRHSNorm
(
    const ThreadVector<const scalar> y,
    const ThreadVector<const scalar> dydt
)
{
    return log10(SMALL + sqrt(magSqr(dydt/(SMALL + cmptMag(y)))/scalar(kSystemSize)));
}

// Orders the systems of a batch by one or more scalar keys, so that
// neighbouring positions - and therefore the threads of one warp - integrate
// systems with similar properties.
//
// Every thread integrates its own system on its own, but the 32 threads of a
// warp still issue together: a warp is done when its slowest system is, and a
// stiff system among 31 easy ones keeps the other lanes idle for every step it
// takes beyond theirs. Each thread also starts a system from the step its
// slot ended the previous one with, which is a better guess when the two are
// alike. Grouping like with like is what this buys on both counts. What it
// costs, besides its own passes, is coalescing: a warp now loads and stores
// systems from anywhere in the batch, not 32 neighbours.
//
// The ordering is a bucket sort and runs entirely on the device: the keys never
// leave it. See balance() for the four passes. It decides only which position
// of the batch a system is integrated at; the solve kernel reads it through
// systemAt(), and nothing is moved.
//
// A device object in the sense of Factory/device_object.cuh: it lives in device
// memory and a host side stub holds the same views. The device object is
// placement-newed on top of a byte copy of that stub, so the constructor must
// set value members only and leave every view alone - initialising one here
// would overwrite the address the stub allocated.
class Balancer
{
protected:

    label   batchSize_;
    label   scratchSize_;
    label   blocks_;
    label   systemSize_;
    label   parameterSize_;

    label   numOfKeys_;
    label   numOfBins_;      // bins per key
    label   numOfBuckets_;   // numOfBins_^numOfKeys_

    // whether key() calls the right hand side, and therefore needs the system
    // in the calling thread's scratch slot and a vector of its own to write to
    bool    usesDerivatives_;

    // state space: numOfKeys_ per system
    KeyTable keys_;

    // one entry per system of the batch: the bucket it falls in, and - by
    // position - the system the balanced order puts there
    DeviceArray<label> bucket_;
    DeviceArray<label> order_;

    // one entry per key: the range it spans over the batch, as orderedBits
    DeviceArray<unsigned long long> keyMin_;
    DeviceArray<unsigned long long> keyMax_;

    // one entry per bucket: systems in it, then the next free position of it
    DeviceArray<label> counts_;
    DeviceArray<label> cursor_;

    // scratch space, allocated only when usesDerivatives_
    ThreadVector<scalar> dydt_;

public:

    __device__ __host__
    Balancer
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize,
        const label numOfKeys,
        const bool usesDerivatives
    )
        : batchSize_(batchSize),
          scratchSize_(scratchSize),
          blocks_(blocks),
          systemSize_(systemSize),
          parameterSize_(parameterSize),
          numOfKeys_(numOfKeys),
          numOfBins_(binsPerKey(numOfKeys)),
          numOfBuckets_(bucketsFor(numOfKeys)),
          usesDerivatives_(usesDerivatives)
    {}

    __device__ __host__
    virtual ~Balancer() = default;

    __device__ static void* operator new(size_t size, void* ptr) { return ptr; }

    // The properties that decide where a system ends up in the batch, most
    // significant first: the batch is grouped on key[0], and key[1] only
    // reorders inside a group of key[0]. numOfKeys() of them are expected.
    //
    // When usesDerivatives() is set the system has already been loaded into the
    // calling thread's scratch slot of `batch`, so `ode` can be evaluated on
    // batch->currentVector() into dydt_.
    __device__ virtual void
    key
    (
        const DeviceResources* batch,
        const ODESystem* ode,
        const label system,
        scalar* key
    ) const = 0;

    // The system the balanced order puts at `position` of the batch
    __device__ label systemAt(const label position) const
    {
        return order_[position];
    }

    // Bins one key is cut into so that numOfKeys of them still fit in
    // KODES_BALANCER_BUCKETS buckets. Walked rather than solved for, to keep
    // the count exact; it runs once, in the constructor.
    __host__ __device__ static label binsPerKey(const label numOfKeys)
    {
        if (numOfKeys <= 1)
        {
            return KODES_BALANCER_BUCKETS;
        }

        label bins = 1;

        while (true)
        {
            label total = 1;

            for (label k = 0; k < numOfKeys; ++k)
            {
                if (total > KODES_BALANCER_BUCKETS/(bins + 1))
                {
                    return bins;
                }

                total *= (bins + 1);
            }

            ++bins;
        }
    }

    __host__ __device__ static label bucketsFor(const label numOfKeys)
    {
        const label bins = binsPerKey(numOfKeys);

        label buckets = 1;

        for (label k = 0; k < numOfKeys; ++k)
        {
            buckets *= bins;
        }

        return buckets;
    }

    // The views, for the four passes to work through
    __host__ __device__ KeyTable keys() const { return keys_; }

    __host__ __device__ DeviceArray<label> bucket() const { return bucket_; }

    __host__ __device__ DeviceArray<label> order() const { return order_; }

    __host__ __device__ DeviceArray<unsigned long long> keyMin() const { return keyMin_; }

    __host__ __device__ DeviceArray<unsigned long long> keyMax() const { return keyMax_; }

    __host__ __device__ DeviceArray<label> counts() const { return counts_; }

    __host__ __device__ DeviceArray<label> cursor() const { return cursor_; }

    __host__ __device__ label batchSize() const { return batchSize_; }

    __host__ __device__ label numOfKeys() const { return numOfKeys_; }

    __host__ __device__ label numOfBins() const { return numOfBins_; }

    __host__ __device__ label numOfBuckets() const { return numOfBuckets_; }

    __host__ __device__ bool usesDerivatives() const { return usesDerivatives_; }

    // Sort the batch, on the device from start to finish. Called on the host
    // stub; `devBalancer` is its device side twin, on which the kernels
    // dispatch, and `batch` the integration method holding the batch.
    // Everything runs on the default stream, so the solve that follows sees
    // the finished order without a synchronisation.
    __host__ void balance
    (
        Balancer* devBalancer,
        DeviceResources* batch,
        const ODESystem* ode,
        const label realBatchSize,
        const LaunchConfig& config
    ) const;

    // Device memory the balancer adds per system of the batch, and per resident
    // thread. Every subclass restates both with the signature a TypeEntry
    // expects - (systemSize, parameterSize) - filling in the key count and the
    // derivative flag it already fixes. The per bucket arrays, a few hundred
    // KiB whatever the batch, are left to the headroom of the plan.
    __host__ static size_t keyBytesPerSystem(const label numOfKeys)
    {
        return size_t(numOfKeys) * sizeof(scalar) + 2 * sizeof(label);
    }

    __host__ static size_t
    keyScratchBytesPerThread(const label systemSize, const bool usesDerivatives)
    {
        return usesDerivatives ? size_t(systemSize) * sizeof(scalar) : 0;
    }

    // A balancer owns nothing per block: its passes are grid-stride loops over
    // the batch, and none of them is collective. Stated here once rather than
    // in each subclass, which is why they do not restate it.
    __host__ static size_t
    scratchBytesPerBlock(const label systemSize, const label parameterSize)
    {
        return 0;
    }

    __host__ void allocate();

    __host__ void deallocate();
};

// Pass 1: the keys of every system, and the range each of them spans
__global__ void fillKeysKernel
(
    Balancer* balancer,
    DeviceResources* batch,
    const ODESystem* ode,
    const label realBatchSize
);

// Pass 2: the bucket every system falls in, and how many fall in each
__global__ void fillBucketsKernel(Balancer* balancer, const label realBatchSize);

// Pass 3: where each bucket starts in the ordered batch
__global__ void scanBucketsKernel(Balancer* balancer);

// Pass 4: every system into a position of its bucket
__global__ void scatterOrderKernel(Balancer* balancer, const label realBatchSize);

}

#endif
