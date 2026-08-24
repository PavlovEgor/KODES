#ifndef KODES_BALANCER
#define KODES_BALANCER

#pragma once

#include "basic_types.cuh"
#include "DeviceResources.cuh"
#include "ODESystem.cuh"
#include "LaunchConfig.cuh"

// Buckets the batch is spread over. The order is exact between buckets and
// arbitrary inside one, so this number is the resolution of the balancing: a
// batch of a million systems leaves ~60 of them, two warps, per bucket.
#define KODES_BALANCER_BUCKETS 16384

// Keys one balancer may order by. The bucket count above is shared out between
// them, so every key added costs the others resolution.
#define KODES_MAX_KEYS 4

// Threads of the single block that scans the bucket histogram. A power of two,
// since the scan doubles its stride.
#define KODES_BALANCER_SCAN_BLOCK 512

namespace kodes
{

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
__device__ inline scalar
relativeRHSNorm(const scalar* y, const scalar* dydt, const label systemSize)
{
    scalar sum = 0.0;

    for (label i = 0; i < systemSize; ++i)
    {
        const scalar rate = dydt[INDEXVEC(i)]/(SMALL + fabs(y[INDEXVEC(i)]));

        sum += rate*rate;
    }

    return log10(SMALL + sqrt(sum/scalar(systemSize)));
}

// Orders the systems of a batch by one or more scalar keys, so that
// neighbouring positions - and therefore the threads of one warp - integrate
// systems with similar properties. A warp runs at the speed of its stiffest
// member, so grouping like with like is what keeps the other 31 lanes from
// idling.
//
// The ordering is a bucket sort and runs entirely on the device: the keys never
// leave it. See balance() for the four passes.
//
// Follows the DeviceResources pattern: the object lives in device memory and a
// host side stub holds the same pointers. The device object is placement-newed
// on top of a byte copy of that stub, so the constructor must set value members
// only and leave every buffer pointer alone - initialising one here would
// overwrite the address the stub allocated.
class Balancer
{
protected:

    label   batchSize_;
    label   scratchSize_;
    label   systemSize_;

    label   numOfKeys_;
    label   numOfBins_;      // bins per key
    label   numOfBuckets_;   // numOfBins_^numOfKeys_

    // whether key() calls the right hand side, and therefore needs the system
    // in the calling thread's scratch slot and a slot of its own to write to
    bool    usesDerivatives_;

    // numOfKeys_*batchSize_ long, key k of system s at keys_[k*batchSize_ + s]
    scalar* keys_;

    // batchSize_ long
    label*  bucket_;
    label*  order_;

    // numOfKeys_ long: the range each key spans over the batch, as orderedBits
    unsigned long long* keyMin_;
    unsigned long long* keyMax_;

    // numOfBuckets_ long: systems per bucket, then the next free slot of each
    label*  counts_;
    label*  cursor_;

    // scratchSize_*systemSize_ long when usesDerivatives_, null otherwise
    scalar* dydt_;

public:

    __device__ __host__
    Balancer
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        const label numOfKeys,
        const bool usesDerivatives
    )
        : batchSize_(batchSize),
          scratchSize_(scratchSize),
          systemSize_(systemSize),
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
    // calling thread's scratch slot, so `ode` can be evaluated on
    // resources->currentVector() into dydt().
    __device__ virtual void
    key
    (
        DeviceResources* resources,
        const ODESystem* ode,
        const label system,
        scalar* key
    ) const = 0;

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

    __host__ __device__ scalar* keys() { return keys_; }

    __host__ __device__ label* bucket() { return bucket_; }

    __host__ __device__ label* order() { return order_; }

    __host__ __device__ const label* order() const { return order_; }

    __host__ __device__ unsigned long long* keyMin() { return keyMin_; }

    __host__ __device__ unsigned long long* keyMax() { return keyMax_; }

    __host__ __device__ label* counts() { return counts_; }

    __host__ __device__ label* cursor() { return cursor_; }

    __device__ scalar* dydt() const { return dydt_; }

    __host__ __device__ label batchSize() const { return batchSize_; }

    __host__ __device__ label systemSize() const { return systemSize_; }

    __host__ __device__ label numOfKeys() const { return numOfKeys_; }

    __host__ __device__ label numOfBins() const { return numOfBins_; }

    __host__ __device__ label numOfBuckets() const { return numOfBuckets_; }

    __host__ __device__ bool usesDerivatives() const { return usesDerivatives_; }

    // Sort the batch, on the device from start to finish. Called on the host
    // stub; `devBalancer` is its device side twin, on which the kernels
    // dispatch. Everything runs on the default stream, so the solve that
    // follows sees the finished order without a synchronisation.
    __host__ void balance
    (
        Balancer* devBalancer,
        DeviceResources* resources,
        const ODESystem* ode,
        const label realBatchSize,
        const LaunchConfig& config
    );

    // Device memory the balancer adds per system of the batch, and per resident
    // thread. Subclasses restate both without the arguments they already fix.
    __host__ static size_t bytesPerSystem(const label numOfKeys)
    {
        return size_t(numOfKeys) * sizeof(scalar) + 2 * sizeof(label);
    }

    __host__ static size_t
    scratchBytesPerThread(const label systemSize, const bool usesDerivatives)
    {
        return usesDerivatives ? size_t(systemSize) * sizeof(scalar) : 0;
    }

    __host__ void allocate();

    __host__ void deallocate();
};

// Pass 1: the keys of every system, and the range each of them spans
__global__ void fillKeys
(
    Balancer* balancer,
    DeviceResources* resources,
    const ODESystem* ode,
    const label realBatchSize
);

// Pass 2: the bucket every system falls in, and how many fall in each
__global__ void fillBuckets(Balancer* balancer, const label realBatchSize);

// Pass 3: where each bucket starts in the ordered batch
__global__ void scanBuckets(Balancer* balancer);

// Pass 4: every system into a slot of its bucket
__global__ void scatterOrder(Balancer* balancer, const label realBatchSize);

}

#endif
