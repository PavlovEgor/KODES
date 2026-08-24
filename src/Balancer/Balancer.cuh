#ifndef KODES_BALANCER
#define KODES_BALANCER

#pragma once

#include "basic_types.cuh"
#include "DeviceResources.cuh"
#include "LaunchConfig.cuh"

// Buckets the batch is spread over. The order is exact between buckets and
// arbitrary inside one, so this number is the resolution of the balancing: a
// batch of a million systems leaves ~60 of them, two warps, per bucket.
#define KODES_BALANCER_BUCKETS 16384

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

// Bucket of `x` in the [lo, hi] range cut into `bins` equal parts. A key that
// is not a number - a system that has already blown up - fails every
// comparison and lands in bin 0, where it cannot drag a whole warp along.
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

// Orders the systems of a batch by a scalar key, so that neighbouring positions
// - and therefore the threads of one warp - integrate systems with similar
// properties. A warp runs at the speed of its stiffest member, so grouping like
// with like is what keeps the other 31 lanes from idling.
//
// The ordering is a bucket sort and runs entirely on the device: the keys never
// leave it. See balance() for the four passes.
//
// Follows the DeviceResources pattern: the object lives in device memory and a
// host side stub holds the same pointers. Note that the device object is
// placement-newed on top of a copy of the stub, so the constructor must leave
// every buffer pointer alone - initialising one here would overwrite the
// address the stub allocated.
class Balancer
{
protected:

    label   batchSize_;
    label   numOfBuckets_;

    // batchSize_ long
    scalar* keys_;
    label*  bucket_;
    label*  order_;

    // the range of the keys of the batch, as orderedBits()
    unsigned long long* keyMin_;
    unsigned long long* keyMax_;

    // numOfBuckets_ long: systems per bucket, then the next free slot of each
    label*  counts_;
    label*  cursor_;

public:

    __device__ __host__
    Balancer(const label batchSize)
        : batchSize_(batchSize), numOfBuckets_(KODES_BALANCER_BUCKETS) {}

    __device__ __host__
    virtual ~Balancer() = default;

    __device__ static void* operator new(size_t size, void* ptr) { return ptr; }

    // The property that decides where a system ends up in the batch
    __device__ virtual scalar
    key(const DeviceResources* resources, const label system) const = 0;

    __host__ __device__ scalar* keys() { return keys_; }

    __host__ __device__ label* bucket() { return bucket_; }

    __host__ __device__ label* order() { return order_; }

    __host__ __device__ const label* order() const { return order_; }

    __host__ __device__ unsigned long long* keyMin() { return keyMin_; }

    __host__ __device__ unsigned long long* keyMax() { return keyMax_; }

    __host__ __device__ label* counts() { return counts_; }

    __host__ __device__ label* cursor() { return cursor_; }

    __host__ __device__ label batchSize() const { return batchSize_; }

    __host__ __device__ label numOfBuckets() const { return numOfBuckets_; }

    // Sort the batch, on the device from start to finish. Called on the host
    // stub; `devBalancer` is its device side twin, on which the kernels
    // dispatch. Everything runs on the default stream, so the solve that
    // follows sees the finished order without a synchronisation.
    __host__ void balance
    (
        Balancer* devBalancer,
        DeviceResources* resources,
        const label realBatchSize,
        const LaunchConfig& config
    );

    // Device memory the balancer adds per system of the batch
    __host__ static size_t bytesPerSystem()
    {
        return sizeof(scalar) + 2 * sizeof(label);
    }

    __host__ void allocate(const label batchSize);

    __host__ void deallocate();
};

// Pass 1: the key of every system, and the range they span
__global__ void computeKeys
(
    Balancer* balancer,
    const DeviceResources* resources,
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
