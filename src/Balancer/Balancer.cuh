#ifndef KODES_BALANCER
#define KODES_BALANCER

#pragma once

#include "basic_types.cuh"
#include "DeviceResources.cuh"
#include "LaunchConfig.cuh"

namespace kodes
{

// Orders the systems of a batch by a scalar key, so that neighbouring positions
// - and therefore the threads of one warp - integrate systems with similar
// properties. A warp runs at the speed of its stiffest member, so grouping like
// with like is what keeps the other 31 lanes from idling.
//
// Follows the DeviceResources pattern: the object lives in device memory and a
// host side stub holds the same pointers. keys_/order_ are the device arrays,
// hostKeys_/hostOrder_ their host mirrors, which the sort works on and which
// only ever get touched from the host.
class Balancer
{
protected:

    label   batchSize_;

    scalar* keys_;
    label*  order_;

    scalar* hostKeys_;
    label*  hostOrder_;

public:

    __device__ __host__
    Balancer(const label batchSize) : batchSize_(batchSize) {}

    __device__ __host__
    virtual ~Balancer() = default;

    __device__ static void* operator new(size_t size, void* ptr) { return ptr; }

    // The property that decides where a system ends up in the batch
    __device__ virtual scalar
    key(const DeviceResources* resources, const label system) const = 0;

    __host__ __device__ scalar* keys() { return keys_; }

    __host__ __device__ const label* order() const { return order_; }

    __host__ __device__ label batchSize() const { return batchSize_; }

    // Compute the keys on the device, sort them on the host and upload the
    // resulting order. Called on the host stub; `devBalancer` is its device
    // side twin, on which the key kernel dispatches.
    __host__ void balance
    (
        Balancer* devBalancer,
        DeviceResources* resources,
        const label realBatchSize,
        const LaunchConfig& config
    );

    // Device memory the balancer adds per system of the batch
    __host__ static size_t bytesPerSystem() { return sizeof(scalar) + sizeof(label); }

    __host__ void allocate(const label batchSize);

    __host__ void deallocate();
};

// In place quicksort of `keys`, applying every move to `order` as well
__host__ void quickSortByKey(scalar* keys, label* order, const label size);

}

#endif
