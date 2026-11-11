#ifndef KODES_DEVICE_ARRAY
#define KODES_DEVICE_ARRAY

#pragma once

#include "basic_types.cuh"
#include "DeviceBuffer.cuh"

namespace kodes
{

// A plain array of device memory, element i at i: one entry per something that
// is neither a resident thread nor a component of the batch, so neither
// INDEXVEC nor INDEXSTATE applies. The balancer's are of this kind - an entry
// per position of the batch, per bucket, per key. A view, see DeviceBuffer.cuh:
// it owns nothing and cannot be assigned.
template <typename T>
class DeviceArray
    :
    public DeviceBuffer<T>
{
    label size_;

public:

    DeviceArray() = default;

    DeviceArray(const DeviceArray&) = default;

    DeviceArray& operator=(const DeviceArray&) = delete;

    __device__ T& operator[](const label i) const
    {
        return this->data_[i];
    }

    __host__ __device__ label size() const
    {
        return size_;
    }

    __host__ void allocate(const label size)
    {
        size_ = size;
        this->allocateElements(size_t(size));
    }

    // Every byte of every element set to `byte`, as cudaMemset does: what a
    // pass that accumulates into the array starts from
    __host__ void setBytes(const int byte) const
    {
        CUDA_CHECK(cudaMemset(this->data_, byte, size_t(size_) * sizeof(T)));
    }
};

}

#endif
