#ifndef KODES_BATCH_ARRAY
#define KODES_BATCH_ARRAY

#pragma once

#include "basic_types.cuh"
#include "system_dimensions.cuh"
#include "DeviceBuffer.cuh"

namespace kodes
{

// The whole batch in state space, N components of batchSize systems:
// component i of a system is at INDEXSTATE(system, i, batchSize_). N is fixed
// when the library is built, the batch size when the plan is made. A view,
// see DeviceBuffer.cuh: it owns nothing and cannot be assigned.
template <typename T, label N>
class BatchArray
    :
    public DeviceBuffer<T>
{
    label batchSize_;

public:

    static constexpr label size = N;

    BatchArray() = default;

    BatchArray(const BatchArray&) = default;

    BatchArray& operator=(const BatchArray&) = delete;

    __device__ T& operator()(const label system, const label i) const
    {
        return this->data_[INDEXSTATE(system, i, batchSize_)];
    }

    // Where component i of the whole batch starts, for the host to copy a
    // batch in and out with cudaMemcpy
    __host__ T* component(const label i) const
    {
        return this->data_ + size_t(i) * batchSize_;
    }

    __host__ void allocate(const label systems)
    {
        batchSize_ = systems;
        this->allocateElements(size_t(N) * systems);
    }
};

}

#endif
