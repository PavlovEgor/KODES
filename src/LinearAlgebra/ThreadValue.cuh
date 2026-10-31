#ifndef KODES_THREAD_VALUE
#define KODES_THREAD_VALUE

#pragma once

#include "basic_types.cuh"
#include "DeviceBuffer.cuh"

namespace kodes
{

// One value per resident thread, the calling thread's at INDEXVEC(0). It
// stands for that value: it is read where a T is expected and written by
// assigning to it, and get() names it where a T& is wanted - printf()
// included, which takes no class. A view, see DeviceBuffer.cuh: copied like a
// pointer, assigned like a reference.
template <typename T>
class ThreadValue
    :
    public DeviceBuffer<T>
{
public:

    ThreadValue() = default;

    ThreadValue(const ThreadValue&) = default;

    __device__ T& get() const
    {
        return this->data_[INDEXVEC(0)];
    }

    __device__ operator T&() const
    {
        return get();
    }

    __device__ const ThreadValue& operator=(const T& value) const
    {
        get() = value;
        return *this;
    }

    __device__ const ThreadValue& operator=(const ThreadValue& v) const
    {
        get() = v.get();
        return *this;
    }

    __host__ void allocate(const label scratchSize)
    {
        this->allocateElements(size_t(scratchSize));
    }
};

}

#endif
