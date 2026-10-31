#ifndef KODES_THREAD_VALUE
#define KODES_THREAD_VALUE

#pragma once

#include "basic_types.cuh"

// A view, see ThreadVector.cuh: copied like a pointer, assigned like a
// reference.
namespace kodes
{

// One value per resident thread, the calling thread's at INDEXVEC(0). It
// stands for that value: it is read where a T is expected and written by
// assigning to it, and get() names it where a T& is wanted - printf()
// included, which takes no class.
template <typename T>
class ThreadValue
{
    T* data_;

public:

    ThreadValue() = default;

    ThreadValue(const ThreadValue&) = default;

    __device__ T& get() const
    {
        return data_[INDEXVEC(0)];
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
        CUDA_CHECK(cudaMalloc(&data_, size_t(scratchSize) * sizeof(T)));
    }

    __host__ void deallocate()
    {
        CUDA_CHECK(cudaFree(data_));
    }
};

}

#endif
