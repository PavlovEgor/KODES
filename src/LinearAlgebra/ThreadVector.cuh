#ifndef KODES_THREAD_VECTOR
#define KODES_THREAD_VECTOR

#pragma once

#include <type_traits>

#include "basic_types.cuh"
#include "system_dimensions.cuh"
#include "DeviceBuffer.cuh"

namespace kodes
{

template <typename T, label N> class ThreadVector;

// a + b, evaluated element by element when it is assigned to a vector
template <typename T, label N>
class ThreadVectorSum
{
    ThreadVector<const T, N> a_;
    ThreadVector<const T, N> b_;

public:

    __device__ ThreadVectorSum
    (
        const ThreadVector<const T, N> a,
        const ThreadVector<const T, N> b
    )
        : a_(a), b_(b)
    {}

    __device__ T operator[](const label i) const
    {
        return a_[i] + b_[i];
    }
};

// The calling thread's own vector of N elements in scratch space: element i is
// at INDEXVEC(i). N is a system unless said otherwise. A view, see
// DeviceBuffer.cuh: copied like a pointer, assigned like a reference.
template <typename T, label N = kSystemSize>
class ThreadVector
    :
    public DeviceBuffer<T>
{
public:

    using value_type = std::remove_const_t<T>;

    static constexpr label size = N;

    ThreadVector() = default;

    ThreadVector(const ThreadVector&) = default;

    __host__ __device__ explicit ThreadVector(T* data)
        : DeviceBuffer<T>(data)
    {}

    // a read only view of a vector that may be written
    template <typename U, typename = std::enable_if_t<std::is_same_v<const U, T>>>
    __host__ __device__ ThreadVector(const ThreadVector<U, N>& v)
        : DeviceBuffer<T>(v.data())
    {}

    __device__ T& operator[](const label i) const
    {
        return this->data_[INDEXVEC(i)];
    }

    __device__ const ThreadVector& operator=(const ThreadVector& v) const
    {
        return assign(v);
    }

    template <typename U>
    __device__ const ThreadVector& operator=(const ThreadVector<U, N>& v) const
    {
        return assign(v);
    }

    __device__ const ThreadVector& operator=(const ThreadVectorSum<value_type, N>& v) const
    {
        return assign(v);
    }

    template <typename U>
    __device__ const ThreadVector& operator+=(const ThreadVector<U, N>& v) const
    {
        #pragma unroll 4
        for (label i = 0; i < N; ++i)
        {
            (*this)[i] += v[i];
        }
        return *this;
    }

    // N elements per resident thread
    __host__ void allocate(const label scratchSize)
    {
        this->allocateElements(size_t(N) * scratchSize);
    }

private:

    template <typename V>
    __device__ const ThreadVector& assign(const V& v) const
    {
        #pragma unroll 4
        for (label i = 0; i < N; ++i)
        {
            (*this)[i] = v[i];
        }
        return *this;
    }
};

template <typename A, typename B, label N>
__device__ ThreadVectorSum<std::remove_const_t<A>, N>
operator+(const ThreadVector<A, N>& a, const ThreadVector<B, N>& b)
{
    static_assert
    (
        std::is_same_v<std::remove_const_t<A>, std::remove_const_t<B>>,
        "a sum of vectors of one type"
    );
    return {a, b};
}

}

#endif
