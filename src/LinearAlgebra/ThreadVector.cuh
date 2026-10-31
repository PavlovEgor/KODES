#ifndef KODES_THREAD_VECTOR
#define KODES_THREAD_VECTOR

#pragma once

#include <type_traits>

#include "basic_types.cuh"
#include "system_dimensions.cuh"

// Views of device memory, which every buffer of the library is held and
// passed as.
//
// A view owns nothing. Copying one, or passing it by value, gives another name
// for the same memory, as copying a pointer does. Assigning to one writes the
// elements, as assigning through a reference does: `y = yTemp` copies the
// vector, `y = yTemp + dy` sums into it, neither repoints y. A view there is
// nothing sensible to assign to - a matrix, the batch - cannot be assigned at
// all, so that nobody repoints one by accident either.
//
// A default constructed view is left alone, which is what the device objects
// need: their constructors run after the stub has been byte-copied to the
// device, and must not overwrite what the copy brought (see
// Factory/device_object.cuh). allocate() and deallocate() are called on that
// stub, on the host.
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
// at INDEXVEC(i). N is a system unless said otherwise.
template <typename T, label N = kSystemSize>
class ThreadVector
{
    T* data_;

public:

    using value_type = std::remove_const_t<T>;

    static constexpr label size = N;

    ThreadVector() = default;

    ThreadVector(const ThreadVector&) = default;

    __host__ __device__ explicit ThreadVector(T* data)
        : data_(data)
    {}

    // a read only view of a vector that may be written
    template <typename U, typename = std::enable_if_t<std::is_same_v<const U, T>>>
    __host__ __device__ ThreadVector(const ThreadVector<U, N>& v)
        : data_(v.data())
    {}

    // The bare buffer, for what takes one: pyJac in PyJacSystem
    __host__ __device__ T* data() const
    {
        return data_;
    }

    __device__ T& operator[](const label i) const
    {
        return data_[INDEXVEC(i)];
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
        CUDA_CHECK(cudaMalloc(&data_, size_t(N) * scratchSize * sizeof(T)));
    }

    __host__ void deallocate()
    {
        CUDA_CHECK(cudaFree(data_));
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
