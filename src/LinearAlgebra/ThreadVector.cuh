#ifndef KODES_THREAD_VECTOR
#define KODES_THREAD_VECTOR

#pragma once

#include <type_traits>

#include "basic_types.cuh"

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

template <typename T> struct ThreadVector;

// a + b, evaluated element by element when it is assigned to a vector
template <typename T>
struct ThreadVectorSum
{
    ThreadVector<const T> a;
    ThreadVector<const T> b;

    __device__ T operator[](const label i) const
    {
        return a[i] + b[i];
    }
};

// The calling thread's own vector in scratch space: element i is at
// INDEXVEC(i)
template <typename T>
struct ThreadVector
{
    using value_type = std::remove_const_t<T>;

    T* data;
    label size;

    ThreadVector() = default;

    ThreadVector(const ThreadVector&) = default;

    __host__ __device__ ThreadVector(T* data, const label size)
        : data(data), size(size)
    {}

    // a read only view of a vector that may be written
    template <typename U, typename = std::enable_if_t<std::is_same_v<const U, T>>>
    __host__ __device__ ThreadVector(const ThreadVector<U>& v)
        : data(v.data), size(v.size)
    {}

    __device__ T& operator[](const label i) const
    {
        return data[INDEXVEC(i)];
    }

    __device__ const ThreadVector& operator=(const ThreadVector& v) const
    {
        return assign(v);
    }

    template <typename U>
    __device__ const ThreadVector& operator=(const ThreadVector<U>& v) const
    {
        return assign(v);
    }

    __device__ const ThreadVector& operator=(const ThreadVectorSum<value_type>& v) const
    {
        return assign(v);
    }

    template <typename U>
    __device__ const ThreadVector& operator+=(const ThreadVector<U>& v) const
    {
        for (label i = 0; i < size; ++i)
        {
            (*this)[i] += v[i];
        }
        return *this;
    }

    // `n` elements per resident thread
    __host__ void allocate(const label n, const label scratchSize)
    {
        size = n;
        CUDA_CHECK(cudaMalloc(&data, size_t(n) * scratchSize * sizeof(T)));
    }

    __host__ void deallocate()
    {
        CUDA_CHECK(cudaFree(data));
    }

private:

    template <typename V>
    __device__ const ThreadVector& assign(const V& v) const
    {
        for (label i = 0; i < size; ++i)
        {
            (*this)[i] = v[i];
        }
        return *this;
    }
};

template <typename A, typename B>
__device__ ThreadVectorSum<std::remove_const_t<A>>
operator+(const ThreadVector<A>& a, const ThreadVector<B>& b)
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
