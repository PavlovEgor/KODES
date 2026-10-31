#ifndef KODES_THREAD_MATRIX
#define KODES_THREAD_MATRIX

#pragma once

#include <type_traits>

#include "basic_types.cuh"

// A view, see ThreadVector.cuh: it owns nothing and cannot be assigned.
namespace kodes
{

// The calling thread's own size x size matrix in scratch space: element
// (i, j) is at INDEXMAT(i, j, size)
template <typename T>
struct ThreadMatrix
{
    T* data;
    label size;

    ThreadMatrix() = default;

    ThreadMatrix(const ThreadMatrix&) = default;

    ThreadMatrix& operator=(const ThreadMatrix&) = delete;

    __host__ __device__ ThreadMatrix(T* data, const label size)
        : data(data), size(size)
    {}

    template <typename U, typename = std::enable_if_t<std::is_same_v<const U, T>>>
    __host__ __device__ ThreadMatrix(const ThreadMatrix<U>& m)
        : data(m.data), size(m.size)
    {}

    __device__ T& operator()(const label i, const label j) const
    {
        return data[INDEXMAT(i, j, size)];
    }

    // `n` x `n` elements per resident thread
    __host__ void allocate(const label n, const label scratchSize)
    {
        size = n;
        CUDA_CHECK(cudaMalloc(&data, size_t(n) * n * scratchSize * sizeof(T)));
    }

    __host__ void deallocate()
    {
        CUDA_CHECK(cudaFree(data));
    }
};

}

#endif
