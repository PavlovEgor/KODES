#ifndef KODES_THREAD_MATRIX
#define KODES_THREAD_MATRIX

#pragma once

#include <type_traits>

#include "basic_types.cuh"
#include "system_dimensions.cuh"

// A view, see ThreadVector.cuh: it owns nothing and cannot be assigned.
namespace kodes
{

// The calling thread's own N x N matrix in scratch space: element (i, j) is at
// INDEXMAT(i, j, N). N is a system unless said otherwise.
template <typename T, label N = kSystemSize>
struct ThreadMatrix
{
    static constexpr label size = N;

    T* data;

    ThreadMatrix() = default;

    ThreadMatrix(const ThreadMatrix&) = default;

    ThreadMatrix& operator=(const ThreadMatrix&) = delete;

    __host__ __device__ explicit ThreadMatrix(T* data)
        : data(data)
    {}

    template <typename U, typename = std::enable_if_t<std::is_same_v<const U, T>>>
    __host__ __device__ ThreadMatrix(const ThreadMatrix<U, N>& m)
        : data(m.data)
    {}

    __device__ T& operator()(const label i, const label j) const
    {
        return data[INDEXMAT(i, j, N)];
    }

    // N x N elements per resident thread
    __host__ void allocate(const label scratchSize)
    {
        CUDA_CHECK(cudaMalloc(&data, size_t(N) * N * scratchSize * sizeof(T)));
    }

    __host__ void deallocate()
    {
        CUDA_CHECK(cudaFree(data));
    }
};

}

#endif
