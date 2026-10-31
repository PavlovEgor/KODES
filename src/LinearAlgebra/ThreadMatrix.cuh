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
class ThreadMatrix
{
    T* data_;

public:

    static constexpr label size = N;

    ThreadMatrix() = default;

    ThreadMatrix(const ThreadMatrix&) = default;

    ThreadMatrix& operator=(const ThreadMatrix&) = delete;

    __host__ __device__ explicit ThreadMatrix(T* data)
        : data_(data)
    {}

    template <typename U, typename = std::enable_if_t<std::is_same_v<const U, T>>>
    __host__ __device__ ThreadMatrix(const ThreadMatrix<U, N>& m)
        : data_(m.data())
    {}

    // The bare buffer, for what takes one: pyJac in PyJacSystem
    __host__ __device__ T* data() const
    {
        return data_;
    }

    __device__ T& operator()(const label i, const label j) const
    {
        return data_[INDEXMAT(i, j, N)];
    }

    // N x N elements per resident thread
    __host__ void allocate(const label scratchSize)
    {
        CUDA_CHECK(cudaMalloc(&data_, size_t(N) * N * scratchSize * sizeof(T)));
    }

    __host__ void deallocate()
    {
        CUDA_CHECK(cudaFree(data_));
    }
};

}

#endif
