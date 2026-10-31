#ifndef KODES_EXTRAPOLATION_TABLE
#define KODES_EXTRAPOLATION_TABLE

#pragma once

#include "basic_types.cuh"
#include "ThreadVector.cuh"
#include "seulex_constants.cuh"

namespace kodes
{

// Seulex's extrapolation table of the calling thread, in scratch space: column
// k-1 holds the state the k-th sub-stepping sequence ended with, so there are
// kSeulexMaxOrder columns of `size` entries. Entry (i, k) is at
// INDEXMAT(i, k, size).
//
// A view, like those of basic_types.cuh: it owns nothing and cannot be
// assigned; a column is a ThreadVector and can.
struct ExtrapolationTable
{
    static constexpr label kColumns = kSeulexMaxOrder;

    scalar* data;
    label size;

    ExtrapolationTable() = default;

    ExtrapolationTable(const ExtrapolationTable&) = default;

    ExtrapolationTable& operator=(const ExtrapolationTable&) = delete;

    __device__ scalar& operator()(const label i, const label k) const
    {
        return data[INDEXMAT(i, k, size)];
    }

    __device__ ThreadVector<scalar> column(const label k) const
    {
        return {data + size_t(k) * size * GRID_DIM, size};
    }

    // Extrapolate the first k columns down into column 0, and from there into
    // y, which holds the state of the first sequence
    __device__ void extrapolate(const label k, const ThreadVector<scalar> y) const
    {
        for (label j = k - 1; j > 0; j--)
        {
            const scalar coeff = kSeulexExtrapolationCoeff[k + j*kSeulexTableSize];

            for (label i = 0; i < size; i++)
            {
                (*this)(i, j-1) = (*this)(i, j) + coeff*((*this)(i, j) - (*this)(i, j-1));
            }
        }

        for (label i = 0; i < size; i++)
        {
            y[i] = (*this)(i, 0) + kSeulexExtrapolationCoeff[k]*((*this)(i, 0) - y[i]);
        }
    }

    __host__ void allocate(const label n, const label scratchSize)
    {
        size = n;
        CUDA_CHECK(cudaMalloc(&data, size_t(kColumns) * n * scratchSize * sizeof(scalar)));
    }

    __host__ void deallocate()
    {
        CUDA_CHECK(cudaFree(data));
    }
};

}

#endif
