#ifndef KODES_EXTRAPOLATION_TABLE
#define KODES_EXTRAPOLATION_TABLE

#pragma once

#include "basic_types.cuh"
#include "ThreadVector.cuh"
#include "DeviceBuffer.cuh"
#include "seulex_constants.cuh"

namespace kodes
{

// Seulex's extrapolation table of the calling thread, in scratch space: column
// k-1 holds the state the k-th sub-stepping sequence ended with, so there are
// kSeulexMaxOrder columns of a system each. Entry (i, k) is at
// INDEXMAT(i, k, kSystemSize).
//
// A view, see DeviceBuffer.cuh: it owns nothing and cannot be assigned; a
// column is a ThreadVector and can.
class ExtrapolationTable
    :
    public DeviceBuffer<scalar>
{
public:

    static constexpr label kColumns = kSeulexMaxOrder;

    static constexpr label size = kSystemSize;

    ExtrapolationTable() = default;

    ExtrapolationTable(const ExtrapolationTable&) = default;

    ExtrapolationTable& operator=(const ExtrapolationTable&) = delete;

    __device__ scalar& operator()(const label i, const label k) const
    {
        return data_[INDEXMAT(i, k, size)];
    }

    __device__ ThreadVector<scalar> column(const label k) const
    {
        return ThreadVector<scalar>(data_ + size_t(k) * size * GRID_DIM);
    }

    // Extrapolate the first k columns down into column 0, and from there into
    // y, which holds the state of the first sequence
    __device__ void extrapolate(const label k, const ThreadVector<scalar> y) const
    {
        #pragma unroll 4
        for (label j = k - 1; j > 0; j--)
        {
            const scalar coeff = kSeulexExtrapolationCoeff[k + j*kSeulexTableSize];

            #pragma unroll 4
            for (label i = 0; i < size; i++)
            {
                (*this)(i, j-1) = (*this)(i, j) + coeff*((*this)(i, j) - (*this)(i, j-1));
            }
        }

        #pragma unroll 4
        for (label i = 0; i < size; i++)
        {
            y[i] = (*this)(i, 0) + kSeulexExtrapolationCoeff[k]*((*this)(i, 0) - y[i]);
        }
    }

    __host__ void allocate(const label scratchSize)
    {
        allocateElements(size_t(kColumns) * size * scratchSize);
    }
};

}

#endif
