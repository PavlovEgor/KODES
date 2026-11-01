#ifndef KODES_DIFFERENCE_TABLE
#define KODES_DIFFERENCE_TABLE

#pragma once

#include "basic_types.cuh"
#include "ThreadVector.cuh"
#include "ThreadMatrix.cuh"
#include "DeviceBuffer.cuh"
#include "bdf_constants.cuh"

namespace kodes
{

// Bdf's history of the calling thread, in scratch space: column k holds the
// k-th backward difference of the solution, scaled by the step to the power
// k, so that column 0 is the current state and the sum of the first order+1
// columns is the prediction of the next one. Two more columns than the
// highest order: the difference the step solves for, and the one above it
// that the choice of order looks at. Entry (i, k) is at
// INDEXMAT(i, k, kSystemSize).
//
// A view, see DeviceBuffer.cuh: it owns nothing and cannot be assigned; a
// column is a ThreadVector and can.
class DifferenceTable
    :
    public DeviceBuffer<scalar>
{
public:

    static constexpr label kColumns = kBdfMaxOrder + 3;

    static constexpr label size = kSystemSize;

    DifferenceTable() = default;

    DifferenceTable(const DifferenceTable&) = default;

    DifferenceTable& operator=(const DifferenceTable&) = delete;

    __device__ scalar& operator()(const label i, const label k) const
    {
        return data_[INDEXMAT(i, k, size)];
    }

    __device__ ThreadVector<scalar> column(const label k) const
    {
        return ThreadVector<scalar>(data_ + size_t(k) * size * GRID_DIM);
    }

    // The predicted state of the next step, the sum of the first order+1
    // columns, and psi, the part of the corrector the history gives
    __device__ void
    predict
    (
        const label order,
        const ThreadVector<scalar> yPredict,
        const ThreadVector<scalar> psi
    ) const
    {
        const scalar alpha = kBdfCoefficients.alpha[order];

        #pragma unroll 4
        for (label i=0; i<size; i++)
        {
            scalar y = (*this)(i, 0);
            scalar p = 0;

            for (label k=1; k<=order; k++)
            {
                const scalar dk = (*this)(i, k);

                y += dk;
                p += kBdfCoefficients.gamma[k]*dk;
            }

            yPredict[i] = y;
            psi[i] = p/alpha;
        }
    }

    // After an accepted step, whose correction to the prediction was d: d is
    // the difference of order order+1, the one above it is d less the
    // previous one, and the lower ones follow by D^j y_n = D^j y_{n-1} + D^{j+1} y_n
    __device__ void update(const label order, const ThreadVector<const scalar> d) const
    {
        #pragma unroll 4
        for (label i=0; i<size; i++)
        {
            const scalar di = d[i];

            (*this)(i, order + 2) = di - (*this)(i, order + 1);
            (*this)(i, order + 1) = di;

            for (label k=order; k>=0; k--)
            {
                (*this)(i, k) += (*this)(i, k + 1);
            }
        }
    }

    // Rescale the first order+1 columns to a step factor times the present
    // one: D <- (R U)^T D, with R the same matrix as kBdfCoefficients.u but
    // for the ratio factor instead of 1. ru is the thread's scratch for R U.
    __device__ void
    rescale
    (
        const label order,
        const scalar factor,
        const ThreadMatrix<scalar, kBdfTableSize> ru
    ) const
    {
        for (label k=0; k<=order; k++)
        {
            for (label j=0; j<=order; j++)
            {
                ru(k, j) = 0;
            }
        }

        // R a column m at a time: r = R[k][m] down the column. u is upper
        // triangular, so column j of R U sums over m <= j only.
        for (label m=0; m<=order; m++)
        {
            scalar r = 1;

            for (label k=0; k<=order; k++)
            {
                if (k > 0)
                {
                    r = r*(k - 1 - factor*m)/k;
                }

                for (label j=m; j<=order; j++)
                {
                    ru(k, j) += r*kBdfCoefficients.u[m][j];
                }
            }
        }

        #pragma unroll 4
        for (label i=0; i<size; i++)
        {
            // In registers: the loops are unrolled to the table size, and the
            // orders above order skipped
            scalar column[kBdfTableSize];

            #pragma unroll
            for (label j=0; j<kBdfTableSize; j++)
            {
                column[j] = 0;

                #pragma unroll
                for (label k=0; k<kBdfTableSize; k++)
                {
                    if (j <= order && k <= order)
                    {
                        column[j] += ru(k, j)*(*this)(i, k);
                    }
                }
            }

            #pragma unroll
            for (label j=0; j<kBdfTableSize; j++)
            {
                if (j <= order)
                {
                    (*this)(i, j) = column[j];
                }
            }
        }
    }

    __host__ void allocate(const label scratchSize)
    {
        allocateElements(size_t(kColumns) * size * scratchSize);
    }
};

}

#endif
