#ifndef KODES_BASIC_LINALG
#define KODES_BASIC_LINALG

#pragma once

#include <cuda/cmath>
#include <cuda_runtime.h>
#include "basic_types.cuh"
#include "ThreadVector.cuh"
#include "ThreadMatrix.cuh"

namespace kodes
{

template <typename T>
__device__ inline
T sqr(const T& x)
{
    return x * x;
}

__device__ inline
scalar normalizeError
(
    const ThreadVector<const scalar> y0,
    const ThreadVector<const scalar> y,
    const ThreadVector<const scalar> err,
    const scalar absTol,
    const scalar relTol
)
{
    scalar maxErr = 0.0;
    for (label i=0; i < y.size; ++i)
    {
        scalar tol = absTol + (relTol)*max(fabs(y0[i]), fabs(y[i]));
        maxErr = max(maxErr, fabs(err[i])/tol);
    }

    return maxErr;
}

// In place LU factorization of the calling thread's own matrix, with scaled
// partial pivoting - Crout's form, as OpenFOAM's LUDecompose. matrix and
// pivotIndices are the thread's scratch slot; so is vv, `matrix.size` scalars
// for the row scales of the implicit pivoting. It
// used to be a local array with a hard 128 entry bound, which a mechanism of
// more than 128 species ran past in silence.
__device__
void LUDecompose
(
    const ThreadMatrix<scalar> matrix,
    const ThreadVector<label> pivotIndices,
    const ThreadVector<scalar> vv
);

// Solve for one right hand side against the factors above, in place
__device__
void LUBacksubstitute
(
    const ThreadMatrix<const scalar> luMatrix,
    const ThreadVector<const label> pivotIndices,
    const ThreadVector<scalar> source
);

}

#endif
