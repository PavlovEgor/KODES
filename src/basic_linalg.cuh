#ifndef KODES_BASIC_LINALG
#define KODES_BASIC_LINALG

#pragma once

#include <cuda/cmath>
#include <cuda_runtime.h>
#include "basic_types.cuh"


typedef double scalar;
typedef int    label;

template <typename T>
__device__ inline
void copyVec(T* result, const T* source, const label size)
{
    for(label i=0; i<size; ++i)
    {
        result[INDEXVEC(i)] = source[INDEXVEC(i)];
    }
}

template <typename T>
__device__ inline
void sumVec(T* result, const T* source1, const T* source2, const label size)
{
    for(label i=0; i<size; ++i)
    {
        result[INDEXVEC(i)] = source1[INDEXVEC(i)] + source2[INDEXVEC(i)];
    }
}

template <typename T>
__device__ inline
T sqr(const T& x)
{
    return x * x;
}

template <typename T>
__device__ inline 
T clamp(const T& value, const T& minVal, const T& maxVal)
{
    return (value < minVal) ? minVal : (value > maxVal) ? maxVal : value;
}

template <typename T>
__device__ inline
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__device__ inline
scalar normalizeError (
    const scalar* __restrict__ y0, 
    const scalar* __restrict__ y, 
    const scalar* __restrict__ err, 
    const label sizeOfSystem, 
    const scalar absTol, 
    const scalar relTol)
{
    scalar maxErr = 0.0;
    for (label i=0; i < sizeOfSystem; ++i)
    {
        scalar tol = absTol + (relTol)*max(fabs(y0[INDEXVEC(i)]), fabs(y[INDEXVEC(i)]));
        maxErr = max(maxErr, fabs(err[INDEXVEC(i)])/tol);
    }

    return maxErr;
}

// In place LU factorization of the matrix a block shares, with scaled partial
// pivoting. Collective: every thread of the block has to call it, and it ends
// on a barrier, so the factors are visible to all of them when it returns.
//
// matrix and pivotIndices are the block's own slot, indexed with INDEXBLOCKMAT
// and directly; vv is `size` scalars of the same slot, the row scales of the
// implicit pivoting. It used to be a local array with a hard 128 entry bound,
// which a mechanism of more than 128 species would have run past in silence.
__device__
void blockLUDecompose
(
    scalar* matrix,
    label* pivotIndices,
    scalar* vv,
    const label size
);

// Solve for one right hand side against the factors above. Per thread, not
// collective: the block shares luMatrix and pivotIndices, but `source` is the
// calling thread's own vector, in its own scratch slot. This is where the
// sharing pays - every thread of the block reads the same matrix element at
// the same time, which is one broadcast instead of blockDim.x loads.
__device__
void LUBacksubstitute
(
    const scalar* luMatrix,
    const label* pivotIndices,
    scalar* source,
    const label size
);

#endif
