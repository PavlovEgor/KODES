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
    const scalar* y0, 
    const scalar* y, 
    const scalar* err, 
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

__device__
void LUDecompose (scalar* matrix, label* pivotIndices, const label size);

__device__
void LUDecompose (scalar* matrix, label* pivotIndices, const label size, int* sign);

__device__
void LUBacksubstitute (const scalar* luMatrix, const label* pivotIndices, scalar* source, const label size);


// --- shifted systems, gamma*I - J for a whole family of gamma ----------------
//
// An implicit ODE solver that changes its step size while holding the Jacobian
// fixed keeps solving with matrices that differ only by a multiple of the
// identity. There is no exact O(n^2) update of an LU factorisation under a full
// rank diagonal shift, Sherman-Morrison-Woodbury only covers low rank changes,
// but an *orthogonal similarity* is shift invariant:
//
//     J = Q H Q^T   =>   gamma*I - J = Q (gamma*I - H) Q^T
//
// so reducing J to upper Hessenberg form once, for (10/3)n^3, leaves every
// shifted system to be factorised in O(n^2) instead of O(n^3).

// Householder reduction of a general matrix to upper Hessenberg form, in place.
// On exit the upper triangle and the first subdiagonal hold H, the entries
// below the subdiagonal hold the Householder vectors generating Q (the leading
// one of each is implicit) and tau, of length size, holds their coefficients.
__device__
void hessenbergReduce (scalar* matrix, scalar* tau, const label size);

// Builds gamma*I - H from a reduced matrix and factorises it into factors with
// partial pivoting. An upper Hessenberg matrix has a single subdiagonal to
// eliminate per column, so this is O(n^2) and needs only one bit of pivoting
// information per column. factors is a full size*size work matrix, distinct
// from the reduced one, which stays untouched for the next shift.
__device__
void hessenbergShiftedFactorise
(
    const scalar* hessenberg,
    const scalar  gamma,
    scalar*       factors,
    label*        pivots,
    const label   size
);

// Solves (gamma*I - J) x = source for the shift the factors were built with,
// source overwritten by the solution:
//
//     z = Q^T source,   (gamma*I - H) w = z,   x = Q w
//
// The two orthogonal transforms are applied as sequences of Householder
// reflections read straight out of the reduced matrix, Q is never formed.
__device__
void hessenbergSolve
(
    const scalar* hessenberg,
    const scalar* tau,
    const scalar* factors,
    const label*  pivots,
    scalar*       source,
    const label   size
);
