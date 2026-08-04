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
void zeroVec(T* result, const label size)
{
    for(label i=0; i<size; ++i)
    {
        result[INDEXVEC(i)] = T(0);
    }
}

template <typename T>
__device__ inline
T sqr(const T& x)
{
    return x * x;
}

__device__ inline
scalar dotProduct(const scalar* a, const scalar* b, const label size)
{
    scalar sum = 0;
    for(label i=0; i<size; ++i)
    {
        sum += a[INDEXVEC(i)]*b[INDEXVEC(i)];
    }

    return sum;
}

// Root mean square of v measured against the per component scale, the same
// norm the seulex step controller uses for its error estimate
__device__ inline
scalar scaledNorm(const scalar* v, const scalar* scale, const label size)
{
    scalar sum = 0;
    for(label i=0; i<size; ++i)
    {
        sum += sqr(v[INDEXVEC(i)]/scale[INDEXVEC(i)]);
    }

    return sqrt(sum/size);
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

// Number of scratch vectors of length sizeOfSystem that shiftedBiCGStab needs
#define BICGSTAB_WORK_VECTORS 7

// Solves (M + eta*I) x = b for a matrix M that is only known through its LU
// factorisation, i.e. a system that differs from an already factorised one by
// a multiple of the identity. Bi-CGStab is used on the left preconditioned
// system
//
//     (I + eta*M^-1) x = M^-1 b
//
// Because the two matrices differ by a *scalar* shift the preconditioned
// operator contains no reference to M itself, only to its inverse: one
// application costs a single back substitution and no matrix vector product at
// all. The iteration therefore never touches an O(n^2) matrix apart from the
// triangular factors, and no shifted matrix has to be assembled.
//
// The spectrum of the preconditioned operator is 1 + eta/(gammaRef - lambda)
// over the eigenvalues lambda of the Jacobian. For a stiff chemical system
// almost all lambda are far from the shift, so the spectrum is tightly
// clustered around one and only the few slow modes are spread out, which is
// what makes the iteration converge in a handful of steps.
//
// x       right hand side b on entry, solution on exit, the iteration always
//         starts from a zero iterate
// scale   per component scale of the convergence test, tol is an rms tolerance
//         on the error of x measured against it
// work    BICGSTAB_WORK_VECTORS vectors of length size, laid out as the seulex
//         extrapolation table is
//
// Returns the number of iterations taken, or -1 if maxIter was exhausted or
// the iteration broke down, in which case x is restored to the right hand side
// it was called with so that the caller can fall back to a direct solve.
__device__
label shiftedBiCGStab
(
    const scalar* luMatrix,
    const label*  pivotIndices,
    const scalar  eta,
    scalar*       x,
    const scalar* scale,
    scalar*       work,
    const label   size,
    const scalar  tol,
    const label   maxIter
);
