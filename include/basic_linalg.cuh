#pragma once

#include <cuda/cmath>
#include <cuda_runtime.h>

#define SMALL 1e-9
#define GREAT 1e9
#define MAX_VEC_SIZE 128

typedef double scalar;
typedef int    label;

template <typename T>
__device__ inline
void copyVec(T* result, const T* source, const label size)
{
    for(label i=0; i<size; ++i)
    {
        result[i] = source[i];
    }
}

template <typename T>
__device__ inline
void sumVec(T* result, const T* source1, const T* source2, const label size)
{
    for(label i=0; i<size; ++i)
    {
        result[i] = source1[i] + source2[i];
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
    const label* sizeOfSystem, 
    const scalar* absTol, 
    const scalar* relTol)
{
    scalar maxErr = 0.0;
    for (label i=0; i < *sizeOfSystem; ++i)
    {
        scalar tol = *absTol + (*relTol)*max(fabs(y0[i]), fabs(y[i]));
        maxErr = max(maxErr, fabs(err[i])/tol);
    }

    return maxErr;
}

__device__
void LUDecompose (scalar* matrix, label* pivotIndices, const label size);

__device__ 
void LUDecompose (scalar* matrix, label* pivotIndices, const label size, int* sign);

__device__  
void LUBacksubstitute (const scalar* luMatrix, const label* pivotIndices, scalar* source, const label size);
