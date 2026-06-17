#pragma once

#include <cuda/cmath>
#include <cuda_runtime.h>

#define SMALL 1e-9
#define GREAT 1e9

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
T& clamp(const T& value, const T& minVal, const T& maxVal)
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

__device__ inline
void LUDecompose (scalar* matrix, label* pivotIndices, const label size, int* sign)
{
    scalar* vv = (scalar*)malloc(size * sizeof(scalar));
    *sign = 1;

    for (label i = 0; i < size; ++i)
    {
        scalar largestCoeff = 0.0;
        scalar temp;
        const scalar* matrixi = (matrix + i * size);

        for (label j = 0; j < size; ++j)
        {
            if ((temp = fabs(matrixi[j])) > largestCoeff)
            {
                largestCoeff = temp;
            }
        }

        // if (largestCoeff == 0.0)
        // {
        //     FatalErrorInFunction
        //         << "Singular matrix" << exit(FatalError);
        // }

        vv[i] = 1.0/largestCoeff;
    }

    for (label j = 0; j < size; ++j)
    {
        scalar* matrixj = matrix + j * size;

        for (label i = 0; i < j; ++i)
        {
            scalar* matrixi = matrix + i * size;

            scalar sum = matrixi[j];
            for (label k = 0; k < i; ++k)
            {
                sum -= matrixi[k]*matrix[k* size + j];
            }
            matrixi[j] = sum;
        }

        label iMax = 0;

        scalar largestCoeff = 0.0;
        for (label i = j; i < size; ++i)
        {
            scalar* matrixi = matrix + i * size;
            scalar sum = matrixi[j];

            for (label k = 0; k < j; ++k)
            {
                sum -= matrixi[k]*matrix[k * size + j];
            }

            matrixi[j] = sum;

            scalar temp;
            if ((temp = vv[i]*fabs(sum)) >= largestCoeff)
            {
                largestCoeff = temp;
                iMax = i;
            }
        }

        pivotIndices[j] = iMax;

        if (j != iMax)
        {
            scalar*  matrixiMax = matrix + iMax * size;

            for (label k = 0; k < size; ++k)
            {
                swap(matrixj[k], matrixiMax[k]);
            }

            *sign *= -1;
            vv[iMax] = vv[j];
        }

        if (matrixj[j] == 0.0)
        {
            matrixj[j] = SMALL;
        }

        if (j != size-1)
        {
            scalar rDiag = 1.0/matrixj[j];

            for (label i = j + 1; i < size; ++i)
            {
                matrix[i*size + j] *= rDiag;
            }
        }
    }
}

__device__ inline 
void LUBacksubstitute (const scalar* luMatrix, const label* pivotIndices, scalar* source, const label size)
{
    label ii = 0;

    for (label i = 0; i < size; ++i)
    {
        label ip = pivotIndices[i];
        scalar sum = source[ip];
        source[ip] = source[i];
        const scalar* luMatrixi = luMatrix + i * size;

        if (ii != 0)
        {
            for (label j = ii - 1; j < i; ++j)
            {
                sum -= luMatrixi[j]*source[j];
            }
        }
        else if (sum != 0.0)
        {
            ii = i + 1;
        }

        source[i] = sum;
    }

    for (int i = size - 1; i >= 0; --i)
    {
        scalar sum = source[i];
        const scalar* luMatrixi = luMatrix + i * size;

        for (label j = i + 1; j < size; ++j)
        {
            sum -= luMatrixi[j]*source[j];
        }

        source[i] = sum/luMatrixi[i];
    }
}