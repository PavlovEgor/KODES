#include "basic_linalg.cuh"


__device__
void LUDecompose (scalar* matrix, label* pivotIndices, const label size)
{
    int sign;
    LUDecompose(matrix, pivotIndices, size, &sign);
}

__device__ 
void LUDecompose (scalar* matrix, label* pivotIndices, const label size, int* sign)
{
    // scalar* vv = (scalar*)malloc(size * sizeof(scalar));
    scalar vv[NSP];
    *sign = 1;

    for (label i = 0; i < size; ++i)
    {
        scalar largestCoeff = 0.0;
        scalar temp;

        for (label j = 0; j < size; ++j)
        {
            if ((temp = fabs(matrix[INDEX(i * size + j)])) > largestCoeff)
            {
                largestCoeff = temp;
            }
        }
        vv[i] = 1.0/largestCoeff;
    }

    for (label j = 0; j < size; ++j)
    {
        for (label i = 0; i < j; ++i)
        {
            scalar sum = matrix[INDEX(i*size + j)];
            for (label k = 0; k < i; ++k)
            {
                sum -= matrix[INDEX(i*size + k)]*matrix[INDEX(k*size + j)];
            }
            matrix[INDEX(i*size + j)] = sum;
        }

        label iMax = 0;

        scalar largestCoeff = 0.0;
        for (label i = j; i < size; ++i)
        {
            scalar sum = matrix[INDEX(i*size + j)];

            for (label k = 0; k < j; ++k)
            {
                sum -= matrix[INDEX(i*size + k)]*matrix[INDEX(k * size + j)];
            }

            matrix[INDEX(i * size + j)] = sum;

            scalar temp;
            if ((temp = vv[i]*fabs(sum)) >= largestCoeff)
            {
                largestCoeff = temp;
                iMax = i;
            }
        }

        pivotIndices[INDEX(j)] = iMax;

        if (j != iMax)
        {
            for (label k = 0; k < size; ++k)
            {
                swap(matrix[INDEX(j*size + k)], matrix[INDEX(iMax*size + k)]);
            }

            *sign *= -1;
            vv[iMax] = vv[j];
        }

        if (matrix[INDEX(j * size + j)] == 0.0)
        {
            matrix[INDEX(j * size + j)] = SMALL;
        }

        if (j != size-1)
        {
            scalar rDiag = 1.0/matrix[INDEX(j * size + j)];

            for (label i = j + 1; i < size; ++i)
            {
                matrix[INDEX(i*size + j)] *= rDiag;
            }
        }
    }
}

__device__  
void LUBacksubstitute (const scalar* luMatrix, const label* pivotIndices, scalar* source, const label size)
{
    label ii = 0;

    for (label i = 0; i < size; ++i)
    {
        label ip = pivotIndices[INDEX(i)];
        scalar sum = source[INDEX(ip)];
        source[INDEX(ip)] = source[INDEX(i)];

        if (ii != 0)
        {
            for (label j = ii - 1; j < i; ++j)
            {
                sum -= luMatrix[INDEX(i*size + j)]*source[INDEX(j)];
            }
        }
        else if (sum != 0.0)
        {
            ii = i + 1;
        }

        source[INDEX(i)] = sum;
    }

    for (int i = size - 1; i >= 0; --i)
    {
        scalar sum = source[INDEX(i)];

        for (label j = i + 1; j < size; ++j)
        {
            sum -= luMatrix[INDEX(i * size + j)]*source[INDEX(j)];
        }

        source[INDEX(i)] = sum/luMatrix[INDEX(i * size + i)];
    }
}