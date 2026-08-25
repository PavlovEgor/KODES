#include "basic_linalg.cuh"


__device__
void LUDecompose (scalar* __restrict__ matrix, label* __restrict__ pivotIndices, const label size)
{
    int sign;
    LUDecompose(matrix, pivotIndices, size, &sign);
}

__device__ 
void LUDecompose (scalar* __restrict__ matrix, label* __restrict__ pivotIndices, const label size, int* sign)
{
    scalar vv[128];
    *sign = 1;

    for (label i = 0; i < size; ++i)
    {
        scalar largestCoeff = 0.0;
        scalar temp;

        for (label j = 0; j < size; ++j)
        {
            if ((temp = fabs(matrix[INDEXMAT(i, j, size)])) > largestCoeff)
            {
                largestCoeff = temp;
            }
        }
        if (largestCoeff == 0.0)
        {
            printf("Singular matrix");
        }
        vv[i] = 1.0/largestCoeff;
    }

    for (label j = 0; j < size; ++j)
    {
        for (label i = 0; i < j; ++i)
        {
            scalar sum = matrix[INDEXMAT(i, j, size)];
            for (label k = 0; k < i; ++k)
            {
                sum -= matrix[INDEXMAT(i, k, size)]*matrix[INDEXMAT(k, j, size)];
            }
            matrix[INDEXMAT(i, j, size)] = sum;
        }

        label iMax = j;

        scalar largestCoeff = 0.0;
        for (label i = j; i < size; ++i)
        {
            scalar sum = matrix[INDEXMAT(i, j, size)];

            for (label k = 0; k < j; ++k)
            {
                sum -= matrix[INDEXMAT(i, k, size)]*matrix[INDEXMAT(k, j, size)];
            }

            matrix[INDEXMAT(i, j, size)] = sum;

            scalar temp;
            if ((temp = vv[i]*fabs(sum)) >= largestCoeff)
            {
                largestCoeff = temp;
                iMax = i;
            }
        }

        pivotIndices[INDEXVEC(j)] = iMax;

        if (j != iMax)
        {
            for (label k = 0; k < size; ++k)
            {
                swap(matrix[INDEXMAT(j, k, size)], matrix[INDEXMAT(iMax, k, size)]);
            }

            *sign *= -1;
            vv[iMax] = vv[j];
        }

        if (matrix[INDEXMAT(j, j, size)] == 0.0)
        {
            matrix[INDEXMAT(j, j, size)] = SMALL;
        }

        if (j != size-1)
        {
            scalar rDiag = 1.0/matrix[INDEXMAT(j, j, size)];

            for (label i = j + 1; i < size; ++i)
            {
                matrix[INDEXMAT(i, j, size)] *= rDiag;
            }
        }
    }
}

__device__  
void LUBacksubstitute (const scalar* __restrict__ luMatrix, const label* __restrict__ pivotIndices, scalar* __restrict__ source, const label size)
{
    label ii = 0;

    for (label i = 0; i < size; ++i)
    {
        label ip = pivotIndices[INDEXVEC(i)];
        scalar sum = source[INDEXVEC(ip)];
        source[INDEXVEC(ip)] = source[INDEXVEC(i)];

        if (ii != 0)
        {
            for (label j = ii - 1; j < i; ++j)
            {
                sum -= luMatrix[INDEXMAT(i, j, size)]*source[INDEXVEC(j)];
            }
        }
        else if (sum != 0.0)
        {
            ii = i + 1;
        }

        source[INDEXVEC(i)] = sum;
    }

    for (int i = size - 1; i >= 0; --i)
    {
        scalar sum = source[INDEXVEC(i)];

        for (label j = i + 1; j < size; ++j)
        {
            sum -= luMatrix[INDEXMAT(i, j, size)]*source[INDEXVEC(j)];
        }

        source[INDEXVEC(i)] = sum/luMatrix[INDEXMAT(i, i, size)];
    }
}
