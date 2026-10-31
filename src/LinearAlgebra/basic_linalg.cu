#include <cuda/std/utility>
#include "basic_linalg.cuh"

__device__
void kodes::LUDecompose
(
    const ThreadMatrix<scalar> matrix,
    const ThreadVector<label> pivotIndices,
    const ThreadVector<scalar> vv
)
{
    const label size = matrix.size;

    for (label i = 0; i < size; ++i)
    {
        scalar largestCoeff = 0.0;
        scalar temp;

        for (label j = 0; j < size; ++j)
        {
            if ((temp = fabs(matrix(i, j))) > largestCoeff)
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
            scalar sum = matrix(i, j);
            for (label k = 0; k < i; ++k)
            {
                sum -= matrix(i, k)*matrix(k, j);
            }
            matrix(i, j) = sum;
        }

        label iMax = j;

        scalar largestCoeff = 0.0;
        for (label i = j; i < size; ++i)
        {
            scalar sum = matrix(i, j);

            for (label k = 0; k < j; ++k)
            {
                sum -= matrix(i, k)*matrix(k, j);
            }

            matrix(i, j) = sum;

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
            for (label k = 0; k < size; ++k)
            {
                cuda::std::swap(matrix(j, k), matrix(iMax, k));
            }

            vv[iMax] = vv[j];
        }

        if (matrix(j, j) == 0.0)
        {
            matrix(j, j) = SMALL;
        }

        if (j != size-1)
        {
            scalar rDiag = 1.0/matrix(j, j);

            for (label i = j + 1; i < size; ++i)
            {
                matrix(i, j) *= rDiag;
            }
        }
    }
}

__device__
void kodes::LUBacksubstitute
(
    const ThreadMatrix<const scalar> luMatrix,
    const ThreadVector<const label> pivotIndices,
    const ThreadVector<scalar> source
)
{
    const label size = luMatrix.size;

    label ii = 0;

    for (label i = 0; i < size; ++i)
    {
        label ip = pivotIndices[i];
        scalar sum = source[ip];
        source[ip] = source[i];

        if (ii != 0)
        {
            for (label j = ii - 1; j < i; ++j)
            {
                sum -= luMatrix(i, j)*source[j];
            }
        }
        else if (sum != 0.0)
        {
            ii = i + 1;
        }

        source[i] = sum;
    }

    for (label i = size - 1; i >= 0; --i)
    {
        scalar sum = source[i];

        for (label j = i + 1; j < size; ++j)
        {
            sum -= luMatrix(i, j)*source[j];
        }

        source[i] = sum/luMatrix(i, i);
    }
}
