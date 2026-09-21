#include "basic_linalg.cuh"
#include "block_reduce.cuh"


__device__
void blockLUDecompose
(
    scalar* __restrict__ matrix,
    label* __restrict__ pivotIndices,
    scalar* __restrict__ vv,
    const label size
)
{
    const label lane = threadIdx.x;
    const label lanes = blockDim.x;

    // Row scales of the implicit pivoting: one over the largest magnitude in
    // each row, so that the pivot is chosen by relative rather than absolute
    // size and a row that is merely badly scaled does not win every column.
    for (label i = lane; i < size; i += lanes)
    {
        scalar largestCoeff = 0.0;

        for (label j = 0; j < size; ++j)
        {
            const scalar candidate = fabs(matrix[INDEXBLOCKMAT(i, j, size)]);

            if (candidate > largestCoeff)
            {
                largestCoeff = candidate;
            }
        }

        if (largestCoeff == 0.0)
        {
            printf("Singular matrix\n");
            largestCoeff = SMALL;
        }

        vv[i] = 1.0/largestCoeff;
    }

    __syncthreads();

    // Right looking Gaussian elimination. The Crout form this replaced walks
    // each column from its top, every entry needing the one above it, and so
    // has nothing for a second thread to do; the right looking form spends its
    // work in a rank one update of the trailing submatrix, which is (size-k)^2
    // independent entries.
    for (label k = 0; k < size; ++k)
    {
        scalar best = -1.0;
        label  pivotRow = size;

        for (label i = k + lane; i < size; i += lanes)
        {
            const scalar candidate = vv[i]*fabs(matrix[INDEXBLOCKMAT(i, k, size)]);

            if (candidate > best)
            {
                best = candidate;
                pivotRow = i;
            }
        }

        // Ties go to the first row rather than the last, which is where this
        // differs from the serial version it replaced. Either is a valid
        // pivot; fixing one of them keeps the factorization reproducible.
        kodes::blockArgMax(best, pivotRow);

        if (lane == 0)
        {
            pivotIndices[k] = pivotRow;
        }

        if (pivotRow != k)
        {
            for (label j = lane; j < size; j += lanes)
            {
                swap
                (
                    matrix[INDEXBLOCKMAT(k, j, size)],
                    matrix[INDEXBLOCKMAT(pivotRow, j, size)]
                );
            }

            if (lane == 0)
            {
                swap(vv[k], vv[pivotRow]);
            }
        }

        __syncthreads();

        if (lane == 0 && matrix[INDEXBLOCKMAT(k, k, size)] == 0.0)
        {
            matrix[INDEXBLOCKMAT(k, k, size)] = SMALL;
        }

        __syncthreads();

        const scalar rDiag = 1.0/matrix[INDEXBLOCKMAT(k, k, size)];

        for (label i = k + 1 + lane; i < size; i += lanes)
        {
            matrix[INDEXBLOCKMAT(i, k, size)] *= rDiag;
        }

        __syncthreads();

        // Flattened, so that the threads share the trailing submatrix evenly
        // however narrow it has become by the last columns.
        const label trailing = size - k - 1;

        for (label t = lane; t < trailing*trailing; t += lanes)
        {
            const label i = k + 1 + t/trailing;
            const label j = k + 1 + t%trailing;

            matrix[INDEXBLOCKMAT(i, j, size)] -=
                matrix[INDEXBLOCKMAT(i, k, size)]*matrix[INDEXBLOCKMAT(k, j, size)];
        }

        __syncthreads();
    }
}

__device__
void LUBacksubstitute
(
    const scalar* __restrict__ luMatrix,
    const label* __restrict__ pivotIndices,
    scalar* __restrict__ source,
    const label size
)
{
    label ii = 0;

    for (label i = 0; i < size; ++i)
    {
        label ip = pivotIndices[i];
        scalar sum = source[INDEXVEC(ip)];
        source[INDEXVEC(ip)] = source[INDEXVEC(i)];

        if (ii != 0)
        {
            for (label j = ii - 1; j < i; ++j)
            {
                sum -= luMatrix[INDEXBLOCKMAT(i, j, size)]*source[INDEXVEC(j)];
            }
        }
        else if (sum != 0.0)
        {
            ii = i + 1;
        }

        source[INDEXVEC(i)] = sum;
    }

    for (label i = size - 1; i >= 0; --i)
    {
        scalar sum = source[INDEXVEC(i)];

        for (label j = i + 1; j < size; ++j)
        {
            sum -= luMatrix[INDEXBLOCKMAT(i, j, size)]*source[INDEXVEC(j)];
        }

        source[INDEXVEC(i)] = sum/luMatrix[INDEXBLOCKMAT(i, i, size)];
    }
}
