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


__device__
void hessenbergReduce (scalar* __restrict__ a, scalar* __restrict__ tau, const label size)
{
    for (label k = 0; k < size - 2; ++k)
    {
        // Householder reflection mapping the k-th column below the diagonal
        // onto a multiple of the first unit vector. Following LAPACK, the
        // reflection is I - tau*u*u^T with u[0] = 1, so only u[1..] has to be
        // stored and it fits exactly where the zeros it creates would go
        const scalar alpha = a[INDEXMAT(k+1, k, size)];

        scalar xNorm2 = 0;
        for (label i = k+2; i < size; ++i)
        {
            xNorm2 += sqr(a[INDEXMAT(i, k, size)]);
        }

        if (xNorm2 == 0)
        {
            // already in Hessenberg form in this column
            tau[INDEXVEC(k)] = 0;
            continue;
        }

        const scalar beta = -copysign(sqrt(alpha*alpha + xNorm2), alpha);
        const scalar tauK = (beta - alpha)/beta;
        const scalar rScale = 1/(alpha - beta);

        for (label i = k+2; i < size; ++i)
        {
            a[INDEXMAT(i, k, size)] *= rScale;
        }

        a[INDEXMAT(k+1, k, size)] = beta;
        tau[INDEXVEC(k)] = tauK;

        // From the left, over the columns right of k. The columns left of k
        // hold the reflections of the previous steps and, as far as H is
        // concerned, zeros that the reflection would leave zero anyway
        for (label j = k+1; j < size; ++j)
        {
            scalar s = a[INDEXMAT(k+1, j, size)];
            for (label i = k+2; i < size; ++i)
            {
                s += a[INDEXMAT(i, k, size)]*a[INDEXMAT(i, j, size)];
            }
            s *= tauK;

            a[INDEXMAT(k+1, j, size)] -= s;
            for (label i = k+2; i < size; ++i)
            {
                a[INDEXMAT(i, j, size)] -= s*a[INDEXMAT(i, k, size)];
            }
        }

        // From the right, over every row. This is what keeps the similarity,
        // and it leaves column k, where the reflection itself lives, alone
        for (label i = 0; i < size; ++i)
        {
            scalar s = a[INDEXMAT(i, k+1, size)];
            for (label j = k+2; j < size; ++j)
            {
                s += a[INDEXMAT(i, j, size)]*a[INDEXMAT(j, k, size)];
            }
            s *= tauK;

            a[INDEXMAT(i, k+1, size)] -= s;
            for (label j = k+2; j < size; ++j)
            {
                a[INDEXMAT(i, j, size)] -= s*a[INDEXMAT(j, k, size)];
            }
        }
    }

    // The last two columns carry no reflection, kept at zero so that a solve
    // can sweep the whole range without a special case
    for (label k = (size > 2 ? size - 2 : 0); k < size; ++k)
    {
        tau[INDEXVEC(k)] = 0;
    }
}


// v <- (I - tau*u*u^T) v, the reflection stored in column k of the reduced
// matrix acting on the components below k
__device__ static inline
void applyReflector
(
    const scalar* __restrict__ hessenberg,
    const scalar tau,
    const label k,
    scalar* __restrict__ v,
    const label size
)
{
    if (tau == 0)
    {
        return;
    }

    scalar s = v[INDEXVEC(k+1)];
    for (label i = k+2; i < size; ++i)
    {
        s += hessenberg[INDEXMAT(i, k, size)]*v[INDEXVEC(i)];
    }
    s *= tau;

    v[INDEXVEC(k+1)] -= s;
    for (label i = k+2; i < size; ++i)
    {
        v[INDEXVEC(i)] -= s*hessenberg[INDEXMAT(i, k, size)];
    }
}


__device__
void hessenbergShiftedFactorise
(
    const scalar* __restrict__ hessenberg,
    const scalar gamma,
    scalar* __restrict__ factors,
    label* __restrict__ pivots,
    const label size
)
{
    // gamma*I - H, only the Hessenberg part of the source is read and only that
    // part of the destination is ever written or looked at again
    for (label j = 0; j < size; ++j)
    {
        const label iEnd = min(j + 1, size - 1);

        for (label i = 0; i <= iEnd; ++i)
        {
            factors[INDEXMAT(i, j, size)] = -hessenberg[INDEXMAT(i, j, size)];
        }

        factors[INDEXMAT(j, j, size)] += gamma;
    }

    // Gaussian elimination, one subdiagonal entry per column. The interchange
    // can only ever be between the current row and the next one, so a single
    // flag records it, and the multiplier of the previous column sits to the
    // left of the range being swapped and stays with its own step
    for (label k = 0; k < size - 1; ++k)
    {
        if
        (
            fabs(factors[INDEXMAT(k+1, k, size)])
          > fabs(factors[INDEXMAT(k, k, size)])
        )
        {
            pivots[INDEXVEC(k)] = 1;

            for (label j = k; j < size; ++j)
            {
                swap(factors[INDEXMAT(k, j, size)], factors[INDEXMAT(k+1, j, size)]);
            }
        }
        else
        {
            pivots[INDEXVEC(k)] = 0;
        }

        if (factors[INDEXMAT(k, k, size)] == 0)
        {
            factors[INDEXMAT(k, k, size)] = SMALL;
        }

        const scalar m =
            factors[INDEXMAT(k+1, k, size)]/factors[INDEXMAT(k, k, size)];

        factors[INDEXMAT(k+1, k, size)] = m;

        for (label j = k+1; j < size; ++j)
        {
            factors[INDEXMAT(k+1, j, size)] -= m*factors[INDEXMAT(k, j, size)];
        }
    }

    if (factors[INDEXMAT(size-1, size-1, size)] == 0)
    {
        factors[INDEXMAT(size-1, size-1, size)] = SMALL;
    }
}


__device__
void hessenbergSolve
(
    const scalar* __restrict__ hessenberg,
    const scalar* __restrict__ tau,
    const scalar* __restrict__ factors,
    const label* __restrict__ pivots,
    scalar* __restrict__ source,
    const label size
)
{
    // Q^T source, the reflections in the order they were produced
    for (label k = 0; k < size - 2; ++k)
    {
        applyReflector(hessenberg, tau[INDEXVEC(k)], k, source, size);
    }

    // The same interchanges and eliminations the factorisation performed
    for (label k = 0; k < size - 1; ++k)
    {
        if (pivots[INDEXVEC(k)])
        {
            swap(source[INDEXVEC(k)], source[INDEXVEC(k+1)]);
        }

        source[INDEXVEC(k+1)] -=
            factors[INDEXMAT(k+1, k, size)]*source[INDEXVEC(k)];
    }

    for (label i = size - 1; i >= 0; --i)
    {
        scalar sum = source[INDEXVEC(i)];

        for (label j = i + 1; j < size; ++j)
        {
            sum -= factors[INDEXMAT(i, j, size)]*source[INDEXVEC(j)];
        }

        source[INDEXVEC(i)] = sum/factors[INDEXMAT(i, i, size)];
    }

    // Q w, the reflections in reverse
    for (label k = size - 3; k >= 0; --k)
    {
        applyReflector(hessenberg, tau[INDEXVEC(k)], k, source, size);
    }
}
