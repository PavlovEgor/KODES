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


// w = (I + eta*M^-1) u, the operator of the left preconditioned shifted
// system. One back substitution and one axpy, no matrix vector product
__device__ static inline
void applyShiftedOperator
(
    const scalar* __restrict__ luMatrix,
    const label* __restrict__ pivotIndices,
    const scalar eta,
    const scalar* __restrict__ u,
    scalar* __restrict__ w,
    const label size
)
{
    copyVec(w, u, size);

    LUBacksubstitute(luMatrix, pivotIndices, w, size);

    for (label i = 0; i < size; ++i)
    {
        w[INDEXVEC(i)] = u[INDEXVEC(i)] + eta*w[INDEXVEC(i)];
    }
}


__device__
label shiftedBiCGStab
(
    const scalar* __restrict__ luMatrix,
    const label* __restrict__ pivotIndices,
    const scalar eta,
    scalar* __restrict__ x,
    const scalar* __restrict__ scale,
    scalar* __restrict__ work,
    const label size,
    const scalar tol,
    const label maxIter
)
{
    const label stride = size*GRID_DIM;

    scalar* r     = work;
    scalar* rHat  = work + 1*stride;
    scalar* p     = work + 2*stride;
    scalar* v     = work + 3*stride;
    scalar* s     = work + 4*stride;
    scalar* t     = work + 5*stride;
    scalar* bSave = work + 6*stride;

    copyVec(bSave, x, size);

    // Residual of the preconditioned system at the zero iterate, r = M^-1 b
    copyVec(r, x, size);

    LUBacksubstitute(luMatrix, pivotIndices, r, size);

    zeroVec(x, size);

    // M^-1 is close to the inverse of the shifted matrix, so this residual is
    // already an estimate of the error of x itself and can be compared with
    // the solution scale directly
    if (scaledNorm(r, scale, size) <= tol)
    {
        copyVec(x, r, size);
        return 0;
    }

    copyVec(rHat, r, size);
    copyVec(p, r, size);

    scalar rho = dotProduct(rHat, r, size);

    label iter = 1;
    bool converged = false;
    bool brokeDown = (rho == 0);

    for (; iter <= maxIter && !converged && !brokeDown; ++iter)
    {
        applyShiftedOperator(luMatrix, pivotIndices, eta, p, v, size);

        const scalar rHatV = dotProduct(rHat, v, size);

        if (rHatV == 0)
        {
            brokeDown = true;
            break;
        }

        const scalar alpha = rho/rHatV;

        for (label i = 0; i < size; ++i)
        {
            s[INDEXVEC(i)] = r[INDEXVEC(i)] - alpha*v[INDEXVEC(i)];
            x[INDEXVEC(i)] += alpha*p[INDEXVEC(i)];
        }

        if (scaledNorm(s, scale, size) <= tol)
        {
            converged = true;
            break;
        }

        applyShiftedOperator(luMatrix, pivotIndices, eta, s, t, size);

        const scalar tt = dotProduct(t, t, size);

        if (tt == 0)
        {
            // t = (I + eta*M^-1) s = 0 with a non singular operator means
            // s = 0, so the update above already gave the exact solution
            converged = true;
            break;
        }

        const scalar omega = dotProduct(t, s, size)/tt;

        for (label i = 0; i < size; ++i)
        {
            x[INDEXVEC(i)] += omega*s[INDEXVEC(i)];
            r[INDEXVEC(i)] = s[INDEXVEC(i)] - omega*t[INDEXVEC(i)];
        }

        if (scaledNorm(r, scale, size) <= tol)
        {
            converged = true;
            break;
        }

        if (omega == 0)
        {
            // the recurrence for p cannot be continued
            brokeDown = true;
            break;
        }

        const scalar rhoOld = rho;
        rho = dotProduct(rHat, r, size);

        if (rho == 0)
        {
            brokeDown = true;
            break;
        }

        const scalar beta = (rho/rhoOld)*(alpha/omega);

        for (label i = 0; i < size; ++i)
        {
            p[INDEXVEC(i)] =
                r[INDEXVEC(i)] + beta*(p[INDEXVEC(i)] - omega*v[INDEXVEC(i)]);
        }
    }

    if (!converged)
    {
        copyVec(x, bSave, size);
        return -1;
    }

    return iter;
}
