#include "seulex_constants.cuh"

__constant__ scalar kSeulexStepFactor1 = 0.6;
__constant__ scalar kSeulexStepFactor2 = 0.93;
__constant__ scalar kSeulexStepFactor3 = 0.1;
__constant__ scalar kSeulexStepFactor4 = 4;
__constant__ scalar kSeulexStepFactor5 = 0.5;
__constant__ scalar kSeulexKFactor1 = 0.7;
__constant__ scalar kSeulexKFactor2 = 0.9;

__constant__ label  kSeulexStepSequence[kSeulexTableSize];
__constant__ scalar kSeulexWorkEstimate[kSeulexTableSize];
__constant__ scalar kSeulexExtrapolationCoeff
[
    kSeulexTableSize * kSeulexTableSize
];

__host__ void kodes::uploadSeulexConstants()
{
    // relative cost on the GPU of the major parts of the algorithm
    const scalar costFunc = 2, costJac = 40, costLU = 17, costSolve = 1;

    label  stepSequence[kSeulexTableSize];
    scalar workEstimate[kSeulexTableSize];
    scalar extrapolationCoeff[kSeulexTableSize * kSeulexTableSize];

    stepSequence[0] = 2;
    stepSequence[1] = 3;

    for (label i = 2; i < kSeulexTableSize; i++)
    {
        stepSequence[i] = 2 * stepSequence[i-2];
    }

    workEstimate[0] = costJac + costLU + stepSequence[0]*(costFunc + costSolve);

    for (label k = 0; k < kSeulexMaxOrder; k++)
    {
        workEstimate[k+1] = workEstimate[k]
                          + (stepSequence[k+1] - 1)*(costFunc + costSolve)
                          + costLU;
    }

    for (label k = 0; k < kSeulexTableSize; k++)
    {
        for (label l = 0; l < k; l++)
        {
            const scalar ratio = scalar(stepSequence[k])/stepSequence[l];

            extrapolationCoeff[k + l*kSeulexTableSize] = 1/(ratio - 1);
        }
    }

    CUDA_CHECK
    (
        cudaMemcpyToSymbol
        (
            kSeulexStepSequence, stepSequence,
            kSeulexTableSize * sizeof(label)
        )
    );

    CUDA_CHECK
    (
        cudaMemcpyToSymbol
        (
            kSeulexWorkEstimate, workEstimate,
            kSeulexTableSize * sizeof(scalar)
        )
    );

    CUDA_CHECK
    (
        cudaMemcpyToSymbol
        (
            kSeulexExtrapolationCoeff, extrapolationCoeff,
            kSeulexTableSize * kSeulexTableSize * sizeof(scalar)
        )
    );
}
