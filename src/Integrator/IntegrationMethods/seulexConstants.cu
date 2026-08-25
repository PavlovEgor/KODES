#include "seulexConstants.cuh"

__constant__ scalar seulexStepFactor1 = 0.6;
__constant__ scalar seulexStepFactor2 = 0.93;
__constant__ scalar seulexStepFactor3 = 0.1;
__constant__ scalar seulexStepFactor4 = 4;
__constant__ scalar seulexStepFactor5 = 0.5;
__constant__ scalar seulexKFactor1 = 0.7;
__constant__ scalar seulexKFactor2 = 0.9;

__constant__ label  seulexStepSequence[KODES_SEULEX_TABLE_SIZE];
__constant__ scalar seulexWorkEstimate[KODES_SEULEX_TABLE_SIZE];
__constant__ scalar seulexExtrapolationCoeff
[
    KODES_SEULEX_TABLE_SIZE * KODES_SEULEX_TABLE_SIZE
];

__host__ void kodes::uploadSeulexConstants()
{
    // relative cost on the GPU of the major parts of the algorithm
    const scalar costFunc = 2, costJac = 40, costLU = 17, costSolve = 1;

    label  stepSequence[KODES_SEULEX_TABLE_SIZE];
    scalar workEstimate[KODES_SEULEX_TABLE_SIZE];
    scalar extrapolationCoeff[KODES_SEULEX_TABLE_SIZE * KODES_SEULEX_TABLE_SIZE];

    stepSequence[0] = 2;
    stepSequence[1] = 3;

    for (label i = 2; i < KODES_SEULEX_TABLE_SIZE; i++)
    {
        stepSequence[i] = 2 * stepSequence[i-2];
    }

    workEstimate[0] = costJac + costLU + stepSequence[0]*(costFunc + costSolve);

    for (label k = 0; k < KODES_SEULEX_MAX_ORDER; k++)
    {
        workEstimate[k+1] = workEstimate[k]
                          + (stepSequence[k+1] - 1)*(costFunc + costSolve)
                          + costLU;
    }

    for (label k = 0; k < KODES_SEULEX_TABLE_SIZE; k++)
    {
        for (label l = 0; l < k; l++)
        {
            const scalar ratio = scalar(stepSequence[k])/stepSequence[l];

            extrapolationCoeff[k + l*KODES_SEULEX_TABLE_SIZE] = 1/(ratio - 1);
        }
    }

    CUDA_CHECK
    (
        cudaMemcpyToSymbol
        (
            seulexStepSequence, stepSequence,
            KODES_SEULEX_TABLE_SIZE * sizeof(label)
        )
    );

    CUDA_CHECK
    (
        cudaMemcpyToSymbol
        (
            seulexWorkEstimate, workEstimate,
            KODES_SEULEX_TABLE_SIZE * sizeof(scalar)
        )
    );

    CUDA_CHECK
    (
        cudaMemcpyToSymbol
        (
            seulexExtrapolationCoeff, extrapolationCoeff,
            KODES_SEULEX_TABLE_SIZE * KODES_SEULEX_TABLE_SIZE * sizeof(scalar)
        )
    );
}
