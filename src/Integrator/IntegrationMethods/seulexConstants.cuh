#ifndef KODES_SEULEX_CONSTANTS
#define KODES_SEULEX_CONSTANTS

#pragma once

#include "basicTypes.cuh"

// Highest extrapolation order Seulex may reach, and the length of the tables
// indexed by it. Also what sizes the order-indexed part of the Seulex scratch,
// which is why SeulexDeviceResources includes this header.
#define KODES_SEULEX_MAX_ORDER  12
#define KODES_SEULEX_TABLE_SIZE (KODES_SEULEX_MAX_ORDER + 1)

// Step size and order control coefficients
extern __constant__ scalar seulexStepFactor1;
extern __constant__ scalar seulexStepFactor2;
extern __constant__ scalar seulexStepFactor3;
extern __constant__ scalar seulexStepFactor4;
extern __constant__ scalar seulexStepFactor5;
extern __constant__ scalar seulexKFactor1;
extern __constant__ scalar seulexKFactor2;

// The step sequence, the cost model and the extrapolation coefficients. Worked
// out on the host from KODES_SEULEX_MAX_ORDER rather than written out by hand,
// and pushed into these symbols by uploadSeulexConstants().
extern __constant__ label  seulexStepSequence[KODES_SEULEX_TABLE_SIZE];
extern __constant__ scalar seulexWorkEstimate[KODES_SEULEX_TABLE_SIZE];
extern __constant__ scalar seulexExtrapolationCoeff
[
    KODES_SEULEX_TABLE_SIZE * KODES_SEULEX_TABLE_SIZE
];

namespace kodes
{

// Fill the three tables above. Called from SeulexDeviceResources::allocate,
// since it is the same tableau that sizes the order-indexed scratch.
__host__ void uploadSeulexConstants();

}

#endif
