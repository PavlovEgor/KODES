#ifndef KODES_SEULEX_CONSTANTS
#define KODES_SEULEX_CONSTANTS

#pragma once

#include "basic_types.cuh"

// Highest extrapolation order Seulex may reach, and the length of the tables
// indexed by it. Also what sizes the order-indexed part of the Seulex scratch,
// which is why SeulexDeviceResources includes this header.
//
// Constants rather than macros: nothing about a build should override them, and
// a constant obeys the scoping rules a macro ignores.
constexpr label kSeulexMaxOrder = 12;
constexpr label kSeulexTableSize = kSeulexMaxOrder + 1;

// Step size and order control coefficients
extern __constant__ scalar kSeulexStepFactor1;
extern __constant__ scalar kSeulexStepFactor2;
extern __constant__ scalar kSeulexStepFactor3;
extern __constant__ scalar kSeulexStepFactor4;
extern __constant__ scalar kSeulexStepFactor5;
extern __constant__ scalar kSeulexKFactor1;
extern __constant__ scalar kSeulexKFactor2;

// The step sequence, the cost model and the extrapolation coefficients. Worked
// out on the host from kSeulexMaxOrder rather than written out by hand, and
// pushed into these symbols by uploadSeulexConstants().
extern __constant__ label  kSeulexStepSequence[kSeulexTableSize];
extern __constant__ scalar kSeulexWorkEstimate[kSeulexTableSize];
extern __constant__ scalar kSeulexExtrapolationCoeff[kSeulexTableSize * kSeulexTableSize];

namespace kodes
{

// Fill the three tables above. Called from SeulexDeviceResources::allocate,
// since it is the same tableau that sizes the order-indexed scratch.
__host__ void uploadSeulexConstants();

}

#endif
