#ifndef KODES_SEULEX_CONSTANTS
#define KODES_SEULEX_CONSTANTS

#pragma once

#include "basic_types.cuh"

namespace kodes
{

// Highest extrapolation order Seulex may reach, and the length of the tables
// indexed by it. Also what sizes the order-indexed part of the Seulex scratch,
// which is why Seulex.cuh includes this header.
//
// Constants rather than macros: nothing about a build should override them, and
// a constant obeys the scoping rules a macro ignores.
constexpr label kSeulexMaxOrder = 12;
constexpr label kSeulexTableSize = kSeulexMaxOrder + 1;

// Step size and order control coefficients. constexpr, not __constant__: only
// a table read at an index known at run time needs to be in memory, and a
// value the compiler sees can fold into the arithmetic around it. Until 1.2.8
// they were __constant__, defined apart in seulex_constants.cu.
constexpr scalar kSeulexStepFactor1 = 0.6;
constexpr scalar kSeulexStepFactor2 = 0.93;
constexpr scalar kSeulexStepFactor3 = 0.1;
constexpr scalar kSeulexStepFactor4 = 4;
constexpr scalar kSeulexStepFactor5 = 0.5;
constexpr scalar kSeulexKFactor1 = 0.7;
constexpr scalar kSeulexKFactor2 = 0.9;

// The step sequence, the cost model and the extrapolation coefficients. Worked
// out on the host from kSeulexMaxOrder rather than written out by hand, and
// pushed into these symbols by uploadSeulexConstants().
extern __constant__ label  kSeulexStepSequence[kSeulexTableSize];
extern __constant__ scalar kSeulexWorkEstimate[kSeulexTableSize];
extern __constant__ scalar kSeulexExtrapolationCoeff[kSeulexTableSize * kSeulexTableSize];

// Fill the three tables above. Called from Seulex::allocate,
// since it is the same tableau that sizes the order-indexed scratch.
__host__ void uploadSeulexConstants();

}

#endif
