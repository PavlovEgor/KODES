#ifndef KODES_BDF_CONSTANTS
#define KODES_BDF_CONSTANTS

#pragma once

#include "basic_types.cuh"

namespace kodes
{

// Highest order Bdf may reach, and the length of the tables indexed by it
constexpr label kBdfMaxOrder = 5;
constexpr label kBdfTableSize = kBdfMaxOrder + 1;

// Newton iterations a trial may take before it counts as not converging
constexpr label kBdfNewtonMaxIter = 4;

// Bounds of the factor a step may grow or shrink by
constexpr scalar kBdfMinFactor = 0.2;
constexpr scalar kBdfMaxFactor = 10;

// The coefficients indexed by the order, which is known at run time only, so
// they are in memory - filled at compile time, unlike Seulex's tables:
//
//   gamma[k]      = 1 + 1/2 + ... + 1/k
//   alpha[k]      = (1 - kappa[k]) gamma[k]
//   errorConst[k] = kappa[k] gamma[k] + 1/(k+1)
//   u[m][j]       = prod_{l=1..m} (l - 1 - j)/l,
//
// with kappa Shampine and Reichelt's NDF coefficients, as scipy's BDF has
// them. u is the matrix that change of step size multiplies by - see
// DifferenceTable::rescale - for the step ratio 1, the same for any order.
struct BdfCoefficients
{
    scalar gamma[kBdfTableSize];
    scalar alpha[kBdfTableSize];
    scalar errorConst[kBdfTableSize];
    scalar u[kBdfTableSize][kBdfTableSize];
};

extern __constant__ BdfCoefficients kBdfCoefficients;

}

#endif
