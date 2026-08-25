#ifndef SEULEXCONST
#define SEULEXCONST

#include "basic_types.cuh"

#pragma once

#define kMaxx_ (12)
#define iMaxx_ (kMaxx_ + 1)

extern __constant__ scalar stepFactor1_;
extern __constant__ scalar stepFactor2_;
extern __constant__ scalar stepFactor3_;
extern __constant__ scalar stepFactor4_;
extern __constant__ scalar stepFactor5_;
extern __constant__ scalar kFactor1_;
extern __constant__ scalar kFactor2_;

// The step sequence, the cost model and the extrapolation coefficients. Built
// from kMaxx_ rather than written out, so they are worked out on the host once
// and pushed into the constants above - see uploadSeulexConstants().
extern __constant__ label nSeq_[iMaxx_];
extern __constant__ scalar gpu_[iMaxx_];
extern __constant__ scalar coeff_[iMaxx_ * iMaxx_];

namespace kodes
{
    // Fill the three tables above. Called from SeulexDeviceResources::allocate,
    // since it is the same tableau that sizes the order-indexed scratch.
    __host__ void uploadSeulexConstants();
}

#endif
