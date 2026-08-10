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

extern __constant__ label nSeq_[iMaxx_];
extern __constant__ scalar gpu_[iMaxx_];
extern __constant__ scalar coeff_[iMaxx_ * iMaxx_];

#endif 
