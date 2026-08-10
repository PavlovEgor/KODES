#include "SeulexConstants.cuh"

__constant__ scalar stepFactor1_ = 0.6;
__constant__ scalar stepFactor2_ = 0.4;
__constant__ scalar stepFactor3_ = 0.3;
__constant__ scalar stepFactor4_ = 0.2;
__constant__ scalar stepFactor5_ = 0.1;
__constant__ scalar kFactor1_ = 0.5;
__constant__ scalar kFactor2_ = 0.5;

__constant__ label nSeq_[iMaxx_];
__constant__ scalar gpu_[iMaxx_];
__constant__ scalar coeff_[iMaxx_ * iMaxx_];
