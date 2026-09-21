#include "StepState.cuh"

__host__
void kodes::StepState::allocate(const label blocks)
{
    CUDA_CHECK(cudaMalloc(&deltaTTry, size_t(blocks) * sizeof(scalar)));
}

__host__
void kodes::StepState::deallocate()
{
    CUDA_CHECK(cudaFree(deltaTTry));
}

__device__
void kodes::StepState::setDeltaT(const scalar deltaT)
{
    deltaTTry[B_ID] = deltaT;
}
