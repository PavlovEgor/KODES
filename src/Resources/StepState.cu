#include "StepState.cuh"

__device__ __host__
kodes::StepState::StepState()
: deltaTMin(GREAT)
{}

__host__
void kodes::StepState::allocate(const label scratchSize)
{
    CUDA_CHECK(cudaMalloc(&forward, scratchSize * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&deltaTTry, scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&deltaTDid, scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&currentT, scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&first, scratchSize * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&last, scratchSize * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&reject, scratchSize * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&prevReject, scratchSize * sizeof(bool)));
}

__host__
void kodes::StepState::deallocate()
{
    CUDA_CHECK(cudaFree(forward));
    CUDA_CHECK(cudaFree(deltaTTry));
    CUDA_CHECK(cudaFree(deltaTDid));
    CUDA_CHECK(cudaFree(currentT));
    CUDA_CHECK(cudaFree(first));
    CUDA_CHECK(cudaFree(last));
    CUDA_CHECK(cudaFree(reject));
    CUDA_CHECK(cudaFree(prevReject));
}

__device__
void kodes::StepState::setDeltaT(const scalar deltaT)
{
    forward[INDEXVEC(0)] = deltaT > 0.0 ? true : false;
    deltaTTry[INDEXVEC(0)] = deltaT;
    deltaTDid[INDEXVEC(0)] = 0.0;
    currentT[INDEXVEC(0)] = 0.0;
    first[INDEXVEC(0)] = true;
    last[INDEXVEC(0)]  = false;
    reject[INDEXVEC(0)]= false;
    prevReject[INDEXVEC(0)] = false;
}

__device__
void kodes::StepState::resetStep()
{
    setDeltaT(deltaTTry[INDEXVEC(0)]);
}
