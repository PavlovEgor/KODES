#include "StepState.cuh"

__host__
void kodes::StepState::allocate(const label scratchSize)
{
    CUDA_CHECK(cudaMalloc(&forward, size_t(scratchSize) * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&deltaTTry, size_t(scratchSize) * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&deltaTDid, size_t(scratchSize) * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&currentT, size_t(scratchSize) * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&first, size_t(scratchSize) * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&last, size_t(scratchSize) * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&reject, size_t(scratchSize) * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&prevReject, size_t(scratchSize) * sizeof(bool)));
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
    forward[INDEXVEC(0)] = deltaT > 0.0;
    deltaTTry[INDEXVEC(0)] = deltaT;
    deltaTDid[INDEXVEC(0)] = 0.0;
    currentT[INDEXVEC(0)] = 0.0;
    first[INDEXVEC(0)] = true;
    last[INDEXVEC(0)] = false;
    reject[INDEXVEC(0)] = false;
    prevReject[INDEXVEC(0)] = false;
}

__device__
void kodes::StepState::resetStep()
{
    setDeltaT(deltaTTry[INDEXVEC(0)]);
}
