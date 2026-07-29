#include "StepState.cuh"

__device__
kodes::StepState::StepState(label batchSize)
: batchSize_(batchSize)
{
    cudaMalloc(&forward, batchSize * sizeof(bool));
    cudaMalloc(&deltaTTry, batchSize * sizeof(scalar));
    cudaMalloc(&deltaTDid, batchSize * sizeof(scalar));
    cudaMalloc(&first, batchSize * sizeof(bool));
    cudaMalloc(&last, batchSize * sizeof(bool));
    cudaMalloc(&reject, batchSize * sizeof(bool));
    cudaMalloc(&prevReject, batchSize * sizeof(bool));
}

__device__
kodes::StepState::~StepState()
{
    cudaFree(forward);
    cudaFree(deltaTTry);
    cudaFree(deltaTDid);
    cudaFree(first);
    cudaFree(last);
    cudaFree(reject);
    cudaFree(prevReject);
}

__device__
void kodes::StepState::setDeltaT(const scalar deltaT)
{
    forward[INDEXVEC(0)] = deltaT > 0.0 ? true : false;
    deltaTTry[INDEXVEC(0)] = deltaT;
    deltaTDid[INDEXVEC(0)] = 0.0;
    first[INDEXVEC(0)] = true;
    last[INDEXVEC(0)]  = false;
    reject[INDEXVEC(0)]= false;
    prevReject[INDEXVEC(0)] = false;
}

__device__
scalar kodes::StepState::findMinDeltaT()
{

}