

#ifndef Integrator_H
#define Integrator_H

#pragma once

#include <cuda_runtime.h>

#include "basic_types.cuh"


// Resets the ensemble wide minimum of deltaTTry, launched with a single thread
template<class SolverDeviceResources>
__global__
void resetDeltaTMinKernel(SolverDeviceResources* resources)
{
    resources->setDeltaTMinToGreat();
}


// Copies the ensemble wide minimum of deltaTTry into a buffer the host can read
template<class SolverDeviceResources>
__global__
void fetchDeltaTMinKernel(SolverDeviceResources* resources, scalar* deltaTMin)
{
    *deltaTMin = resources->deltaTMin;
}


namespace kodes
{
template<class ODESystem, class SolverDeviceResources>
class Integrator
{

protected:
    label threads;
    label blocks;
    size_t sharedMemSize;

    ODESystem* ode_;
    SolverDeviceResources* resources_;

    // Single scalar on the device, used to read deltaTMin back to the host
    scalar* deltaTMinDevice_;

public:

    Integrator(ODESystem* ode, SolverDeviceResources* resources, label batchSize);

    virtual ~Integrator();

    virtual void solve(scalar deltaT, label realBatchSize) =0;

    // Smallest deltaTTry over every system solved since the last resetDeltaTMin(),
    // synchronises with the device
    scalar deltaTMin();

    void resetDeltaTMin();
};


template<class ODESystem, class SolverDeviceResources>
Integrator<ODESystem, SolverDeviceResources>::Integrator(ODESystem* ode, SolverDeviceResources* resources, label batchSize)
: ode_(ode), resources_(resources)
{
    threads = kodes::blockSize(batchSize);
    blocks = kodes::numOfBlocks(batchSize);
    sharedMemSize = kodes::sharedMemorySize(batchSize);

    if (batchSize != threads * blocks)
    {
        printf("batchSize != threads * blocks");
    }

    cudaMalloc(&deltaTMinDevice_, sizeof(scalar));
}

template<class ODESystem, class SolverDeviceResources>
Integrator<ODESystem, SolverDeviceResources>::~Integrator()
{
    cudaFree(deltaTMinDevice_);
}

template<class ODESystem, class SolverDeviceResources>
void Integrator<ODESystem, SolverDeviceResources>::resetDeltaTMin()
{
    resetDeltaTMinKernel<SolverDeviceResources><<<1, 1>>>(resources_);
}

template<class ODESystem, class SolverDeviceResources>
scalar Integrator<ODESystem, SolverDeviceResources>::deltaTMin()
{
    scalar minDeltaT = GREAT;

    fetchDeltaTMinKernel<SolverDeviceResources><<<1, 1>>>(resources_, deltaTMinDevice_);

    // Blocking copy, waits for every solve() launched before it
    cudaMemcpy(&minDeltaT, deltaTMinDevice_, sizeof(scalar), cudaMemcpyDeviceToHost);

    return minDeltaT;
}

}

#endif
