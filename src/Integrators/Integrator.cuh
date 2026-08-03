

#ifndef Integrator_H
#define Integrator_H

#pragma once

#include <cuda_runtime.h>

#include "basic_types.cuh"
#include "IntegratorControls.cuh"

// Resets the ensemble wide minimum of deltaTTry, launched with a single thread
template<class IntegratorDeviceResources>
__global__
void resetDeltaTMinKernel(IntegratorDeviceResources* resources)
{
    resources->setDeltaTMinToGreat();
}


// Copies the ensemble wide minimum of deltaTTry into a buffer the host can read
template<class IntegratorDeviceResources>
__global__
void fetchDeltaTMinKernel(IntegratorDeviceResources* resources, scalar* deltaTMin)
{
    *deltaTMin = resources->deltaTMin;
}


namespace kodes
{

template<class ODESystem, class IntegratorDeviceResources>
class Integrator
{

protected:
    label threads;
    label blocks;
    size_t sharedMemSize;

    ODESystem* ode_;
    IntegratorDeviceResources* resources_;

    IntegratorControls controls_;

    // Single scalar on the device, used to read deltaTMin back to the host
    scalar* deltaTMinDevice_;

public:

    Integrator
    (
        ODESystem* ode,
        IntegratorDeviceResources* resources,
        label batchSize,
        const IntegratorControls& controls = IntegratorControls()
    );

    virtual ~Integrator();

    virtual void solve(scalar deltaT, label realBatchSize, bool firstBatch) =0;

    const IntegratorControls& controls() const { return controls_; }

    // Smallest deltaTTry over every system solved since the last resetDeltaTMin(),
    // synchronises with the device
    scalar deltaTMin();

    void resetDeltaTMin();
};


template<class ODESystem, class IntegratorDeviceResources>
Integrator<ODESystem, IntegratorDeviceResources>::Integrator
(
    ODESystem* ode,
    IntegratorDeviceResources* resources,
    label batchSize,
    const IntegratorControls& controls
)
: ode_(ode), resources_(resources), controls_(controls)
{
    threads = kodes::blockSize(batchSize);
    blocks = kodes::numOfBlocks(batchSize);
    sharedMemSize = kodes::sharedMemorySize(batchSize);

    if (batchSize != threads * blocks)
    {
        printf("batchSize != threads * blocks\n");
    }

    cudaMalloc(&deltaTMinDevice_, sizeof(scalar));
}

template<class ODESystem, class IntegratorDeviceResources>
Integrator<ODESystem, IntegratorDeviceResources>::~Integrator()
{
    cudaFree(deltaTMinDevice_);
}

template<class ODESystem, class IntegratorDeviceResources>
void Integrator<ODESystem, IntegratorDeviceResources>::resetDeltaTMin()
{
    resetDeltaTMinKernel<IntegratorDeviceResources><<<1, 1>>>(resources_);
}

template<class ODESystem, class IntegratorDeviceResources>
scalar Integrator<ODESystem, IntegratorDeviceResources>::deltaTMin()
{
    scalar minDeltaT = GREAT;

    fetchDeltaTMinKernel<IntegratorDeviceResources><<<1, 1>>>(resources_, deltaTMinDevice_);

    // Blocking copy, waits for every solve() launched before it
    cudaMemcpy(&minDeltaT, deltaTMinDevice_, sizeof(scalar), cudaMemcpyDeviceToHost);

    return minDeltaT;
}

}

#endif
