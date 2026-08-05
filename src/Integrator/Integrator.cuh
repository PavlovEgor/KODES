#ifndef Integrator_H
#define Integrator_H

#pragma once

#include <cuda_runtime.h>

#include "basic_types.cuh"
#include "IntegratorControls.cuh"

namespace kodes
{
template<class IntegratorDeviceResources>
__global__
void resetDeltaTMinKernel(IntegratorDeviceResources* resources);


template<class IntegratorDeviceResources>
__global__
void fetchDeltaTMinKernel(IntegratorDeviceResources* resources, scalar* deltaTMin);


template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
__global__
void adaptive_solve
(
    ODESystem* ode,
    IntegratorDeviceResources* resources,
    scalar deltaT,
    label realBatchSize,
    kodes::IntegratorControls controls
);
}

namespace kodes
{

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
class Integrator
{

protected:
    label threads;
    label blocks;
    size_t sharedMemSize;

    ODESystem* ode_;
    IntegratorDeviceResources* resources_;

    IntegratorControls controls_;

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

    const IntegratorControls& controls() const { return controls_; }

    scalar deltaTMin();

    void resetDeltaTMin();

    void solve(scalar deltaT, label realBatchSize);
};

}

#include "Integrator.cu"

#endif
