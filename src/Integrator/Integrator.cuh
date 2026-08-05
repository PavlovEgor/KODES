#ifndef Integrator_H
#define Integrator_H

#pragma once

#include <cuda_runtime.h>

#include "basic_types.cuh"
#include "IntegratorControls.cuh"

namespace kodes
{

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
__global__
void adaptive_solve
(
    ODESystem* ode,
    IntegratorDeviceResources* resources,
    kodes::IntegratorControls controls
);

template<class IntegratorDeviceResources>
__global__
void setDeltaT
(
    const scalar deltaT, 
    IntegratorDeviceResources* resources
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

    void setDeltaT(const scalar deltaT);

    void solve(scalar deltaT, label realBatchSize);
};

}

#include "Integrator.cu"

#endif
