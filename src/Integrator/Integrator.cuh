#ifndef Integrator_H
#define Integrator_H

#pragma once

#include <cuda_runtime.h>

#include "basic_types.cuh"
#include "IntegratorControls.cuh"
#include "LaunchConfig.cuh"
#include "Balancer.cuh"

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

template<class IntegratorDeviceResources>
__global__
void useOrder
(
    IntegratorDeviceResources* resources,
    const label* order
);

// Systems the device can integrate at the same time with this (ODESystem,
// IntegrationMethod, Resources) combination, from the occupancy of the solve
// kernel - so it accounts for the registers and shared memory the mechanism
// needs.
template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
__host__ label maxConcurrentSystems(const label threads = KODES_BLOCK_SIZE);

// Resolve `request` against this device: how many threads to launch (and
// therefore how many scratch slots to allocate) and how many systems to ship
// per batch. The two extras cover memory owned outside the resources object -
// for a pyJac mechanism required_mechanism_size() per thread, and for a sorted
// batch Balancer::bytesPerSystem() per system.
template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
__host__ LaunchConfig planLaunch
(
    const label ensembleSize,
    const label systemSize,
    const label parameterSize,
    const size_t extraScratchBytesPerThread = 0,
    const size_t extraStateBytesPerSystem = 0,
    const LaunchConfig& request = LaunchConfig()
);
}

namespace kodes
{

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
class Integrator
{

protected:
    LaunchConfig config_;

    ODESystem* ode_;
    IntegratorDeviceResources* resources_;

    Balancer* balancer_;
    Balancer* balancerStub_;

    IntegratorControls controls_;

public:

    Integrator
    (
        ODESystem* ode,
        IntegratorDeviceResources* resources,
        const LaunchConfig& config,
        const IntegratorControls& controls = IntegratorControls()
    );

    virtual ~Integrator();

    const IntegratorControls& controls() const { return controls_; }

    const LaunchConfig& config() const { return config_; }

    // Integrate the batch in the order this balancer sorts it into. Rebalanced
    // at the start of every solve(); pass nulls to go back to the copy order.
    //
    // A balancer whose key evaluates the right hand side is handed `ode` as a
    // kodes::ODESystem, so an ODESystem template argument that does not derive
    // from it can only be used with a balancer that does not need one.
    void setBalancer(Balancer* balancer, Balancer* hostStub);

    void setDeltaT(const scalar deltaT);

    __device__
    static void adaptiveStep
    (
        ODESystem* ode,
        IntegratorDeviceResources* resources,
        IntegratorControls controls
    );

    void solve(scalar deltaT, label realBatchSize);
};

}

#include "Integrator.cu"

#endif
