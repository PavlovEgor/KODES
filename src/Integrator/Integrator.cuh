#ifndef Integrator_H
#define Integrator_H

#pragma once

#include <cuda_runtime.h>

#include "basic_types.cuh"
#include "IntegratorControls.cuh"
#include "LaunchConfig.cuh"
#include "Balancer.cuh"
#include "DeviceResources.cuh"
#include "IntegrationMethod.cuh"
#include "ODESystem.cuh"

namespace kodes
{

// The solve kernel. One instantiation for the whole library: the ODE system,
// the method, the resources and the balancer are all device objects the kernel
// dispatches on, so nothing about it depends on which ones were chosen.
__global__
void adaptive_solve
(
    ODESystem* ode,
    DeviceResources* resources,
    const IntegrationMethod* method,
    IntegratorControls controls
);

__global__
void setDeltaTKernel(const scalar deltaT, DeviceResources* resources);

__global__
void useOrderKernel(DeviceResources* resources, const label* order);

// Systems the device can integrate at the same time, from the occupancy of the
// solve kernel.
__host__ label maxConcurrentSystems(const label threads = KODES_BLOCK_SIZE);

// Resolve `request` against this device: how many threads to launch (and
// therefore how many scratch slots to allocate) and how many systems to ship
// per batch.
//
// The names are what tie the plan to the run: the method's entry knows what its
// resources cost per thread and per system, the balancer's entry knows what the
// keys and the order cost, and this adds them up before either exists. Anything
// owned outside both - for a pyJac mechanism, required_mechanism_size() - goes
// in the extra.
__host__ LaunchConfig planLaunch
(
    const label ensembleSize,
    const label systemSize,
    const label parameterSize,
    const char* methodName,
    const char* balancerName,
    const size_t extraScratchBytesPerThread = 0,
    const LaunchConfig& request = LaunchConfig()
);

// Drives the solve: owns the grid-stride loop over the batch, the step count
// loop that walks one system from local time 0 to the target end time, and the
// balancing pass in front of both.
//
// It holds all four of the runtime-selected objects as their abstract base, so
// which method, which balancer and which mechanism a run uses are decisions
// made when the program starts rather than when it was compiled.
class Integrator
{

protected:
    LaunchConfig config_;

    ODESystem* ode_;
    DeviceResources* resources_;
    const IntegrationMethod* method_;

    Balancer* balancer_;
    Balancer* balancerStub_;

    IntegratorControls controls_;

public:

    Integrator
    (
        ODESystem* ode,
        DeviceResources* resources,
        const IntegrationMethod* method,
        const LaunchConfig& config,
        const IntegratorControls& controls = IntegratorControls()
    );

    virtual ~Integrator();

    const IntegratorControls& controls() const { return controls_; }

    const LaunchConfig& config() const { return config_; }

    // Integrate the batch in the order this balancer sorts it into. Rebalanced
    // at the start of every solve(); pass nulls to go back to the copy order.
    void setBalancer(Balancer* balancer, Balancer* hostStub);

    void setDeltaT(const scalar deltaT);

    void solve(scalar deltaT, label realBatchSize);
};

}

#endif
