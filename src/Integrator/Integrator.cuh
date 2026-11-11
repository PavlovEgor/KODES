#ifndef KODES_INTEGRATOR
#define KODES_INTEGRATOR
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
// the method - which holds the batch and the scratch, see IntegrationMethod -
// and the balancer are device objects the kernel dispatches on, so nothing
// about it depends on which ones were chosen.
//
// One system per thread: each thread takes its share of the batch in a
// grid-stride loop and integrates every system it picks up on its own, from
// local time 0 to the target end time, with its own step size. The balancer,
// null for none, decides only which system is at which position of that loop.
__global__
void solveKernel
(
    ODESystem* ode,
    IntegrationMethod* method,
    const Balancer* balancer,
    IntegratorControls controls
);

__global__
void setDeltaTKernel(const scalar deltaT, DeviceResources* resources);

// Systems the device can integrate at the same time, from the occupancy of the
// solve kernel. The shared memory the launch will ask for is an argument
// because it is what the occupancy mostly turns on, and how much of it there
// will be is a decision of the plan - see blockScratchInShared.
__host__ label maxConcurrentSystems
(
    const label threads = KODES_BLOCK_SIZE,
    const size_t sharedMemSize = 0
);

// Resolve `request` against this device: how many threads to launch (and
// therefore how many scratch slots to allocate) and how many systems to ship
// per batch.
//
// The names are what tie the plan to the run: the method's entry knows what its
// scratch costs per thread, per block and per system, the balancer's entry
// knows what the keys and the order cost, the system's entry knows what its
// own workspace costs - for a pyJac mechanism, required_mechanism_size() - and
// this adds all three up before any of them exists. Anything a caller owns
// outside all three still goes in the extra - which is now the empty case,
// since pyJac's workspace, the one thing that ever went there, belongs to the
// system's entry. That is why `request` comes before it: the call a caller
// writes ends at the name of a settings file.
//
// This is also where a device call stack the caller asks for is sized, because
// that has to happen before the free memory is counted; 0, the default, leaves
// the driver's own value alone: see KODES_DEVICE_STACK_BYTES.
__host__ LaunchConfig planLaunch
(
    const label ensembleSize,
    const label systemSize,
    const label parameterSize,
    const char* methodName,
    const char* balancerName,
    const char* systemName,
    const LaunchConfig& request = LaunchConfig(),
    const size_t extraScratchBytesPerThread = 0,
    const size_t deviceStackBytes = KODES_DEVICE_STACK_BYTES
);

// Drives the solve: owns the grid-stride loop over the batch, the step count
// loop that walks one system from local time 0 to the target end time, and the
// balancing pass in front of both.
//
// It holds the runtime-selected objects as their abstract base, so which
// method, which balancer and which mechanism a run uses are decisions made
// when the program starts rather than when it was compiled.
class Integrator
{

protected:
    LaunchConfig config_;

    ODESystem* ode_;
    IntegrationMethod* method_;

    // null for none, both or neither
    Balancer* balancer_;
    Balancer* balancerStub_;

    IntegratorControls controls_;

public:

    Integrator
    (
        ODESystem* ode,
        IntegrationMethod* method,
        const LaunchConfig& config,
        const IntegratorControls& controls = IntegratorControls()
    );

    virtual ~Integrator();

    const IntegratorControls& controls() const { return controls_; }

    const LaunchConfig& config() const { return config_; }

    // Integrate the batch in the order this balancer sorts it into, rebalanced
    // at the start of every solve(). Nulls go back to the order it was copied
    // in, which is also where an Integrator starts.
    void setBalancer(Balancer* balancer, Balancer* hostStub);

    void setDeltaT(const scalar deltaT);

    void solve(scalar deltaT, label realBatchSize);
};

}

#endif
