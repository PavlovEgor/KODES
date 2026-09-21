#ifndef KODES_EULER
#define KODES_EULER
#pragma once

#include <cuda_runtime.h>

#include "IntegrationMethod.cuh"
#include "EulerDeviceResources.cuh"

namespace kodes
{

// Explicit Euler, as a trial step: it takes one step of the block's deltaTTry
// and reports how wrong it was over the whole block, and the controller in
// IntegrationMethod::adaptiveStep does the rest. Useful as the simplest thing
// the machinery can be checked against, not for a stiff mechanism.
//
// It has nothing to gain from stepping a whole block at once - it factorises
// no matrix and evaluates no Jacobian - and something to lose, since it now
// takes the step its stiffest system needs. It does it anyway: one driver with
// one control flow is worth more here than a fast explicit method the library
// does not claim to offer.
//
// Selected by the name "euler", see methodTable, which pairs it with
// EulerDeviceResources.
class Euler
    :
    public IntegrationMethod
{
public:

    static constexpr bool kUsesAdaptiveStep = true;

    __device__ __host__
    Euler
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : IntegrationMethod
          (
              batchSize, scratchSize, blocks, systemSize, parameterSize,
              kUsesAdaptiveStep
          )
    {}

    __device__ __host__
    ~Euler() = default;

    __device__ scalar
    step
    (
        ODESystem* ode,
        DeviceResources* resources,
        BlockStep& step,
        const bool active,
        IntegratorControls controls
    ) const override;

    KODES_DECLARE_DEVICE_OBJECT(Euler)
};

}

#endif
