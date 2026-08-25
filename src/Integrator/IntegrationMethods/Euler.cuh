#ifndef KODES_EULER
#define KODES_EULER
#pragma once

#include <cuda_runtime.h>

#include "IntegrationMethod.cuh"
#include "EulerDeviceResources.cuh"

namespace kodes
{

// Explicit Euler, as a trial step: it takes one step of deltaTTry and reports
// how wrong it was, and the controller in IntegrationMethod::adaptiveStep does
// the rest. Useful as the simplest thing the machinery can be checked against,
// not for a stiff mechanism.
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
        const label systemSize,
        const label parameterSize
    )
        : IntegrationMethod
          (
              batchSize, scratchSize, systemSize, parameterSize,
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
        IntegratorControls controls
    ) const override;

    KODES_DECLARE_DEVICE_OBJECT(Euler)
};

}

#endif
