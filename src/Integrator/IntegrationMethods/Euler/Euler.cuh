#ifndef KODES_EULER
#define KODES_EULER
#pragma once

#include <cuda_runtime.h>

#include "AdaptiveIntegrationMethod.cuh"

namespace kodes
{

// Explicit Euler, as a trial step: it takes one step of deltaTTry and reports
// how wrong it was, and the controller in AdaptiveIntegrationMethod::advance
// does the rest. Useful as the simplest thing the machinery can be checked
// against, not for a stiff mechanism.
//
// Selected by the name "euler", see methodTable.
class Euler
    :
    public AdaptiveIntegrationMethod
{
protected:

    // The step itself, dt*dydt0, which is also the error estimate
    ThreadVector<scalar> err_;

public:

    __device__ __host__
    Euler
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : AdaptiveIntegrationMethod(batchSize, scratchSize, blocks, systemSize, parameterSize)
    {}

    __device__ __host__
    ~Euler() = default;

    __host__ void allocate();

    __host__ void deallocate();

    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        return AdaptiveIntegrationMethod::scratchBytesPerThread(systemSize, parameterSize)
             + size_t(systemSize) * sizeof(scalar);
    }

    __device__ scalar
    trialStep
    (
        ODESystem* ode,
        IntegratorControls controls
    ) const override;

    KODES_DECLARE_DEVICE_OBJECT(Euler)
};

}

#endif
