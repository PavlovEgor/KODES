#ifndef KODES_ADAPTIVE_INTEGRATION_METHOD
#define KODES_ADAPTIVE_INTEGRATION_METHOD

#pragma once

#include "IntegrationMethod.cuh"

namespace kodes
{

// A method that only knows how to take a *trial* step and say how wrong it
// was. advance() is the step size controller they all share: retry with a
// smaller step until the error is inside the tolerance, then grow the next
// one. A subclass gives trialStep() and nothing else.
//
// An intermediate of the family, not a row of any table, so no
// KODES_DECLARE_DEVICE_OBJECT; the four statics belong to the leaves. Kept
// beside IntegrationMethod rather than in a directory of its own: it is part
// of the base, not a choice.
class AdaptiveIntegrationMethod
    :
    public IntegrationMethod
{
protected:

    // The trial state, which advance() copies into the current vector once it
    // is accepted, so that a rejected step is retried from an untouched one
    ThreadVector<scalar> yTemp_;

    // The right hand side at the top of the step, the same for every retry
    ThreadVector<scalar> dydt0_;

public:

    __device__ __host__
    AdaptiveIntegrationMethod
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : IntegrationMethod(batchSize, scratchSize, blocks, systemSize, parameterSize)
    {}

    __device__ __host__
    ~AdaptiveIntegrationMethod() = default;

    __host__ void allocate();

    __host__ void deallocate();

    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        return IntegrationMethod::scratchBytesPerThread(systemSize, parameterSize)
             + 2 * size_t(systemSize) * sizeof(scalar);
    }

    // Whatever the trial steps of one step share and a retry need not redo,
    // worked out once from the state the step starts from, after dydt0_.
    // Nothing, unless a subclass says otherwise.
    //
    // Not given the controls: nvlink matches an indirect call to whatever
    // has its signature, and startStep(ode, controls) would have advance's -
    // a call of advance from advance, a recursion, and a solveKernel stack it
    // can no longer work out.
    __device__ virtual void startStep(ODESystem* ode) const
    {}

    // One trial step of deltaTTry from the current vector into yTemp_, and its
    // error normalised so that 1 is the tolerance
    __device__ virtual scalar
    trialStep
    (
        ODESystem* ode,
        IntegratorControls controls
    ) const = 0;

    __device__ void
    advance
    (
        ODESystem* ode,
        IntegratorControls controls
    ) const override;
};

}

#endif
