#ifndef KODES_RHS_NORM_BALANCER
#define KODES_RHS_NORM_BALANCER

#pragma once

#include "Balancer.cuh"

namespace kodes
{

// Groups the batch by how fast it is moving: log10 of the RMS relative rate of
// change of the state, see relativeRHSNorm(). Where temperature is a proxy for
// stiffness, this is a measurement of it - it sees a cell that is hot but burnt
// out, and one that is cool but about to ignite, for what they are.
//
// It is not free: the pass now loads every system into a scratch slot and
// evaluates the right hand side once per system, which for a generated
// mechanism is the same work as one step of an explicit integrator. Against a
// solve that takes hundreds of implicit steps that is small, but it is not
// nothing, and it is why TemperatureBalancer is still worth having.
class RHSNormBalancer
    :
    public Balancer
{
public:

    static constexpr label keyCount = 1;

    __device__ __host__
    RHSNormBalancer
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize
    )
        : Balancer(batchSize, scratchSize, systemSize, keyCount, true) {}

    __device__ __host__
    ~RHSNormBalancer() = default;

    __device__ void
    key
    (
        DeviceResources* resources,
        const ODESystem* ode,
        const label system,
        scalar* key
    ) const override
    {
        scalar* y = resources->currentVector();
        scalar* dydt = this->dydt();

        ode->derivatives(0.0, resources->currentParameter(0), y, dydt);

        key[0] = relativeRHSNorm(y, dydt, resources->systemSize());
    }

    __host__ static size_t bytesPerSystem()
    {
        return Balancer::bytesPerSystem(keyCount);
    }

    __host__ static size_t scratchBytesPerThread(const label systemSize)
    {
        return Balancer::scratchBytesPerThread(systemSize, true);
    }

    __host__ static RHSNormBalancer*
    create
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        RHSNormBalancer* hostStub
    );

    __host__ static void
    destroy(RHSNormBalancer* devBalancer, RHSNormBalancer* hostStub);

    // see TemperatureBalancer::createStub for why this is not the caller's job
    __host__ static RHSNormBalancer*
    createStub(const label batchSize, const label scratchSize, const label systemSize);

    __host__ static void
    destroyStub(RHSNormBalancer* hostStub);
};

}

#endif
