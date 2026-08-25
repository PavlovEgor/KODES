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
//
// Selected by the name "rhsNorm", see balancerTable.
class RHSNormBalancer
    :
    public Balancer
{
public:

    static constexpr label keyCount = 1;

    static constexpr bool usesDerivatives = true;

    __device__ __host__
    RHSNormBalancer
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        const label parameterSize
    )
        : Balancer
          (
              batchSize, scratchSize, systemSize, parameterSize,
              keyCount, usesDerivatives
          )
    {}

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

    __host__ static size_t
    stateBytesPerSystem(const label systemSize, const label parameterSize)
    {
        return Balancer::keyBytesPerSystem(keyCount);
    }

    __host__ static size_t
    scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        return Balancer::keyScratchBytesPerThread(systemSize, usesDerivatives);
    }

    KODES_DECLARE_DEVICE_OBJECT(RHSNormBalancer)
};

}

#endif
