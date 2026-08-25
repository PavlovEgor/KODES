#ifndef KODES_STIFFNESS_BALANCER
#define KODES_STIFFNESS_BALANCER

#pragma once

#include "Balancer.cuh"

namespace kodes
{

// Two keys: temperature first, then log10 of the RMS relative rate of change
// inside each band of it. The example of what more than one key is for.
//
// Temperature alone leaves a band holding cells of the same temperature that
// are nowhere near each other in composition - fresh mixture next to burnt gas
// - and those do not need the same number of steps. The right hand side norm
// tells them apart, but on its own it collapses the distinction between a cold
// cell and a hot one that has already reached equilibrium, which then share a
// bin and a warp while their Jacobians have nothing in common. Ordering by the
// first and refining by the second keeps both.
//
// The cost is resolution: two keys share KODES_BALANCER_BUCKETS buckets between
// them, so each is cut into 128 bins rather than 16384.
//
// Selected by the name "stiffness", see balancerTable.
class StiffnessBalancer
    :
    public Balancer
{
public:

    static constexpr label kKeyCount = 2;

    static constexpr bool kUsesDerivatives = true;

    __device__ __host__
    StiffnessBalancer
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        const label parameterSize
    )
        : Balancer
          (
              batchSize, scratchSize, systemSize, parameterSize,
              kKeyCount, kUsesDerivatives
          )
    {}

    __device__ __host__
    ~StiffnessBalancer() = default;

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

        key[0] = resources->vectorComponent(system, 0);
        key[1] = relativeRHSNorm(y, dydt, resources->systemSize());
    }

    __host__ static size_t
    stateBytesPerSystem(const label systemSize, const label parameterSize)
    {
        return Balancer::keyBytesPerSystem(kKeyCount);
    }

    __host__ static size_t
    scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        return Balancer::keyScratchBytesPerThread(systemSize, kUsesDerivatives);
    }

    KODES_DECLARE_DEVICE_OBJECT(StiffnessBalancer)
};

}

#endif
