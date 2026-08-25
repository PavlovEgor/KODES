#ifndef KODES_TEMPERATURE_BALANCER
#define KODES_TEMPERATURE_BALANCER

#pragma once

#include "Balancer.cuh"

namespace kodes
{

// Groups the batch by temperature, component 0 of the state vector. The
// cheapest useful key: it is already in the state, so the pass reads one
// scalar per system and nothing else. Cells of similar temperature burn on
// similar time scales, so they need a similar number of steps - but only
// roughly, since two cells at the same temperature and different composition
// do not. StiffnessBalancer refines exactly that.
//
// Selected by the name "temperature", see balancerTable.
class TemperatureBalancer
    :
    public Balancer
{
public:

    static constexpr label kKeyCount = 1;

    static constexpr bool kUsesDerivatives = false;

    __device__ __host__
    TemperatureBalancer
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
    ~TemperatureBalancer() = default;

    __device__ void
    key
    (
        DeviceResources* resources,
        const ODESystem* ode,
        const label system,
        scalar* key
    ) const override
    {
        key[0] = resources->vectorComponent(system, 0);
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

    KODES_DECLARE_DEVICE_OBJECT(TemperatureBalancer)
};

}

#endif
