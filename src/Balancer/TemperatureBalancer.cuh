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
class TemperatureBalancer
    :
    public Balancer
{
public:

    static constexpr label keyCount = 1;

    __device__ __host__
    TemperatureBalancer
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize
    )
        : Balancer(batchSize, scratchSize, systemSize, keyCount, false) {}

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

    __host__ static size_t bytesPerSystem()
    {
        return Balancer::bytesPerSystem(keyCount);
    }

    __host__ static size_t scratchBytesPerThread(const label systemSize)
    {
        return Balancer::scratchBytesPerThread(systemSize, false);
    }

    __host__ static TemperatureBalancer*
    create
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        TemperatureBalancer* hostStub
    );

    __host__ static void
    destroy(TemperatureBalancer* devBalancer, TemperatureBalancer* hostStub);

    // The host side stub, built here rather than by the caller. key() only
    // exists on the device, so the vtable of a host object can only be emitted
    // by a compiler that invents a host stub for a device-only virtual - nvcc
    // does, nvc++ -cuda does not. Keeping the construction in this .cu leaves
    // callers holding nothing but a pointer, which needs no vtable at all.
    __host__ static TemperatureBalancer*
    createStub(const label batchSize, const label scratchSize, const label systemSize);

    __host__ static void
    destroyStub(TemperatureBalancer* hostStub);
};

}

#endif
