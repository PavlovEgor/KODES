#ifndef KODES_TEMPERATURE_BALANCER
#define KODES_TEMPERATURE_BALANCER

#pragma once

#include "Balancer.cuh"

namespace kodes
{

// Groups the batch by temperature, component 0 of the state vector. The
// simplest useful key: cells of similar temperature burn on similar time
// scales, so they need a similar number of steps.
class TemperatureBalancer
    :
    public Balancer
{
public:

    __device__ __host__
    TemperatureBalancer(const label batchSize) : Balancer(batchSize) {}

    __device__ __host__
    ~TemperatureBalancer() = default;

    __device__ scalar
    key(const DeviceResources* resources, const label system) const override
    {
        return resources->vectorComponent(system, 0);
    }

    __host__ static TemperatureBalancer*
    create(const label batchSize, TemperatureBalancer* hostStub);

    __host__ static void
    destroy(TemperatureBalancer* devBalancer, TemperatureBalancer* hostStub);
};

}

#endif
