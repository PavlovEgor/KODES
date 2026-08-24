#include "TemperatureBalancer.cuh"
#include "BalancerFactory.cuh"

__host__ kodes::TemperatureBalancer*
kodes::TemperatureBalancer::create
(
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    kodes::TemperatureBalancer* hostStub
)
{
    return kodes::createBalancer(batchSize, scratchSize, systemSize, hostStub);
}

__host__ void
kodes::TemperatureBalancer::destroy
(
    kodes::TemperatureBalancer* devBalancer,
    kodes::TemperatureBalancer* hostStub
)
{
    kodes::destroyBalancer(devBalancer, hostStub);
}

// ::new and ::delete, since the class hides the global operator new behind its
// own device side placement one
__host__ kodes::TemperatureBalancer*
kodes::TemperatureBalancer::createStub
(
    const label batchSize,
    const label scratchSize,
    const label systemSize
)
{
    return ::new TemperatureBalancer(batchSize, scratchSize, systemSize);
}

__host__ void
kodes::TemperatureBalancer::destroyStub(kodes::TemperatureBalancer* hostStub)
{
    ::delete hostStub;
}
