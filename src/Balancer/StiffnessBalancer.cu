#include "StiffnessBalancer.cuh"
#include "BalancerFactory.cuh"

__host__ kodes::StiffnessBalancer*
kodes::StiffnessBalancer::create
(
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    kodes::StiffnessBalancer* hostStub
)
{
    return kodes::createBalancer(batchSize, scratchSize, systemSize, hostStub);
}

__host__ void
kodes::StiffnessBalancer::destroy
(
    kodes::StiffnessBalancer* devBalancer,
    kodes::StiffnessBalancer* hostStub
)
{
    kodes::destroyBalancer(devBalancer, hostStub);
}

__host__ kodes::StiffnessBalancer*
kodes::StiffnessBalancer::createStub
(
    const label batchSize,
    const label scratchSize,
    const label systemSize
)
{
    return ::new StiffnessBalancer(batchSize, scratchSize, systemSize);
}

__host__ void
kodes::StiffnessBalancer::destroyStub(kodes::StiffnessBalancer* hostStub)
{
    ::delete hostStub;
}
