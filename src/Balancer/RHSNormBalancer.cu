#include "RHSNormBalancer.cuh"
#include "BalancerFactory.cuh"

__host__ kodes::RHSNormBalancer*
kodes::RHSNormBalancer::create
(
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    kodes::RHSNormBalancer* hostStub
)
{
    return kodes::createBalancer(batchSize, scratchSize, systemSize, hostStub);
}

__host__ void
kodes::RHSNormBalancer::destroy
(
    kodes::RHSNormBalancer* devBalancer,
    kodes::RHSNormBalancer* hostStub
)
{
    kodes::destroyBalancer(devBalancer, hostStub);
}

__host__ kodes::RHSNormBalancer*
kodes::RHSNormBalancer::createStub
(
    const label batchSize,
    const label scratchSize,
    const label systemSize
)
{
    return ::new RHSNormBalancer(batchSize, scratchSize, systemSize);
}

__host__ void
kodes::RHSNormBalancer::destroyStub(kodes::RHSNormBalancer* hostStub)
{
    ::delete hostStub;
}
