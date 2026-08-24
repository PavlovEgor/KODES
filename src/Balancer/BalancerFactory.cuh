#ifndef KODES_BALANCER_FACTORY
#define KODES_BALANCER_FACTORY

#pragma once

#include "Balancer.cuh"

// Construction and destruction of a Balancer subclass, the same four steps for
// every one of them.
//
// Included only by the .cu of a subclass, never by a caller: it launches
// kernels, and the whole reason the subclasses hand out create/destroy at all
// is that a caller must not be made to emit them. key() is a device-only
// virtual, so the vtable of a host side object can only be produced by a
// compiler that invents a host stub for one - nvcc does, nvc++ -cuda does not.
//
// Every subclass therefore takes the same three constructor arguments and fixes
// its key count itself.

namespace kodes
{

template<class BalancerType>
__global__ void constructBalancer
(
    BalancerType* devBalancer,
    const label batchSize,
    const label scratchSize,
    const label systemSize
)
{
    new (devBalancer) BalancerType(batchSize, scratchSize, systemSize);
}

template<class BalancerType>
__global__ void destructBalancer(BalancerType* devBalancer)
{
    devBalancer->~BalancerType();
}

template<class BalancerType>
__host__ BalancerType* createBalancer
(
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    BalancerType* hostStub
)
{
    if (!hostStub)
    {
        fprintf(stderr, "kodes::createBalancer error at %s:%d: hostStub is null\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    if (batchSize <= 0 || scratchSize <= 0 || systemSize <= 0)
    {
        fprintf(stderr, "kodes::createBalancer error at %s:%d: non-positive batchSize/scratchSize/systemSize\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    BalancerType* devPtr;
    CUDA_CHECK(cudaMalloc(&devPtr, sizeof(BalancerType)));

    hostStub->allocate();

    // The stub is copied over first so that the device object inherits the
    // addresses it just allocated, then constructed in place so that its vtable
    // is the device one. That order is why the constructor must not touch a
    // buffer pointer - see Balancer.
    CUDA_CHECK(cudaMemcpy(devPtr, hostStub, sizeof(BalancerType), cudaMemcpyHostToDevice));

    constructBalancer<BalancerType><<<1, 1>>>(devPtr, batchSize, scratchSize, systemSize);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    return devPtr;
}

template<class BalancerType>
__host__ void destroyBalancer(BalancerType* devBalancer, BalancerType* hostStub)
{
    if (hostStub)
    {
        hostStub->deallocate();
    }

    if (devBalancer)
    {
        destructBalancer<BalancerType><<<1, 1>>>(devBalancer);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(devBalancer));
    }
}

}

#endif
