#include "TemperatureBalancer.cuh"

__global__ void
constructTemperatureBalancer(kodes::TemperatureBalancer* devBalancer, const label batchSize)
{
    new (devBalancer) kodes::TemperatureBalancer(batchSize);
}

__global__ void
destructTemperatureBalancer(kodes::TemperatureBalancer* devBalancer)
{
    devBalancer->~TemperatureBalancer();
}

__host__ kodes::TemperatureBalancer*
kodes::TemperatureBalancer::create(const label batchSize, kodes::TemperatureBalancer* hostStub)
{
    if (!hostStub)
    {
        fprintf(stderr, "TemperatureBalancer::create error at %s:%d: hostStub is null\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    if (batchSize <= 0)
    {
        fprintf(stderr, "TemperatureBalancer::create error at %s:%d: batchSize <= 0\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    TemperatureBalancer* devPtr;
    CUDA_CHECK(cudaMalloc(&devPtr, sizeof(TemperatureBalancer)));

    hostStub->allocate(batchSize);

    CUDA_CHECK(cudaMemcpy(devPtr, hostStub, sizeof(TemperatureBalancer), cudaMemcpyHostToDevice));

    constructTemperatureBalancer<<<1, 1>>>(devPtr, batchSize);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    return devPtr;
}

__host__ void
kodes::TemperatureBalancer::destroy(kodes::TemperatureBalancer* devBalancer, kodes::TemperatureBalancer* hostStub)
{
    if (hostStub)
    {
        hostStub->deallocate();
    }

    if (devBalancer)
    {
        destructTemperatureBalancer<<<1, 1>>>(devBalancer);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(devBalancer));
    }
}
