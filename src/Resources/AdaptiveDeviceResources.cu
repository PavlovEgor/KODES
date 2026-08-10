#include "AdaptiveDeviceResources.cuh"


__global__ void 
constructAdaptiveDeviceResources(kodes::AdaptiveDeviceResources* devRes, const label batchSize, const label systemSize, const label parameterSize)
{
    new (devRes) kodes::AdaptiveDeviceResources(batchSize, systemSize, parameterSize);
}

__global__ void 
destructAdaptiveDeviceResources(kodes::AdaptiveDeviceResources* devRes) {
    devRes->~AdaptiveDeviceResources();
}

__host__  kodes::AdaptiveDeviceResources* 
kodes::AdaptiveDeviceResources::create(const label batchSize, const label systemSize, const label parameterSize, kodes::AdaptiveDeviceResources* hostStub) {
    if (!hostStub)
    {
        fprintf(stderr, "AdaptiveDeviceResources::create error at %s:%d: hostStub is null\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    AdaptiveDeviceResources* devPtr;
    
    CUDA_CHECK(cudaMalloc(&devPtr, sizeof(AdaptiveDeviceResources)));

    CUDA_CHECK(cudaMalloc(&hostStub->vectors, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->parameters, parameterSize * batchSize * sizeof(scalar)));

    CUDA_CHECK(cudaMalloc(&hostStub->yTemp_, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->dydx0_, systemSize * batchSize * sizeof(scalar)));

    hostStub->allocate(batchSize);

    CUDA_CHECK(cudaMemcpy(devPtr, hostStub, sizeof(AdaptiveDeviceResources), cudaMemcpyHostToDevice));
    
    constructAdaptiveDeviceResources<<<1, 1>>>(devPtr, batchSize, systemSize, parameterSize);

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return devPtr;
}

__host__  void
kodes::AdaptiveDeviceResources::destroy(kodes::AdaptiveDeviceResources* devRes, kodes::AdaptiveDeviceResources* hostStub) {
    if (hostStub) {

        CUDA_CHECK(cudaFree(hostStub->vectors));
        CUDA_CHECK(cudaFree(hostStub->parameters));

        CUDA_CHECK(cudaFree(devRes->yTemp_));
        CUDA_CHECK(cudaFree(devRes->dydx0_));

        hostStub->deallocate();

        if (!devRes)
        {
            fprintf(stderr, "AdaptiveDeviceResources::destroy error at %s:%d: devRes is null\n", __FILE__, __LINE__);
            std::exit(EXIT_FAILURE);
        }

        destructAdaptiveDeviceResources<<<1, 1>>>(devRes);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(devRes));
    }
}