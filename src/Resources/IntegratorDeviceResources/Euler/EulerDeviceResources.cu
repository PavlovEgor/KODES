#include "EulerDeviceResources.cuh"

__global__ void
constructEulerDeviceResources(kodes::EulerDeviceResources* devRes, const label batchSize, const label scratchSize, const label systemSize, const label parameterSize)
{
    new (devRes) kodes::EulerDeviceResources(batchSize, scratchSize, systemSize, parameterSize);
}

__global__ void
destructEulerDeviceResources(kodes::EulerDeviceResources* devRes) {
    devRes->~EulerDeviceResources();
}

__host__  kodes::EulerDeviceResources*
kodes::EulerDeviceResources::create(const label batchSize, const label scratchSize, const label systemSize, const label parameterSize, kodes::EulerDeviceResources* hostStub) {
    if (!hostStub)
    {
        fprintf(stderr, "EulerDeviceResources::create error at %s:%d: hostStub is null\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    EulerDeviceResources* devPtr;

    CUDA_CHECK(cudaMalloc(&devPtr, sizeof(EulerDeviceResources)));

    CUDA_CHECK(cudaMalloc(&hostStub->vectors, size_t(systemSize) * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->parameters, size_t(parameterSize) * batchSize * sizeof(scalar)));

    CUDA_CHECK(cudaMalloc(&hostStub->y_, size_t(systemSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->param_, size_t(parameterSize) * scratchSize * sizeof(scalar)));

    CUDA_CHECK(cudaMalloc(&hostStub->yTemp_, size_t(systemSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->dydx0_, size_t(systemSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->err_, size_t(systemSize) * scratchSize * sizeof(scalar)));

    hostStub->allocate(batchSize);

    CUDA_CHECK(cudaMemcpy(devPtr, hostStub, sizeof(EulerDeviceResources), cudaMemcpyHostToDevice));

    constructEulerDeviceResources<<<1, 1>>>(devPtr, batchSize, scratchSize, systemSize, parameterSize);

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    return devPtr;
}

__host__  void
kodes::EulerDeviceResources::destroy(kodes::EulerDeviceResources* devRes, kodes::EulerDeviceResources* hostStub) {
    if (hostStub) {

        CUDA_CHECK(cudaFree(hostStub->vectors));
        CUDA_CHECK(cudaFree(hostStub->parameters));

        CUDA_CHECK(cudaFree(hostStub->y_));
        CUDA_CHECK(cudaFree(hostStub->param_));

        CUDA_CHECK(cudaFree(hostStub->yTemp_));
        CUDA_CHECK(cudaFree(hostStub->dydx0_));
        CUDA_CHECK(cudaFree(hostStub->err_));

        hostStub->deallocate();

        if (!devRes)
        {
            fprintf(stderr, "EulerDeviceResources::destroy error at %s:%d: devRes is null\n", __FILE__, __LINE__);
            std::exit(EXIT_FAILURE);
        }

        destructEulerDeviceResources<<<1, 1>>>(devRes);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(devRes));
    }
}
