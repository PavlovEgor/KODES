#include "SeulexDeviceResources.cuh"

__global__ void 
constructSeulexDeviceResources(kodes::SeulexDeviceResources* devRes, const label batchSize, const label systemSize, const label parameterSize)
{
    new (devRes) kodes::SeulexDeviceResources(batchSize, systemSize, parameterSize);
}

__global__ void 
destructSeulexDeviceResources(kodes::SeulexDeviceResources* devRes) {
    devRes->~SeulexDeviceResources();
}

__host__  kodes::SeulexDeviceResources* 
kodes::SeulexDeviceResources::create(const label batchSize, const label systemSize, const label parameterSize, kodes::SeulexDeviceResources* hostStub) {
    if (!hostStub)
    {
        fprintf(stderr, "SeulexDeviceResources::create error at %s:%d: hostStub is null\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    SeulexDeviceResources* devPtr;
    
    CUDA_CHECK(cudaMalloc(&devPtr, sizeof(SeulexDeviceResources)));

    CUDA_CHECK(cudaMalloc(&hostStub->vectors, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->parameters, parameterSize * batchSize * sizeof(scalar)));

    CUDA_CHECK(cudaMalloc(&hostStub->table_, 12 * systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->dfdt_, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->dfdy_, systemSize * systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->a_, systemSize * systemSize * batchSize * sizeof(scalar)));

    CUDA_CHECK(cudaMalloc(&hostStub->pivotIndices_, systemSize * batchSize * sizeof(label)));

    CUDA_CHECK(cudaMalloc(&hostStub->dtOpt_, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->temp_, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->y0_, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->ySequence_, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->scale_, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->dy_, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->yTemp_, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->dydt_, systemSize * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->y_, systemSize * batchSize * sizeof(scalar)));

    hostStub->allocate(batchSize);

    CUDA_CHECK(cudaMemcpy(devPtr, hostStub, sizeof(SeulexDeviceResources), cudaMemcpyHostToDevice));
    
    constructSeulexDeviceResources<<<1, 1>>>(devPtr, batchSize, systemSize, parameterSize);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return devPtr;
}

__host__  void
kodes::SeulexDeviceResources::destroy(kodes::SeulexDeviceResources* devRes, kodes::SeulexDeviceResources* hostStub) {
    if (hostStub) {

        CUDA_CHECK(cudaFree(hostStub->vectors));
        CUDA_CHECK(cudaFree(hostStub->parameters));

        CUDA_CHECK(cudaFree(hostStub->table_));
        CUDA_CHECK(cudaFree(hostStub->dfdt_));
        CUDA_CHECK(cudaFree(hostStub->dfdy_));
        CUDA_CHECK(cudaFree(hostStub->a_));

        CUDA_CHECK(cudaFree(hostStub->pivotIndices_));

        CUDA_CHECK(cudaFree(hostStub->dtOpt_));
        CUDA_CHECK(cudaFree(hostStub->temp_));
        CUDA_CHECK(cudaFree(hostStub->y0_));
        CUDA_CHECK(cudaFree(hostStub->ySequence_));
        CUDA_CHECK(cudaFree(hostStub->scale_));
        CUDA_CHECK(cudaFree(hostStub->dy_));
        CUDA_CHECK(cudaFree(hostStub->yTemp_));
        CUDA_CHECK(cudaFree(hostStub->dydt_));
        CUDA_CHECK(cudaFree(hostStub->y_));

        hostStub->deallocate();

        if (!devRes)
        {
            fprintf(stderr, "SeulexDeviceResources::destroy error at %s:%d: devRes is null\n", __FILE__, __LINE__);
            std::exit(EXIT_FAILURE);
        }

        destructSeulexDeviceResources<<<1, 1>>>(devRes);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(devRes));
    }
}