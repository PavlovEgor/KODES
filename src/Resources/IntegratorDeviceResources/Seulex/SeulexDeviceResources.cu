#include "SeulexDeviceResources.cuh"
#include "SeulexConstants.cuh"

__global__ void
constructSeulexDeviceResources(kodes::SeulexDeviceResources* devRes, const label batchSize, const label scratchSize, const label systemSize, const label parameterSize)
{
    new (devRes) kodes::SeulexDeviceResources(batchSize, scratchSize, systemSize, parameterSize);
}

__global__ void
destructSeulexDeviceResources(kodes::SeulexDeviceResources* devRes) {
    devRes->~SeulexDeviceResources();
}

__host__  kodes::SeulexDeviceResources*
kodes::SeulexDeviceResources::create(const label batchSize, const label scratchSize, const label systemSize, const label parameterSize, kodes::SeulexDeviceResources* hostStub) {
    if (!hostStub)
    {
        fprintf(stderr, "SeulexDeviceResources::create error at %s:%d: hostStub is null\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    if (batchSize <= 0 || scratchSize <= 0)
    {
        fprintf(stderr, "SeulexDeviceResources::create error at %s:%d: batchSize/scratchSize <= 0\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    SeulexDeviceResources* devPtr;

    CUDA_CHECK(cudaMalloc(&devPtr, sizeof(SeulexDeviceResources)));

    CUDA_CHECK(cudaMalloc(&hostStub->vectors, size_t(systemSize) * batchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->parameters, size_t(parameterSize) * batchSize * sizeof(scalar)));

    const label orderSize = orderStorage(systemSize);

    CUDA_CHECK(cudaMalloc(&hostStub->currentVector_, size_t(systemSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->currentParameters_, size_t(parameterSize) * scratchSize * sizeof(scalar)));

    CUDA_CHECK(cudaMalloc(&hostStub->table_, 12 * size_t(systemSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->dfdt_, size_t(systemSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->dfdy_, size_t(systemSize) * systemSize * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->a_, size_t(systemSize) * systemSize * scratchSize * sizeof(scalar)));

    CUDA_CHECK(cudaMalloc(&hostStub->pivotIndices_, size_t(systemSize) * scratchSize * sizeof(label)));

    CUDA_CHECK(cudaMalloc(&hostStub->dtOpt_, size_t(orderSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->temp_, size_t(orderSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->y0_, size_t(systemSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->ySequence_, size_t(systemSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->scale_, size_t(systemSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->dy_, size_t(systemSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->yTemp_, size_t(systemSize) * scratchSize * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&hostStub->dydt_, size_t(systemSize) * scratchSize * sizeof(scalar)));

    hostStub->allocate(batchSize);

    CUDA_CHECK(cudaMemcpy(devPtr, hostStub, sizeof(SeulexDeviceResources), cudaMemcpyHostToDevice));

    constructSeulexDeviceResources<<<1, 1>>>(devPtr, batchSize, scratchSize, systemSize, parameterSize);

    // The gpu time factors for the major parts of the algorithm
    const scalar gpuFunc = 2, gpuJac = 40, gpuLU = 17, gpuSolve = 1;

    label host_nSeq_[iMaxx_];
    scalar host_gpu_[iMaxx_];
    scalar host_coeff_[iMaxx_ * iMaxx_];

    host_nSeq_[0] = 2;
    host_nSeq_[1] = 3;

    for (int i=2; i<iMaxx_; i++)
    {
        host_nSeq_[i] = 2 * host_nSeq_[i-2];
    }
    host_gpu_[0] = gpuJac + gpuLU + host_nSeq_[0]*(gpuFunc + gpuSolve);

    for (int k=0; k<kMaxx_; k++)
    {
        host_gpu_[k+1] = host_gpu_[k] + (host_nSeq_[k+1]-1)*(gpuFunc + gpuSolve) + gpuLU;
    }

    // Set the extrapolation coefficients array
    for (int k=0; k<iMaxx_; k++)
    {
        for (int l=0; l<k; l++)
        {
            scalar ratio = scalar(host_nSeq_[k])/host_nSeq_[l];
            host_coeff_[k + l*iMaxx_] = 1/(ratio - 1);
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(nSeq_, host_nSeq_, iMaxx_ * sizeof(label)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu_, host_gpu_, iMaxx_ * sizeof(scalar)));
    CUDA_CHECK(cudaMemcpyToSymbol(coeff_, host_coeff_, iMaxx_ * iMaxx_ * sizeof(scalar)));

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return devPtr;
}

__host__  void
kodes::SeulexDeviceResources::destroy(kodes::SeulexDeviceResources* devRes, kodes::SeulexDeviceResources* hostStub) {
    if (hostStub) {

        CUDA_CHECK(cudaFree(hostStub->vectors));
        CUDA_CHECK(cudaFree(hostStub->parameters));

        CUDA_CHECK(cudaFree(hostStub->currentVector_));
        CUDA_CHECK(cudaFree(hostStub->currentParameters_));

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