#include "SeulexDeviceResources.cuh"
#include "seulex_constants.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::SeulexDeviceResources)

__host__ void
kodes::SeulexDeviceResources::allocate()
{
    DeviceResources::allocate();

    const label orderSize = orderStorage(systemSize_);

    CUDA_CHECK(cudaMalloc(&table_, 12 * size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&dfdt_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&dfdy_, size_t(systemSize_) * systemSize_ * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&a_, size_t(systemSize_) * systemSize_ * scratchSize_ * sizeof(scalar)));

    CUDA_CHECK(cudaMalloc(&pivotIndices_, size_t(systemSize_) * scratchSize_ * sizeof(label)));

    // indexed by the extrapolation order, not by the component
    CUDA_CHECK(cudaMalloc(&dtOpt_, size_t(orderSize) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&temp_, size_t(orderSize) * scratchSize_ * sizeof(scalar)));

    CUDA_CHECK(cudaMalloc(&y0_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&ySequence_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&scale_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&dy_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&yTemp_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&dydt_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));

    uploadSeulexConstants();
}

__host__ void
kodes::SeulexDeviceResources::deallocate()
{
    CUDA_CHECK(cudaFree(table_));
    CUDA_CHECK(cudaFree(dfdt_));
    CUDA_CHECK(cudaFree(dfdy_));
    CUDA_CHECK(cudaFree(a_));

    CUDA_CHECK(cudaFree(pivotIndices_));

    CUDA_CHECK(cudaFree(dtOpt_));
    CUDA_CHECK(cudaFree(temp_));

    CUDA_CHECK(cudaFree(y0_));
    CUDA_CHECK(cudaFree(ySequence_));
    CUDA_CHECK(cudaFree(scale_));
    CUDA_CHECK(cudaFree(dy_));
    CUDA_CHECK(cudaFree(yTemp_));
    CUDA_CHECK(cudaFree(dydt_));

    DeviceResources::deallocate();
}
