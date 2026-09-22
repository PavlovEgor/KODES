#include "SeulexDeviceResources.cuh"
#include "seulex_constants.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::SeulexDeviceResources)

__host__ void
kodes::SeulexDeviceResources::allocate()
{
    DeviceResources::allocate();

    CUDA_CHECK(cudaMalloc(&table_, 12 * size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&dfdy_, size_t(systemSize_) * systemSize_ * scratchSize_ * sizeof(scalar)));

    CUDA_CHECK(cudaMalloc(&y0_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&ySequence_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&scale_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&dy_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&yTemp_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&dydt_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));

    // One region per block. Allocated whether or not the plan decided to work
    // in shared memory instead: the decision is the kernel's, and a region
    // small enough to have fitted in shared memory is small enough that the
    // spare copy does not matter.
    CUDA_CHECK(cudaMalloc(&blockScratch_, size_t(blocks_) * blockScratchBytes(systemSize_)));

    uploadSeulexConstants();
}

__host__ void
kodes::SeulexDeviceResources::deallocate()
{
    CUDA_CHECK(cudaFree(table_));
    CUDA_CHECK(cudaFree(dfdy_));

    CUDA_CHECK(cudaFree(y0_));
    CUDA_CHECK(cudaFree(ySequence_));
    CUDA_CHECK(cudaFree(scale_));
    CUDA_CHECK(cudaFree(dy_));
    CUDA_CHECK(cudaFree(yTemp_));
    CUDA_CHECK(cudaFree(dydt_));

    CUDA_CHECK(cudaFree(blockScratch_));

    DeviceResources::deallocate();
}
