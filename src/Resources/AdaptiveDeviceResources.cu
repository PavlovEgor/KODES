#include "AdaptiveDeviceResources.cuh"

// No KODES_DEFINE_DEVICE_OBJECT here: an intermediate of the family is never a
// row of a type table. See the head of DeviceResources.cuh.

__host__ void
kodes::AdaptiveDeviceResources::allocate()
{
    DeviceResources::allocate();

    CUDA_CHECK(cudaMalloc(&yTemp_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
    CUDA_CHECK(cudaMalloc(&dydt0_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
}

__host__ void
kodes::AdaptiveDeviceResources::deallocate()
{
    CUDA_CHECK(cudaFree(yTemp_));
    CUDA_CHECK(cudaFree(dydt0_));

    DeviceResources::deallocate();
}
