#include "EulerDeviceResources.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::EulerDeviceResources)

__host__ void
kodes::EulerDeviceResources::allocate()
{
    AdaptiveDeviceResources::allocate();

    CUDA_CHECK(cudaMalloc(&err_, size_t(systemSize_) * scratchSize_ * sizeof(scalar)));
}

__host__ void
kodes::EulerDeviceResources::deallocate()
{
    CUDA_CHECK(cudaFree(err_));

    AdaptiveDeviceResources::deallocate();
}
