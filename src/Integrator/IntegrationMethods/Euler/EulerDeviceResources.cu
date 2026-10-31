#include "EulerDeviceResources.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::EulerDeviceResources)

__host__ void
kodes::EulerDeviceResources::allocate()
{
    AdaptiveDeviceResources::allocate();

    err_.allocate(scratchSize_);
}

__host__ void
kodes::EulerDeviceResources::deallocate()
{
    err_.deallocate();

    AdaptiveDeviceResources::deallocate();
}
