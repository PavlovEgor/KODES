#include "AdaptiveDeviceResources.cuh"

// No KODES_DEFINE_DEVICE_OBJECT here: an intermediate of the family is never a
// row of a type table. See the head of DeviceResources.cuh.

__host__ void
kodes::AdaptiveDeviceResources::allocate()
{
    DeviceResources::allocate();

    yTemp_.allocate(scratchSize_);
    dydt0_.allocate(scratchSize_);
}

__host__ void
kodes::AdaptiveDeviceResources::deallocate()
{
    yTemp_.deallocate();
    dydt0_.deallocate();

    DeviceResources::deallocate();
}
