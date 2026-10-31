#include "SeulexDeviceResources.cuh"
#include "seulex_constants.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::SeulexDeviceResources)

__host__ void
kodes::SeulexDeviceResources::allocate()
{
    DeviceResources::allocate();

    const label orderSize = orderStorage(systemSize_);

    table_.allocate(systemSize_, scratchSize_);
    dfdy_.allocate(systemSize_, scratchSize_);
    a_.allocate(systemSize_, scratchSize_);

    pivotIndices_.allocate(systemSize_, scratchSize_);
    vv_.allocate(systemSize_, scratchSize_);

    // indexed by the extrapolation order, not by the component
    dtOpt_.allocate(orderSize, scratchSize_);
    temp_.allocate(orderSize, scratchSize_);

    y0_.allocate(systemSize_, scratchSize_);
    ySequence_.allocate(systemSize_, scratchSize_);
    scale_.allocate(systemSize_, scratchSize_);
    dy_.allocate(systemSize_, scratchSize_);
    yTemp_.allocate(systemSize_, scratchSize_);
    dydt_.allocate(systemSize_, scratchSize_);

    // one entry per thread: step control that survives a step, not a system
    kTarg_.allocate(scratchSize_);
    theta_.allocate(scratchSize_);

    uploadSeulexConstants();
}

__host__ void
kodes::SeulexDeviceResources::deallocate()
{
    table_.deallocate();
    dfdy_.deallocate();
    a_.deallocate();

    pivotIndices_.deallocate();
    vv_.deallocate();

    dtOpt_.deallocate();
    temp_.deallocate();

    y0_.deallocate();
    ySequence_.deallocate();
    scale_.deallocate();
    dy_.deallocate();
    yTemp_.deallocate();
    dydt_.deallocate();

    kTarg_.deallocate();
    theta_.deallocate();

    DeviceResources::deallocate();
}
