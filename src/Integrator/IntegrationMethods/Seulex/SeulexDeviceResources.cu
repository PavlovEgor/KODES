#include "SeulexDeviceResources.cuh"
#include "seulex_constants.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::SeulexDeviceResources)

__host__ void
kodes::SeulexDeviceResources::allocate()
{
    DeviceResources::allocate();

    table_.allocate(scratchSize_);
    dfdy_.allocate(scratchSize_);
    a_.allocate(scratchSize_);

    pivotIndices_.allocate(scratchSize_);
    vv_.allocate(scratchSize_);

    // indexed by the extrapolation order, not by the component
    dtOpt_.allocate(scratchSize_);
    temp_.allocate(scratchSize_);

    y0_.allocate(scratchSize_);
    ySequence_.allocate(scratchSize_);
    scale_.allocate(scratchSize_);
    dy_.allocate(scratchSize_);
    yTemp_.allocate(scratchSize_);
    dydt_.allocate(scratchSize_);

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
