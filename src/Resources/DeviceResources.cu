#include "DeviceResources.cuh"

// No KODES_DEFINE_DEVICE_OBJECT here: this class is never a row of a type
// table, only its leaves are. See the head of DeviceResources.cuh.

__host__ void
kodes::DeviceResources::allocate()
{
    checkSystemDimensions(systemSize_, parameterSize_, "DeviceResources::allocate");

    // state space: the whole batch, one slot per system
    vectors.allocate(ensembleSize_);
    parameters.allocate(ensembleSize_);

    // scratch space: one slot per resident thread
    currentVector_.allocate(scratchSize_);
    currentParameters_.allocate(scratchSize_);

    StepState::allocate(scratchSize_);
}

__host__ void
kodes::DeviceResources::deallocate()
{
    vectors.deallocate();
    parameters.deallocate();

    currentVector_.deallocate();
    currentParameters_.deallocate();

    StepState::deallocate();
}

__device__ void
kodes::DeviceResources::printVectori(const label i) const
{
    for (label j = 0; j < kSystemSize; ++j) {
        printf("%0.2f ", vectors(i, j));
    }
    printf("\n");
}
