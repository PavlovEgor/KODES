#ifndef KODES_ADAPTIVE_DEVICE_RESOURCES
#define KODES_ADAPTIVE_DEVICE_RESOURCES
#include "DeviceResources.cuh"

namespace kodes
{

// The scratch a method that takes *trial* steps needs whatever else it does:
// the state a rejected step has to be restored from, and the right hand side
// at the top of it. IntegrationMethod::adaptiveStep() casts down to this, so
// every method with usesAdaptiveStep() set brings resources derived from here.
//
// An intermediate of the family, not a row of any table - see the head of
// DeviceResources.cuh - so no KODES_DECLARE_DEVICE_OBJECT. The placement
// operator new is inherited; the four statics belong to the leaves.
class AdaptiveDeviceResources
    :
    public DeviceResources
{
protected:
    scalar* yTemp_;
    scalar* dydt0_;

public:

    __device__ __host__
    AdaptiveDeviceResources
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : DeviceResources(batchSize, scratchSize, blocks, systemSize, parameterSize) {}

    __device__ __host__
    ~AdaptiveDeviceResources() = default;

    __host__ void allocate();

    __host__ void deallocate();

    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        return DeviceResources::scratchBytesPerThread(systemSize, parameterSize)
             + 2 * size_t(systemSize) * sizeof(scalar);
    }

    __device__ scalar* __restrict__
    yTemp() { return yTemp_; }

    __device__ scalar* __restrict__
    dydt0() { return dydt0_; }

};

}

#endif
