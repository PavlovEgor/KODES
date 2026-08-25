#ifndef KODES_ADAPTIVE_DEVICE_RESOURCES
#define KODES_ADAPTIVE_DEVICE_RESOURCES
#include "DeviceResources.cuh"

namespace kodes 
{

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
        const label systemSize,
        const label parameterSize
    )
        : DeviceResources(batchSize, scratchSize, systemSize, parameterSize) {}

    __device__ __host__
    ~AdaptiveDeviceResources() = default;

    KODES_DECLARE_DEVICE_OBJECT(AdaptiveDeviceResources)

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
