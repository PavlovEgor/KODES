#ifndef EULERDEVRES 
#define EULERDEVRES

#include "AdaptiveDeviceResources.cuh"

namespace kodes 
{

class EulerDeviceResources 
    :
    public AdaptiveDeviceResources
{
protected:
    scalar* err_;

public:

    __device__ __host__
    EulerDeviceResources
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        const label parameterSize
    )
        : AdaptiveDeviceResources(batchSize, scratchSize, systemSize, parameterSize) {}

    __device__ __host__
    ~EulerDeviceResources() = default;

    KODES_DECLARE_DEVICE_OBJECT(EulerDeviceResources)

    __host__ void allocate();

    __host__ void deallocate();

    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        return AdaptiveDeviceResources::scratchBytesPerThread(systemSize, parameterSize)
             + size_t(systemSize) * sizeof(scalar);
    }

    __device__ scalar* __restrict__
    err() { return err_; }
};

}

#endif
