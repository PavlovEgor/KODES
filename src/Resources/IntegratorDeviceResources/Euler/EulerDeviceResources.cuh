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
    EulerDeviceResources(const label batchSize, const label systemSize, const label parameterSize)
        : AdaptiveDeviceResources(batchSize, systemSize, parameterSize) {}

    __device__ __host__
    ~EulerDeviceResources() = default;
    
    __host__ static EulerDeviceResources* 
    create(const label ensembleSize, const label systemSize, const label parameterSize, EulerDeviceResources* hostStub);

    __host__ static void
    destroy(EulerDeviceResources* devRes, EulerDeviceResources* hostStub);

    __device__ scalar* __restrict__
    err() { return err_; }
};

}

#endif
