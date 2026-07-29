#ifndef KODES_RESOURCES_CUH
#define KODES_RESOURCES_CUH

#include "basic_types.cuh"
#include <stdio.h>

namespace kodes 
{
class Resources
{
protected:
    label ensembleSize_;
    label systemSize_;
    label parameterSize_;

public:
    __device__ __host__
    Resources(const label ensembleSize, const label systemSize, const label parameterSize)
        : ensembleSize_(ensembleSize), systemSize_(systemSize), parameterSize_(parameterSize)
        {}

    __device__ __host__
    virtual ~Resources() = default;

    __device__ __host__ label ensembleSize() { return ensembleSize_; }
    __device__ __host__ label systemSize() { return systemSize_; }
    __device__ __host__ label parameterSize() { return parameterSize_; }
};
}
#endif
