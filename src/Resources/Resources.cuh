#ifndef KODES_RESOURCES_CUH
#define KODES_RESOURCES_CUH

#include "basic_types.cuh"
#include <stdio.h>

namespace kodes 
{
class Resources
{
protected:
    label numOfSystems_;
    label systemSize_;
    label parameterSize_;

public:
    __device__ __host__
    Resources(const label numOfSystems, const label systemSize, const label parameterSize)
        : numOfSystems_(numOfSystems), systemSize_(systemSize), parameterSize_(parameterSize)
        {}

    __device__ __host__
    virtual ~Resources() = default;

    __device__ __host__ label numOfSystems() { return numOfSystems_; }
    __device__ __host__ label systemSize() { return systemSize_; }
    __device__ __host__ label parameterSize() { return parameterSize_; }
};
}
#endif
