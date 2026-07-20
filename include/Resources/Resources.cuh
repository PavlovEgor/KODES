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
    label sizeOfSystem_;
    label numOfParameters_;

public:
    __device__ __host__
    Resources(const label numOfSystems, const label sizeOfSystem, const label numOfParameters) 
        : numOfSystems_(numOfSystems), sizeOfSystem_(sizeOfSystem), numOfParameters_(numOfParameters) {}
        
    __device__ __host__
    virtual ~Resources() = default;

    __device__ __host__ label numOfSystems() { return numOfSystems_; }
    __device__ __host__ label sizeOfSystem() { return sizeOfSystem_; }
    __device__ __host__ label numOfParameters() { return numOfParameters_; }
};
}
#endif
