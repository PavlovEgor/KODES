
#ifndef ODESystem1_H
#define ODESystem1_H

#pragma once

#include "basic_types.cuh"

namespace kodes 
{
class ODESystem
{
    label sizeOfSystem_;

    // __global__ void 
    // constructGPU(kodes::HIRESSystem* system); 
    // {
    //     new (system) kodes::HIRESSystem(8);
    // }

public:
    __device__ __host__
    ODESystem(const label sizeOfSystem) : sizeOfSystem_(sizeOfSystem) {}

    __device__ __host__
    virtual ~ODESystem() = default;

    __device__  __host__ label 
    nEqns() const {return sizeOfSystem_;}

    __device__ virtual void 
    derivatives
    (
        const scalar x, const scalar* y, scalar* dydx
    ) const = 0;

    __device__ virtual void 
    jacobian
    (
        const scalar x, const scalar* y, scalar* dfdx, scalar* dfdy
    ) const = 0;
};
}

#endif
