
#ifndef ODESystem1_H
#define ODESystem1_H

#pragma once

#include "basic_types.cuh"

namespace kodes 
{
class ODESystem
{

public:
    __device__ __host__
    ODESystem() {}

    __device__ __host__
    virtual ~ODESystem() = default;

    __device__ virtual void 
    derivatives
    (
        const scalar x, const scalar param, const scalar* y, scalar* dydx
    ) const = 0;

    __device__ virtual void 
    jacobian
    (
        const scalar x, const scalar param, const scalar* y, scalar* dfdx, scalar* dfdy
    ) const = 0;
};
}

#endif
