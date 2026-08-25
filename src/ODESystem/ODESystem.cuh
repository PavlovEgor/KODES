
#ifndef KODES_ODE_SYSTEM
#define KODES_ODE_SYSTEM
#pragma once

#include "basicTypes.cuh"

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
        const scalar t, const scalar parameter, const scalar* y, scalar* dydt
    ) const = 0;

    __device__ virtual void
    jacobian
    (
        const scalar t, const scalar parameter, const scalar* y, scalar* dfdt, scalar* dfdy
    ) const = 0;
};
}

#endif
