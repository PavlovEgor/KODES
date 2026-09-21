
#ifndef KODES_ODE_SYSTEM
#define KODES_ODE_SYSTEM
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
        const scalar t, const scalar parameter, const scalar* y, scalar* dydt
    ) const = 0;

    // df/dy only: the systems this library integrates are autonomous, so the
    // df/dt this used to take alongside it was a vector every caller allocated
    // per thread, passed in, and never read. pyJac never wrote it either.
    __device__ virtual void
    jacobian
    (
        const scalar t, const scalar parameter, const scalar* y, scalar* dfdy
    ) const = 0;
};
}

#endif
