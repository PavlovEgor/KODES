
// H2O2System.h
#ifndef H2O2System1_H
#define H2O2System1_H

#pragma once

#include "ODESystem.cuh"
#include "dydt.cuh"
#include "jacob.cuh"
#include "gpu_memory.cuh"
#include "mechanism.cuh"

// __constant__ scalar pressure_  = 101325.0;

namespace kodes 
{
class H2O2System
    : public ODESystem
{
    mechanism_memory* device_memory;

public:
    __device__ __host__
    H2O2System(mechanism_memory *d_mem) : device_memory(d_mem), ODESystem(NN+1)  {}
    __device__ __host__
    virtual ~H2O2System() = default;

    __host__ static
    H2O2System* createGPU(mechanism_memory *d_mem);

    __host__ static void
    destroyGPU(H2O2System* system);

    __device__ static void* operator new(size_t size, void* ptr) {
        return ptr;
    }

    __device__  void 
    derivatives
    (
        const scalar x, const scalar param, const scalar* y, scalar* dydx
    ) const override;

    __device__  void 
    jacobian
    (
        const scalar x, const scalar param, const scalar* y, scalar* dfdx, scalar* dfdy
    ) const override;
};
}

#endif
