
// GRIMESHSystem.h
#ifndef GRIMESHSystem1_H
#define GRIMESHSystem1_H

#pragma once

#include "ODESystem.cuh"
#include "dydt.cuh"
#include "jacob.cuh"
#include "gpu_memory.cuh"
#include "mechanism.cuh"

// __constant__ scalar pressure_  = 101325.0;

namespace kodes 
{
class GRIMESHSystem
    : public ODESystem
{
    mechanism_memory* device_memory;

public:
    __device__ __host__
    GRIMESHSystem(mechanism_memory *d_mem) : device_memory(d_mem), ODESystem(NN+1)  {}
    __device__ __host__
    virtual ~GRIMESHSystem() = default;

    __host__ static
    GRIMESHSystem* createGPU(mechanism_memory *d_mem);

    __host__ static void
    destroyGPU(GRIMESHSystem* system);

    __device__ static void* operator new(size_t size, void* ptr) {
        return ptr;
    }

    __device__ void 
    derivatives
    (
        const scalar x, const scalar* y, scalar* dydx
    ) const override;

    __device__ void 
    jacobian
    (
        const scalar x, const scalar* y, scalar* dfdx, scalar* dfdy
    ) const override;
};
}

#endif
