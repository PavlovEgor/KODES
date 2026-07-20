
#ifndef pyJacSystem_H
#define pyJacSystem_H

#pragma once

#include "ODESystem.cuh"
#include "dydt.cuh"
#include "jacob.cuh"
#include "gpu_memory.cuh"
#include "mechanism.cuh"

namespace kodes 
{
class pyJacSystem
    : public ODESystem
{
    mechanism_memory* device_memory;

public:
    __device__ __host__
    pyJacSystem(mechanism_memory *d_mem) : ODESystem(), device_memory(d_mem) {}

    __device__ __host__
    virtual ~pyJacSystem() = default;

    __host__ static
    pyJacSystem* createGPU(mechanism_memory *d_mem);

    __host__ static void
    destroyGPU(pyJacSystem* system);

    __device__ static void* operator new(size_t size, void* ptr) {
        return ptr;
    }

    __device__ void 
    derivatives
    (
        const scalar x, const scalar rho, const scalar* y, scalar* dydx
    ) const override;

    __device__ void 
    jacobian
    (
        const scalar x, const scalar rho, const scalar* y, scalar* dfdx, scalar* dfdy
    ) const override;
};
}

#endif
