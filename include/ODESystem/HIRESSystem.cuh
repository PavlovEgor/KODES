
// HIRESSystem.h
#ifndef HIRESSystem1_H
#define HIRESSystem1_H

#pragma once

#include "ODESystem.cuh"

namespace kodes 
{
class HIRESSystem
    : public ODESystem
{
public:
    __device__ __host__
    HIRESSystem(const label sizeOfSystem) : ODESystem(sizeOfSystem) {}
    __device__ __host__
    virtual ~HIRESSystem() = default;

    __host__ static
    HIRESSystem* createGPU(const label sizeOfSystem);

    __host__ static void
    destroyGPU(HIRESSystem* system);

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
