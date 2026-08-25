
#ifndef KODES_PY_JAC_SYSTEM
#define KODES_PY_JAC_SYSTEM
#pragma once

#include "ODESystem.cuh"
#include "dydt.cuh"
#include "jacob.cuh"
#include "gpu_memory.cuh"
#include "mechanism.cuh"

namespace kodes 
{
class PyJacSystem
    : public ODESystem
{
    mechanism_memory* device_memory;

public:
    __device__ __host__
    PyJacSystem(mechanism_memory *d_mem) : ODESystem(), device_memory(d_mem) {}

    __device__ __host__
    virtual ~PyJacSystem() = default;

    __host__ static
    PyJacSystem* create(mechanism_memory *d_mem);

    __host__ static void
    destroy(PyJacSystem* system);

    __device__ static void* operator new(size_t size, void* ptr) {
        return ptr;
    }

    __device__ void
    derivatives
    (
        const scalar t, const scalar pressure, const scalar* y, scalar* dy
    ) const override;

    __device__ void
    jacobian
    (
        const scalar t, const scalar pressure, const scalar* y, scalar* dfdt, scalar* dfdy
    ) const override;
};
}

#endif
