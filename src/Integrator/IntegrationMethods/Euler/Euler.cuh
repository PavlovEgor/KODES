#ifndef Euler_H
#define Euler_H

#include <cuda/cmath>
#include <cuda_runtime.h>

#include "basic_linalg.cuh"

#include "IntegratorControls.cuh"
#include "ODESystem.cuh"
#include "EulerDeviceResources.cuh"

#pragma once

namespace kodes
{
template<class ODESystem>
class Euler
{
private:
    Euler() = delete;

    ~Euler() = delete;

    Euler(const Euler&) = delete;

    Euler& operator=(const Euler&) = delete;

public:

    static const bool useAdaptiveStep = true;

    __device__
    static scalar step
    (
        ODESystem* ode,
        EulerDeviceResources* resources,
        IntegratorControls controls
    );
};

}

#include "Euler.cu"

#endif