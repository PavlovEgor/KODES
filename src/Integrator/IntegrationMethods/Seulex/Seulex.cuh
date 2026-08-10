#ifndef Seulex_H
#define Seulex_H

#include <cuda/cmath>
#include <cuda_runtime.h>

#include "basic_linalg.cuh"

#include "IntegratorControls.cuh"
#include "ODESystem.cuh"
#include "SeulexDeviceResources.cuh"
#include "SeulexConstants.cuh"

#pragma once

namespace kodes
{
template<class ODESystem>
class Seulex
{
private:
    Seulex() = delete;

    ~Seulex() = delete;

    Seulex(const Seulex&) = delete;

    Seulex& operator=(const Seulex&) = delete;

public:

    static const bool useAdaptiveStep = false;

    __device__
    static bool seul (
        SeulexDeviceResources* resources,
        ODESystem* ode,
        const scalar t0,
        const scalar dtTot,
        const label k,
        scalar& theta
    );

    __device__
    static void extrapolate (const label k,const label sizeOfSystem, scalar* __restrict__ table, scalar* __restrict__ y);

    __device__
    static void step
    (
        ODESystem* ode,
        SeulexDeviceResources* resources,
        IntegratorControls controls
    );
};

}

#include "Seulex.cu"

#endif
