#ifndef Seulex_H
#define Seulex_H

#pragma once

#include <cuda_runtime.h>

#include "IntegrationMethod.cuh"
#include "SeulexDeviceResources.cuh"
#include "SeulexConstants.cuh"

namespace kodes
{

// A GPU port of the semi-implicit Bulirsch-Stoer extrapolation method, the same
// algorithm as OpenFOAM's own seulex ODE solver.
//
// It controls its own step size and order, so usesAdaptiveStep is false and
// step() is the whole of one accepted step rather than a trial one.
//
// Selected by the name "seulex", see methodTable; the entry pairs it with
// SeulexDeviceResources, which holds the extrapolation table, the Jacobian and
// the LU work matrix that step() casts down to reach.
class Seulex
    :
    public IntegrationMethod
{
public:

    static constexpr bool usesAdaptiveStep = false;

    __device__ __host__
    Seulex
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        const label parameterSize
    )
        : IntegrationMethod
          (
              batchSize, scratchSize, systemSize, parameterSize,
              usesAdaptiveStep
          )
    {}

    __device__ __host__
    ~Seulex() = default;

    // One linearly implicit sub-stepping sequence of nSeq_[k] steps over dtTot
    __device__ static bool
    seul
    (
        SeulexDeviceResources* resources,
        ODESystem* ode,
        const scalar t0,
        const scalar dtTot,
        const label k,
        scalar& theta
    );

    __device__ static void
    extrapolate
    (
        const label k,
        const label systemSize,
        scalar* __restrict__ table,
        scalar* __restrict__ y
    );

    __device__ scalar
    step
    (
        ODESystem* ode,
        DeviceResources* resources,
        IntegratorControls controls
    ) const override;

    KODES_DECLARE_DEVICE_OBJECT(Seulex)
};

}

#endif
