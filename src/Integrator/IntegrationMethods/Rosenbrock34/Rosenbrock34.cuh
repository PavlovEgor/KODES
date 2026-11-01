#ifndef KODES_ROSENBROCK34
#define KODES_ROSENBROCK34
#pragma once

#include <cuda_runtime.h>

#include "AdaptiveIntegrationMethod.cuh"
#include "ThreadMatrix.cuh"

namespace kodes
{

// The four stage, third order L-stable Rosenbrock method with an embedded
// second order error estimate - OpenFOAM's Rosenbrock34, with the same
// coefficients and the same work arrays under the same names.
//
// A linearly implicit method: every stage is one solve against the same
// factorised I/(gamma*dt) - df/dy, so it takes stiff mechanisms with one
// Jacobian and one LU decomposition per trial step and no Newton iteration.
// As a trial step it is an AdaptiveIntegrationMethod, and the controller there
// sizes the steps.
//
// OpenFOAM also adds the df/dt terms, d1..d4 times dt*dfdx. The systems here
// are autonomous and ODESystem::jacobian gives no df/dt, so they are left out
// rather than multiplied by a zero vector.
//
// Selected by the name "rosenbrock34", see methodTable.
class Rosenbrock34
    :
    public AdaptiveIntegrationMethod
{
protected:

    ThreadMatrix<scalar> dfdy_;
    ThreadMatrix<scalar> a_;

    ThreadVector<label> pivotIndices_;

    // Row scales of the implicit pivoting in LUDecompose
    ThreadVector<scalar> vv_;

    // The stages, each solved for in place
    ThreadVector<scalar> k1_;
    ThreadVector<scalar> k2_;
    ThreadVector<scalar> k3_;
    ThreadVector<scalar> k4_;

    ThreadVector<scalar> err_;
    ThreadVector<scalar> dydt_;

public:

    // L-stable coefficients of Hairer and Wanner, as OpenFOAM's
    static constexpr scalar kA21 = 2;
    static constexpr scalar kA31 = 48.0/25.0;
    static constexpr scalar kA32 = 6.0/25.0;

    static constexpr scalar kC21 = -8;
    static constexpr scalar kC31 = 372.0/25.0;
    static constexpr scalar kC32 = 12.0/5.0;

    static constexpr scalar kC41 = -112.0/125.0;
    static constexpr scalar kC42 = -54.0/125.0;
    static constexpr scalar kC43 = -2.0/5.0;

    static constexpr scalar kB1 = 19.0/9.0;
    static constexpr scalar kB2 = 1.0/2.0;
    static constexpr scalar kB3 = 25.0/108.0;
    static constexpr scalar kB4 = 125.0/108.0;

    // e3 is 0, and dropped
    static constexpr scalar kE1 = 34.0/108.0;
    static constexpr scalar kE2 = 7.0/36.0;
    static constexpr scalar kE4 = 125.0/108.0;

    static constexpr scalar kGamma = 1.0/2.0;
    static constexpr scalar kC2 = 1;
    static constexpr scalar kC3 = 3.0/5.0;

    __device__ __host__
    Rosenbrock34
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : AdaptiveIntegrationMethod(batchSize, scratchSize, blocks, systemSize, parameterSize)
    {}

    __device__ __host__
    ~Rosenbrock34() = default;

    __host__ void allocate();

    __host__ void deallocate();

    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        const size_t n = size_t(systemSize);

        return AdaptiveIntegrationMethod::scratchBytesPerThread(systemSize, parameterSize)
             + (2 * n * n                           // dfdy_, a_
             + 7 * n                                // vv_, k1_..k4_, err_, dydt_
               ) * sizeof(scalar)
             + n * sizeof(label);                   // pivotIndices_
    }

    __device__ scalar
    trialStep
    (
        ODESystem* ode,
        IntegratorControls controls
    ) const override;

    KODES_DECLARE_DEVICE_OBJECT(Rosenbrock34)
};

}

#endif
