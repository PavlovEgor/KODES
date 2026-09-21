#ifndef KODES_SEULEX
#define KODES_SEULEX
#pragma once

#include <cuda_runtime.h>

#include "IntegrationMethod.cuh"
#include "SeulexDeviceResources.cuh"
#include "seulex_constants.cuh"

namespace kodes
{

// A GPU port of the semi-implicit Bulirsch-Stoer extrapolation method, the same
// algorithm as OpenFOAM's own seulex ODE solver.
//
// It controls its own step size and order, so kUsesAdaptiveStep is false and
// step() is the whole of one accepted step rather than a trial one.
//
// A whole block takes that step together. The linearly implicit sub-step
// solves (I/dt - J) dy = f(y), which is a W-method: its order holds for *any*
// J, so the J of one system of the block serves for all of them - an inexact
// one costs contraction, not accuracy, and seul() measures exactly that
// contraction and rejects the step when it is too poor. With one J, one step
// size and one order for the block, the matrix is one matrix and it is
// factorised once instead of blockDim.x times.
//
// Selected by the name "seulex", see methodTable; the entry pairs it with
// SeulexDeviceResources, which holds both halves of the scratch - the per
// thread state and the per block matrix - that step() casts down to reach.
class Seulex
    :
    public IntegrationMethod
{
public:

    static constexpr bool kUsesAdaptiveStep = false;

    __device__ __host__
    Seulex
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : IntegrationMethod
          (
              batchSize, scratchSize, blocks, systemSize, parameterSize,
              kUsesAdaptiveStep
          )
    {}

    __device__ __host__
    ~Seulex() = default;

    // One linearly implicit sub-stepping sequence of kSeulexStepSequence[k]
    // steps over dtTot. Collective: the block builds and factorises the matrix
    // together, then every thread back-substitutes its own right hand side
    // through it. Returns the same answer in every thread, and `theta` - the
    // contraction it measured - comes back as the worst over the block.
    //
    // `jacTid` is the global index of the thread whose scratch slot holds the
    // Jacobian, since pyJac wrote it in the layout of that one thread.
    __device__ static bool
    seul
    (
        SeulexDeviceResources* resources,
        const SeulexBlockScratch& block,
        ODESystem* ode,
        const label jacTid,
        const bool active,
        const scalar t0,
        const scalar dtTot,
        const label k,
        scalar& theta
    );

    // Per thread: the extrapolation table and the state it refines are the
    // thread's own, and only the order k they are indexed by is shared.
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
        BlockStep& step,
        const bool active,
        IntegratorControls controls
    ) const override;

    KODES_DECLARE_DEVICE_OBJECT(Seulex)
};

}

#endif
