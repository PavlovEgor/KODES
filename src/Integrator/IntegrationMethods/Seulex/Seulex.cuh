#ifndef KODES_SEULEX
#define KODES_SEULEX
#pragma once

#include <cuda_runtime.h>

#include "IntegrationMethod.cuh"
#include "ThreadMatrix.cuh"
#include "ThreadValue.cuh"
#include "seulex_constants.cuh"
#include "ExtrapolationTable.cuh"

namespace kodes
{

// A GPU port of the semi-implicit Bulirsch-Stoer extrapolation method, the same
// algorithm as OpenFOAM's own seulex ODE solver.
//
// It controls its own step size and order, so it is a plain IntegrationMethod
// and advance() is the whole of one accepted step.
//
// Each thread takes that step for its own system alone: its own Jacobian, its
// own factorised matrix, its own step size and order. Everything added here is
// scratch space, sized with scratchSize, and named as OpenFOAM's seulex names
// its own work arrays.
//
// Selected by the name "seulex", see methodTable.
class Seulex
    :
    public IntegrationMethod
{
protected:

    ExtrapolationTable table_;
    ThreadMatrix<scalar> dfdy_;
    ThreadMatrix<scalar> a_;

    ThreadVector<label> pivotIndices_;

    // Row scales of the implicit pivoting in LUDecompose. They used to be a
    // local scalar[128] there, which a mechanism of more than 128 species ran
    // past in silence - into the thread's stack.
    ThreadVector<scalar> vv_;

    // Indexed by the extrapolation order k, not by the component. They were
    // sized max(systemSize, kSeulexTableSize), for no reason but that the size
    // was the system's before it was the table's.
    ThreadVector<scalar, kSeulexTableSize> dtOpt_;
    ThreadVector<scalar, kSeulexTableSize> temp_;

    ThreadVector<scalar> y0_;
    ThreadVector<scalar> ySequence_;
    ThreadVector<scalar> scale_;
    ThreadVector<scalar> dy_;
    ThreadVector<scalar> yTemp_;
    ThreadVector<scalar> dydt_;

    // Order and Jacobian control carried from one step of a system to the next:
    // the target extrapolation order the previous step settled on, and the
    // contraction rate that decides whether the Jacobian may be reused. They
    // outlive a step but not a system, so they are scratch like everything
    // else here - a step re-reads what the previous one left in its slot.
    ThreadValue<label>  kTarg_;
    ThreadValue<scalar> theta_;

public:

    __device__ __host__
    Seulex
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : IntegrationMethod(batchSize, scratchSize, blocks, systemSize, parameterSize)
    {}

    __device__ __host__
    ~Seulex() = default;

    // The buffers above, and the __constant__ tables of seulex_constants.cuh,
    // which are filled here because it is the same tableau that sizes the
    // order-indexed scratch
    __host__ void allocate();

    __host__ void deallocate();

    // Device memory needed per resident thread
    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        const size_t n = size_t(systemSize);

        return IntegrationMethod::scratchBytesPerThread(systemSize, parameterSize)
             + (ExtrapolationTable::kColumns * n    // table_
             + 2 * n * n                            // dfdy_, a_
             + 7 * n                                // vv_, y0_, ySequence_, scale_, dy_, yTemp_, dydt_
             + 2 * size_t(kSeulexTableSize)         // dtOpt_, temp_
             + 1                                    // theta_
               ) * sizeof(scalar)
             + (n + 1) * sizeof(label);             // pivotIndices_, kTarg_
    }

    // One linearly implicit sub-stepping sequence of kSeulexStepSequence[k]
    // steps over dtTot, from y0_ into ySequence_
    __device__ bool
    seul
    (
        ODESystem* ode,
        const scalar t0,
        const scalar dtTot,
        const label k
    ) const;

    // kTarg, or the order below it if that would have done the work for less,
    // by the work estimates temp_ - what a rejected step falls back to
    __device__ label lowerOrderIfCheaper(const label kTarg) const
    {
        if (kTarg > 1 && temp_[kTarg-1] < kSeulexKFactor1*temp_[kTarg])
        {
            return kTarg - 1;
        }

        return kTarg;
    }

    __device__ void
    advance
    (
        ODESystem* ode,
        IntegratorControls controls
    ) const override;

    KODES_DECLARE_DEVICE_OBJECT(Seulex)
};

}

#endif
