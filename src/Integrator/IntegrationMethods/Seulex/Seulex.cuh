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
    // the target extrapolation order the previous step settled on, and whether
    // the next step needs a fresh Jacobian. They outlive a step but not a
    // system, so they are scratch like everything else here - a step re-reads
    // what the previous one left in its slot.
    ThreadValue<label> kTarg_;

    // Set by a system's first step, and by every sequence of order 0 and 1:
    // when it fails, or converges more slowly than jacRedo. Until 1.2.5 this
    // was the contraction rate itself, theta_, which nothing read but to
    // compare with jacRedo.
    ThreadValue<bool> refreshJacobian_;

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
               ) * sizeof(scalar)
             + (n + 1) * sizeof(label)              // pivotIndices_, kTarg_
             + sizeof(bool);                        // refreshJacobian_
    }

    // How one sub-stepping sequence ended. Both failures reject the step
    // alike; they are told apart for what they say about it.
    enum class Sequence
    {
        Converged,

        // An increment beyond its scale, which the next one would overflow
        // past - found before the contraction rate could be measured
        Overflowed,

        // A contraction rate above 1: the sub-steps move away from each other
        Diverged
    };

    // One linearly implicit sub-stepping sequence of kSeulexStepSequence[k]
    // steps over dtTot, from y0_ into ySequence_.
    //
    // The orders 0 and 1 also measure the contraction rate of their first
    // sub-steps, and decide refreshJacobian_ with it there and then: a rate
    // above jacRedo, or a sequence that fails, asks the next step for a fresh
    // Jacobian. Returning the rate instead and deciding in tryOrders() reads
    // better, but keeps it alive across the rest of the sequence's calls into
    // the system and the LU solve, and costs both functions 8 bytes of stack.
    __device__ Sequence
    seul
    (
        ODESystem* ode,
        const scalar t0,
        const scalar dtTot,
        const label k,
        const scalar jacRedo
    ) const;

    // The steps advance() is made of, in the order it takes them

    // Reset the order-indexed estimates, keep the state the step starts from,
    // and on a system's first step settle kTarg_ and ask for a Jacobian. Was
    // `first || prevReject`, but no step ever began with prevReject set.
    __device__ void startStep(const IntegratorControls& controls) const;

    // Raise the order k from 0 until an extrapolation is accepted, and return
    // true, or the step is rejected, and return false with the step to retry
    // with in dtNew. `rejected` says whether an attempt of this step already
    // was; k is left at the order it stopped at.
    __device__ bool
    tryOrders
    (
        ODESystem* ode,
        const IntegratorControls& controls,
        const scalar jacRedo,
        const scalar dt,
        const bool rejected,
        label& k,
        scalar& dtNew
    ) const;

    // The error of the extrapolated state against the lowest column of the
    // table, normalised so that 1 is the tolerance
    __device__ scalar measureError(const IntegratorControls& controls) const;

    // dtOpt_[k], the step order k asks for, and temp_[k], the work per unit
    // time it would cost
    __device__ void
    estimateOrderCost
    (
        const label k,
        const scalar err,
        const scalar dt
    ) const;

    // What the error of order k decides: accept the step, reject it - having
    // lowered kTarg_ if that is cheaper - or try the next order
    enum class Convergence
    {
        Accepted,
        Rejected,
        Undecided
    };

    __device__ Convergence
    checkConvergence
    (
        const label k,
        const scalar err,
        const bool rejected
    ) const;

    // The order to aim the next step at, from the work estimates
    __device__ label chooseOrder(const label k) const;

    // The size of the next step, and kTarg_ for it
    __device__ scalar
    chooseStepSize
    (
        const label k,
        const label kopt,
        const scalar dt,
        const bool rejected
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
