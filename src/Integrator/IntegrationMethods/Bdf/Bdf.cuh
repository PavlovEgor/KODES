#ifndef KODES_BDF
#define KODES_BDF
#pragma once

#include <cuda_runtime.h>

#include "IntegrationMethod.cuh"
#include "ThreadMatrix.cuh"
#include "ThreadValue.cuh"
#include "bdf_constants.cuh"
#include "DifferenceTable.cuh"

namespace kodes
{

// The variable order backward differentiation formulas, orders 1 to 5, with
// Shampine and Reichelt's NDF modification: a port of scipy's BDF solver, the
// same algorithm step for step.
//
// Implicit: each step solves for the new state by a simplified Newton
// iteration against the factorised I - c df/dy. Unlike Seulex and the
// Rosenbrock methods, it keeps the Jacobian and the factorisation across
// steps for as long as the iteration converges, and refreshes the Jacobian
// only when it does not - for a large mechanism the bulk of the saving.
//
// It controls its own step size and order, so it is a plain IntegrationMethod
// and advance() is the whole of one accepted step. The history it steps from
// lives in scratch space between the steps of a system, and starts afresh, at
// order 1, with every system.
//
// Selected by the name "bdf", see methodTable.
class Bdf
    :
    public IntegrationMethod
{
protected:

    DifferenceTable table_;

    // Kept across steps: the Jacobian, and its factorised I - c J for the
    // c = dt/alpha[order] it was made for
    ThreadMatrix<scalar> dfdy_;
    ThreadMatrix<scalar> a_;

    ThreadVector<label> pivotIndices_;

    // Row scales of the implicit pivoting in LUDecompose
    ThreadVector<scalar> vv_;

    // R U of a change of step size, see DifferenceTable::rescale
    ThreadMatrix<scalar, kBdfTableSize> ru_;

    ThreadVector<scalar> yPredict_;
    ThreadVector<scalar> psi_;
    ThreadVector<scalar> scale_;

    // The correction to the prediction the Newton iteration has added up,
    // and its step, solved for in place
    ThreadVector<scalar> d_;
    ThreadVector<scalar> dy_;

    // Carried from one step of a system to the next
    ThreadValue<label> order_;
    ThreadValue<label> nEqualSteps_;
    ThreadValue<scalar> hAbs_;
    ThreadValue<bool> luValid_;

public:

    __device__ __host__
    Bdf
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
    ~Bdf() = default;

    __host__ void allocate();

    __host__ void deallocate();

    // Device memory needed per resident thread
    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        const size_t n = size_t(systemSize);

        return IntegrationMethod::scratchBytesPerThread(systemSize, parameterSize)
             + (DifferenceTable::kColumns * n       // table_
             + 2 * n * n                            // dfdy_, a_
             + 6 * n                                // vv_, yPredict_, psi_, scale_, d_, dy_
             + size_t(kBdfTableSize) * kBdfTableSize  // ru_
             + 1                                    // hAbs_
               ) * sizeof(scalar)
             + (n + 2) * sizeof(label)              // pivotIndices_, order_, nEqualSteps_
             + sizeof(bool);                        // luValid_
    }

    // The steps advance() is made of

    // A system's first step: the history from the state and its derivatives,
    // order 1, and the Jacobian
    __device__ void startSystem(ODESystem* ode) const;

    // Rescale the history to a step factor times the present one, at order
    __device__ void rescale(const label order, const scalar factor) const;

    // The simplified Newton iteration for the state at tNew, from yPredict_
    // into the current vector, with c = dt/alpha[order]. Returns whether it
    // converged, and the iterations it took in nIter.
    __device__ bool
    newton
    (
        ODESystem* ode,
        const scalar tNew,
        const scalar c,
        const scalar newtonTol,
        label& nIter
    ) const;

    // After an accepted step taken at order, with the safety factor and the
    // error norm of that step: change the order by at most one and the step
    // size, by the error the orders around it would have made
    __device__ void
    chooseOrderAndStep
    (
        const label order,
        const scalar errorNorm,
        const scalar safety
    ) const;

    // sqrt of the mean square of a vector expression
    template <typename X>
    __device__ static scalar rms(const X& x)
    {
        return sqrt(magSqr(x)/kSystemSize);
    }

    __device__ void
    advance
    (
        ODESystem* ode,
        IntegratorControls controls
    ) const override;

    KODES_DECLARE_DEVICE_OBJECT(Bdf)
};

}

#endif
