#ifndef KODES_ROSENBROCK
#define KODES_ROSENBROCK
#pragma once

#include <utility>
#include <cuda_runtime.h>

#include "AdaptiveIntegrationMethod.cuh"
#include "ThreadMatrix.cuh"
#include "basic_linalg.cuh"
#include "rosenbrock_tableaux.cuh"

namespace kodes
{

// The linearly implicit Rosenbrock methods: every stage is one solve against
// the same factorised I/(gamma*dt) - df/dy, so they take stiff mechanisms with
// one Jacobian a step, one LU decomposition a trial and no Newton iteration.
// As trial steps they are AdaptiveIntegrationMethods, and the controller there
// sizes the steps.
//
// One algorithm for every method; Tableau, from rosenbrock_tableaux.cuh, is
// what tells them apart. Its coefficients are compile time constants, so the
// stage sums are unrolled with them as immediates, and a zero coefficient
// costs nothing. The leaves - Rosenbrock23, Rosenbrock34, Rodas23, Rodas34 -
// only name a tableau: the device object macros take a class, not a template.
//
// The derivatives of every stage are taken at the time the step starts:
// the systems are autonomous, which is also why there are no df/dt terms.
template <class Tableau>
class Rosenbrock
    :
    public AdaptiveIntegrationMethod
{
public:

    static constexpr label kStages = Tableau::kStages;

protected:

    ThreadMatrix<scalar> dfdy_;
    ThreadMatrix<scalar> a_;

    ThreadVector<label> pivotIndices_;

    // Row scales of the implicit pivoting in LUDecompose
    ThreadVector<scalar> vv_;

    // The stages, each solved for in place. In a struct so that a copy of
    // all of them is one declaration: assigning a view writes its elements.
    struct StageVectors
    {
        ThreadVector<scalar> vectors[kStages];

        __device__ const ThreadVector<scalar>& operator[](const label i) const
        {
            return vectors[i];
        }
    };

    StageVectors k_;

    // The derivatives at the state of the latest stage that needed them
    ThreadVector<scalar> dydt_;

    enum class Coefficients { A, C, M, E };

    // A coefficient of the tableau, as a constant: indexed at run time - in
    // an unrolled loop, even - the arrays would have to be in device memory
    template <Coefficients Which, label I, label J>
    __host__ __device__ static constexpr scalar coefficient()
    {
        if constexpr (Which == Coefficients::A)
        {
            return Tableau::kA[I][J];
        }
        else if constexpr (Which == Coefficients::C)
        {
            return Tableau::kC[I][J];
        }
        else if constexpr (Which == Coefficients::M)
        {
            return Tableau::kM[J];
        }
        else
        {
            return Tableau::kE[J];
        }
    }

    // Whether a stage after the first and up to I evaluated the derivatives,
    // so that stage I reads dydt_ and not dydt0_
    template <label I>
    __host__ __device__ static constexpr bool newFUpTo()
    {
        if constexpr (I == 0)
        {
            return false;
        }
        else
        {
            return Tableau::kNewF[I] || newFUpTo<I - 1>();
        }
    }

    // A stage that does not evaluate the derivatives must be at the state of
    // the stage before it
    static constexpr bool reusedStagesRepeatTheirRow()
    {
        for (label i=1; i<kStages; i++)
        {
            for (label j=0; !Tableau::kNewF[i] && j<kStages; j++)
            {
                if (Tableau::kA[i][j] != Tableau::kA[i-1][j])
                {
                    return false;
                }
            }
        }

        return Tableau::kNewF[0];
    }

    static_assert(reusedStagesRepeatTheirRow(), "a stage that reuses the derivatives repeats the row of kA above it");

    // acc + sum over j in [J, End) of the coefficients of row I times k[j][e],
    // the zero ones left out. k is a local copy of k_, so that a loop writing
    // an element does not read the views back out of this object after it.
    template <Coefficients Which, label I, label End, label J = 0>
    __device__ static scalar combine
    (
        const StageVectors& k,
        const label e,
        const scalar acc
    )
    {
        if constexpr (J == End)
        {
            return acc;
        }
        else
        {
            constexpr scalar c = coefficient<Which, I, J>();

            if constexpr (c == 0)
            {
                return combine<Which, I, End, J + 1>(k, e, acc);
            }
            else
            {
                return combine<Which, I, End, J + 1>(k, e, acc + c*k[J][e]);
            }
        }
    }

    // Stage I: its state and derivatives, its right hand side, and the solve
    template <label I>
    __device__ void
    stage
    (
        ODESystem* ode,
        const scalar t,
        const ThreadVector<const scalar> y,
        const scalar rDt
    ) const
    {
        const StageVectors k = k_;

        if constexpr (I > 0 && Tableau::kNewF[I])
        {
            const ThreadVector<scalar> yTemp = yTemp_;

            #pragma unroll 4
            for (label e=0; e<kSystemSize; e++)
            {
                yTemp[e] = combine<Coefficients::A, I, I>(k, e, y[e]);
            }

            ode->derivatives(t, currentParameter(0), yTemp_, dydt_);
        }

        const ThreadVector<const scalar> f = newFUpTo<I>() ? dydt_ : dydt0_;

        #pragma unroll 4
        for (label e=0; e<kSystemSize; e++)
        {
            k[I][e] = f[e] + rDt*combine<Coefficients::C, I, I>(k, e, 0);
        }

        LUBacksubstitute(a_, pivotIndices_, k_[I]);
    }

    template <label... I>
    __device__ void
    stages
    (
        ODESystem* ode,
        const scalar t,
        const ThreadVector<const scalar> y,
        const scalar rDt,
        std::integer_sequence<label, I...>
    ) const
    {
        (stage<I>(ode, t, y, rDt), ...);
    }

public:

    __device__ __host__
    Rosenbrock
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
    ~Rosenbrock() = default;

    __host__ void allocate()
    {
        AdaptiveIntegrationMethod::allocate();

        dfdy_.allocate(scratchSize_);
        a_.allocate(scratchSize_);

        pivotIndices_.allocate(scratchSize_);
        vv_.allocate(scratchSize_);

        for (ThreadVector<scalar>& k : k_.vectors)
        {
            k.allocate(scratchSize_);
        }

        dydt_.allocate(scratchSize_);
    }

    __host__ void deallocate()
    {
        dfdy_.deallocate();
        a_.deallocate();

        pivotIndices_.deallocate();
        vv_.deallocate();

        for (ThreadVector<scalar>& k : k_.vectors)
        {
            k.deallocate();
        }

        dydt_.deallocate();

        AdaptiveIntegrationMethod::deallocate();
    }

    __host__ static size_t scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        const size_t n = size_t(systemSize);

        return AdaptiveIntegrationMethod::scratchBytesPerThread(systemSize, parameterSize)
             + (2 * n * n                           // dfdy_, a_
             + (kStages + 2) * n                    // vv_, k_, dydt_
               ) * sizeof(scalar)
             + n * sizeof(label);                   // pivotIndices_
    }

    // The Jacobian at the top of the step, the same for every retry. OpenFOAM
    // takes it afresh for every trial, though a retry starts from the same
    // state as the trial before it.
    __device__ void startStep(ODESystem* ode) const override
    {
        ode->jacobian(currentT, currentParameter(0), currentVector(), dfdy_);
    }

    __device__ scalar
    trialStep
    (
        ODESystem* ode,
        IntegratorControls controls
    ) const override
    {
        const ThreadVector<scalar> y = currentVector();
        const scalar t = currentT;
        const scalar dt = deltaTTry;

        {
            // Copies of the two views, as in Seulex::seul: written through a
            // member, a_ would have its pointer read back out of this object
            // after every element
            const ThreadMatrix<scalar> a = a_;
            const ThreadMatrix<const scalar> dfdy = dfdy_;
            const scalar diagonal = 1/(Tableau::kGamma*dt);

            #pragma unroll 4
            for (label i=0; i<kSystemSize; i++)
            {
                #pragma unroll 4
                for (label j=0; j<kSystemSize; j++)
                {
                    a(i, j) = -dfdy(i, j);
                }
                a(i, i) += diagonal;
            }
        }

        LUDecompose(a_, pivotIndices_, vv_);

        stages(ode, t, y, 1/dt, std::make_integer_sequence<label, kStages>());

        // The new state into yTemp_, and its error relative to the tolerance in
        // the worst component, in one pass: the step is accepted at 1 or below
        const StageVectors k = k_;
        const ThreadVector<scalar> yTemp = yTemp_;
        scalar err = 0;

        #pragma unroll 4
        for (label e=0; e<kSystemSize; e++)
        {
            const scalar yNew = combine<Coefficients::M, 0, kStages>(k, e, y[e]);
            const scalar errE = combine<Coefficients::E, 0, kStages>(k, e, 0);

            yTemp[e] = yNew;

            err = max
            (
                err,
                fabs(errE)/(controls.absTol + controls.relTol*max(fabs(y[e]), fabs(yNew)))
            );
        }

        return err;
    }
};

}

#endif
