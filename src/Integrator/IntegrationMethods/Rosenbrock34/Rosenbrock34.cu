#include "Rosenbrock34.cuh"
#include "basic_linalg.cuh"
#include "thread_vector_functions.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::Rosenbrock34)

__host__ void
kodes::Rosenbrock34::allocate()
{
    AdaptiveIntegrationMethod::allocate();

    dfdy_.allocate(scratchSize_);
    a_.allocate(scratchSize_);

    pivotIndices_.allocate(scratchSize_);
    vv_.allocate(scratchSize_);

    k1_.allocate(scratchSize_);
    k2_.allocate(scratchSize_);
    k3_.allocate(scratchSize_);
    k4_.allocate(scratchSize_);

    err_.allocate(scratchSize_);
    dydt_.allocate(scratchSize_);
}

__host__ void
kodes::Rosenbrock34::deallocate()
{
    dfdy_.deallocate();
    a_.deallocate();

    pivotIndices_.deallocate();
    vv_.deallocate();

    k1_.deallocate();
    k2_.deallocate();
    k3_.deallocate();
    k4_.deallocate();

    err_.deallocate();
    dydt_.deallocate();

    AdaptiveIntegrationMethod::deallocate();
}

__device__
void kodes::Rosenbrock34::startStep(kodes::ODESystem* ode) const
{
    ode->jacobian(currentT, currentParameter(0), currentVector(), dfdy_);
}

// OpenFOAM's Rosenbrock34::solve, but for the Jacobian: OpenFOAM takes it
// afresh for every trial, though a retry starts from the same state as the
// trial before it, so here startStep() takes it once a step.
__device__
kodes::scalar kodes::Rosenbrock34::trialStep
(
    kodes::ODESystem* ode,
    kodes::IntegratorControls controls
) const
{
    const ThreadVector<scalar> y = currentVector();
    const scalar t = currentT;
    const scalar dt = deltaTTry;

    {
        // Copies of the two views, as in Seulex::seul: written through a
        // member, a_ would have its pointer read back out of this object after
        // every element
        const ThreadMatrix<scalar> a = a_;
        const ThreadMatrix<const scalar> dfdy = dfdy_;

        #pragma unroll 4
        for (label i=0; i<kSystemSize; i++)
        {
            #pragma unroll 4
            for (label j=0; j<kSystemSize; j++)
            {
                a(i, j) = -dfdy(i, j);
            }
            a(i, i) += 1/(kGamma*dt);
        }
    }

    LUDecompose(a_, pivotIndices_, vv_);

    // k1
    k1_ = dydt0_;
    LUBacksubstitute(a_, pivotIndices_, k1_);

    // k2
    yTemp_ = y + kA21*k1_;
    ode->derivatives(t + kC2*dt, currentParameter(0), yTemp_, dydt_);

    k2_ = dydt_ + (kC21/dt)*k1_;
    LUBacksubstitute(a_, pivotIndices_, k2_);

    // k3
    yTemp_ = y + kA31*k1_ + kA32*k2_;
    ode->derivatives(t + kC3*dt, currentParameter(0), yTemp_, dydt_);

    k3_ = dydt_ + (kC31/dt)*k1_ + (kC32/dt)*k2_;
    LUBacksubstitute(a_, pivotIndices_, k3_);

    // k4, at the state and time of k3: the method's fourth row of a and c
    // repeats its third, so OpenFOAM reuses the derivatives too
    k4_ = dydt_ + (kC41/dt)*k1_ + (kC42/dt)*k2_ + (kC43/dt)*k3_;
    LUBacksubstitute(a_, pivotIndices_, k4_);

    yTemp_ = y + kB1*k1_ + kB2*k2_ + kB3*k3_ + kB4*k4_;
    err_ = kE1*k1_ + kE2*k2_ + kE4*k4_;

    // the error relative to the tolerance, in the worst component: the step is
    // accepted at 1 or below
    return cmptMax
    (
        cmptMag(err_)/(controls.absTol + controls.relTol*cmptMax(cmptMag(y), cmptMag(yTemp_)))
    );
}
