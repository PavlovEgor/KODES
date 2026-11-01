#include "Euler.cuh"
#include "thread_vector_functions.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::Euler)

__host__ void
kodes::Euler::allocate()
{
    AdaptiveIntegrationMethod::allocate();

    err_.allocate(scratchSize_);
}

__host__ void
kodes::Euler::deallocate()
{
    err_.deallocate();

    AdaptiveIntegrationMethod::deallocate();
}

__device__
kodes::scalar kodes::Euler::trialStep
(
    kodes::ODESystem* ode,
    kodes::IntegratorControls controls
) const
{
    const ThreadVector<scalar> y = currentVector();
    const scalar dt = deltaTTry;

    err_ = dt*dydt0_;
    yTemp_ = y + err_;

    // the error relative to the tolerance, in the worst component: the step is
    // accepted at 1 or below
    return cmptMax
    (
        cmptMag(err_)/(controls.absTol + controls.relTol*cmptMax(cmptMag(y), cmptMag(yTemp_)))
    );
}
