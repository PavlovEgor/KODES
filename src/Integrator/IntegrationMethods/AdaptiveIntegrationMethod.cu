#include <cuda/std/algorithm>
#include "AdaptiveIntegrationMethod.cuh"

// No KODES_DEFINE_DEVICE_OBJECT here: an intermediate of the family is never a
// row of a type table, only its leaves are.

__host__ void
kodes::AdaptiveIntegrationMethod::allocate()
{
    IntegrationMethod::allocate();

    yTemp_.allocate(scratchSize_);
    dydt0_.allocate(scratchSize_);
}

__host__ void
kodes::AdaptiveIntegrationMethod::deallocate()
{
    yTemp_.deallocate();
    dydt0_.deallocate();

    IntegrationMethod::deallocate();
}

// Was Integrator::adaptiveStep, a static of the Integrator template, and then
// IntegrationMethod::adaptiveStep: the half of the step control that a method
// taking trial steps does not implement itself.
__device__ void
kodes::AdaptiveIntegrationMethod::advance
(
    kodes::ODESystem* ode,
    kodes::IntegratorControls controls
) const
{
    const scalar safeScale = controls.safeScale;
    const scalar alphaInc = controls.alphaIncrease;
    const scalar alphaDec = controls.alphaDecrease;
    const scalar minScale = controls.minScale;
    const scalar maxScale = controls.maxScale;

    const ThreadVector<scalar> y = currentVector();
    scalar& t = currentT;

    scalar dt = deltaTTry;
    scalar err = 0.0;

    ode->derivatives(t, currentParameter(0), y, dydt0_);

    // Loop over solver and adjust step-size as necessary
    // to achieve desired error
    do
    {
        // Solve step and provide error estimate
        err = trialStep(ode, controls);

        // If error is large reduce dt and retry the step
        if (err > 1)
        {
            scalar scale = max(safeScale*pow(err, -alphaDec), minScale);
            dt *= scale;
            deltaTTry = dt;

            if (dt < SMALL)
            {
                printf("system %d: stepsize underflow \n", controls.system);
            }
        }
    } while (err > 1);

    // Update the state
    t += dt;
    deltaTDid = dt;

    y = yTemp_;

    // If the error is small increase the step-size
    if (err > pow(maxScale/safeScale, -1.0/alphaInc))
    {
        scalar scale = safeScale*pow(err, -alphaInc);
        deltaTTry = cuda::std::clamp(scale, minScale, maxScale)*dt;
    }
    else
    {
        deltaTTry = safeScale*maxScale*dt;
    }
}
