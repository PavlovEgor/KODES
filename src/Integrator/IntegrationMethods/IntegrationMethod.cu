#include "IntegrationMethod.cuh"
#include "AdaptiveDeviceResources.cuh"
#include "basic_linalg.cuh"

// Was Integrator::adaptiveStep, a static of the Integrator template. It belongs
// to the method rather than to the driver: it is the half of the step control
// that a method taking trial steps does not implement itself.
__device__ void
kodes::IntegrationMethod::adaptiveStep
(
    kodes::ODESystem* ode,
    kodes::DeviceResources* resources,
    kodes::IntegratorControls controls
) const
{
    // safe: a method that asks for this controller is registered with resources
    // that carry the two vectors below - see methodTable
    AdaptiveDeviceResources* res =
        static_cast<AdaptiveDeviceResources*>(resources);

    const label systemSize = res->systemSize();

    const scalar safeScale_ = controls.safeScale;
    const scalar alphaInc_ = controls.alphaIncrease;
    const scalar alphaDec_ = controls.alphaDecrease;
    const scalar minScale_ = controls.minScale;
    const scalar maxScale_ = controls.maxScale;

    scalar* __restrict__ yTemp_ = res->yTemp();
    scalar* __restrict__ dydt0_ = res->dydt0();

    scalar* __restrict__ y      = res->currentVector();
    scalar& t      = res->currentT[INDEXVEC(0)];

    scalar dt = res->deltaTTry[INDEXVEC(0)];
    scalar err = 0.0;

    ode->derivatives(t, res->currentParameter(0), y, dydt0_);

    // Loop over solver and adjust step-size as necessary
    // to achieve desired error
    do
    {
        // Solve step and provide error estimate
        err = step(ode, resources, controls);

        // If error is large reduce dt and retry the step
        if (err > 1)
        {
            scalar scale = max(safeScale_*pow(err, -alphaDec_), minScale_);
            dt *= scale;
            res->deltaTTry[INDEXVEC(0)] = dt;

            if (dt < SMALL)
            {
                printf
                (
                    "system: %d stepsize underflow \n",
                    controls.system
                );
            }
        }
    } while (err > 1);

    // Update the state
    t += dt;
    copyVec(y, yTemp_, systemSize);

    // If the error is small increase the step-size
    if (err > pow(maxScale_/safeScale_, -1.0/alphaInc_))
    {
        scalar scale = safeScale_*pow(err, -alphaInc_);
        res->deltaTTry[INDEXVEC(0)] = clamp(scale, minScale_, maxScale_)*dt;
    }
    else
    {
        res->deltaTTry[INDEXVEC(0)] = safeScale_*maxScale_*dt;
    }
}
