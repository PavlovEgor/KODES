#include "IntegrationMethod.cuh"
#include "AdaptiveDeviceResources.cuh"
#include "basic_linalg.cuh"
#include "block_reduce.cuh"

// Was Integrator::adaptiveStep, a static of the Integrator template. It belongs
// to the method rather than to the driver: it is the half of the step control
// that a method taking trial steps does not implement itself.
//
// Collective, like everything else a block does now. The error that decides
// whether to retry is the worst of the block, so the retry loop runs the same
// number of times in every thread and the block keeps one step size. A method
// that shares a Jacobian has no choice about that; Euler, which does not, pays
// for it - see the comment on kUsesAdaptiveStep in Euler.cuh.
__device__ void
kodes::IntegrationMethod::adaptiveStep
(
    kodes::ODESystem* ode,
    kodes::DeviceResources* resources,
    kodes::BlockStep& blockStep,
    const bool active,
    kodes::IntegratorControls controls
) const
{
    // safe: a method that asks for this controller is registered with resources
    // that carry the two vectors below - see methodTable
    AdaptiveDeviceResources* res =
        static_cast<AdaptiveDeviceResources*>(resources);

    const label systemSize = res->systemSize();

    const scalar safeScale = controls.safeScale;
    const scalar alphaInc = controls.alphaIncrease;
    const scalar alphaDec = controls.alphaDecrease;
    const scalar minScale = controls.minScale;
    const scalar maxScale = controls.maxScale;

    scalar* __restrict__ yTemp = res->yTemp();
    scalar* __restrict__ dydt0 = res->dydt0();

    scalar* __restrict__ y = res->currentVector();

    scalar dt = blockStep.deltaTTry;
    scalar err = 0.0;

    if (active)
    {
        ode->derivatives(blockStep.t, res->currentParameter(0), y, dydt0);
    }

    // Loop over solver and adjust step-size as necessary
    // to achieve desired error
    do
    {
        // Solve step and provide error estimate. It comes back already
        // reduced over the block, so this condition is block wide.
        err = step(ode, resources, blockStep, active, controls);

        // If error is large reduce dt and retry the step
        if (err > 1)
        {
            scalar scale = max(safeScale*pow(err, -alphaDec), minScale);
            dt *= scale;
            blockStep.deltaTTry = dt;

            if (dt < SMALL)
            {
                if (threadIdx.x == 0)
                {
                    printf
                    (
                        "positions %d..%d: stepsize underflow \n",
                        controls.position, controls.position + blockDim.x - 1
                    );
                }
            }
        }
    } while (err > 1);

    // Update the state
    blockStep.t += dt;
    blockStep.deltaTDid = dt;

    if (active)
    {
        copyVec(y, yTemp, systemSize);
    }

    // If the error is small increase the step-size
    if (err > pow(maxScale/safeScale, -1.0/alphaInc))
    {
        scalar scale = safeScale*pow(err, -alphaInc);
        blockStep.deltaTTry = clamp(scale, minScale, maxScale)*dt;
    }
    else
    {
        blockStep.deltaTTry = safeScale*maxScale*dt;
    }
}
