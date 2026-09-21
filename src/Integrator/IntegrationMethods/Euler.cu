#include "Euler.cuh"
#include "basic_linalg.cuh"
#include "block_reduce.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::Euler)

__device__
scalar kodes::Euler::step
(
    kodes::ODESystem* ode,
    kodes::DeviceResources* deviceResources,
    kodes::BlockStep& blockStep,
    const bool active,
    kodes::IntegratorControls controls
) const
{
    // safe: the table entry that made this method made these resources
    EulerDeviceResources* resources =
        static_cast<EulerDeviceResources*>(deviceResources);

    const label systemSize = resources->systemSize();

    const scalar absTol = controls.absTol;
    const scalar relTol = controls.relTol;

    scalar* __restrict__ yTemp = resources->yTemp();
    scalar* __restrict__ dydt0 = resources->dydt0();
    scalar* __restrict__ err  = resources->err();

    scalar* __restrict__ y = resources->currentVector();
    const scalar dt = blockStep.deltaTTry;

    scalar threadError = 0.0;

    // the trial state goes to yTemp, adaptiveStep copies the accepted one back
    // into y, so a rejected step is retried from an untouched y
    if (active)
    {
        for(label i=0; i<systemSize; ++i)
        {
            err[INDEXVEC(i)] = dt*dydt0[INDEXVEC(i)];
            yTemp[INDEXVEC(i)] = y[INDEXVEC(i)] + err[INDEXVEC(i)];
        }

        threadError = normalizeError(y, yTemp, err, systemSize, absTol, relTol);
    }

    // The block retries the step together, so it is the worst of its systems
    // that decides. An inactive thread contributed zero, which cannot win.
    return blockMax(threadError);
}
