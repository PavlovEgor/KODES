#include "Euler.cuh"
#include "basic_linalg.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::Euler)

__device__
scalar kodes::Euler::step
(
    kodes::ODESystem* ode,
    kodes::DeviceResources* deviceResources,
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

    scalar* __restrict__ y      = resources->currentVector();
    scalar dt = resources->deltaTTry[INDEXVEC(0)];

    // the trial state goes to yTemp, adaptiveStep copies the accepted one back
    // into y, so a rejected step is retried from an untouched y
    for(label i=0; i<systemSize; ++i)
    {
        err[INDEXVEC(i)] = dt*dydt0[INDEXVEC(i)];
        yTemp[INDEXVEC(i)] = y[INDEXVEC(i)] + err[INDEXVEC(i)];
    }

    return normalizeError(y, yTemp, err, systemSize, absTol, relTol);
}
