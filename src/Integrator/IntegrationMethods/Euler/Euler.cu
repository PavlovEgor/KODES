#include "Euler.cuh"
#include "thread_vector_functions.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::Euler)

__device__
kodes::scalar kodes::Euler::step
(
    kodes::ODESystem* ode,
    kodes::DeviceResources* deviceResources,
    kodes::IntegratorControls controls
) const
{
    // safe: the table entry that made this method made these resources
    EulerDeviceResources* resources =
        static_cast<EulerDeviceResources*>(deviceResources);

    const scalar absTol = controls.absTol;
    const scalar relTol = controls.relTol;

    const ThreadVector<scalar> yTemp = resources->yTemp();
    const ThreadVector<scalar> dydt0 = resources->dydt0();
    const ThreadVector<scalar> err = resources->err();

    const ThreadVector<scalar> y = resources->currentVector();
    const scalar dt = resources->deltaTTry;

    // the trial state goes to yTemp, adaptiveStep copies the accepted one back
    // into y, so a rejected step is retried from an untouched y
    err = dt*dydt0;
    yTemp = y + err;

    // the error relative to the tolerance, in the worst component: the step is
    // accepted at 1 or below
    return cmptMax(cmptMag(err)/(absTol + relTol*cmptMax(cmptMag(y), cmptMag(yTemp))));
}
