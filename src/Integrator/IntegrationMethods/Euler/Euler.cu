template<class ODESystem>
__device__
scalar kodes::Euler<ODESystem>::step
(
    ODESystem* ode,
    kodes::EulerDeviceResources* resources,
    kodes::IntegratorControls controls
)
{
    const label systemSize = resources->systemSize();
    const label system = controls.system;

    const scalar absTol_ = controls.absTol;
    const scalar relTol_ = controls.relTol;

    scalar* __restrict__ yTemp_ = resources->yTemp();
    scalar* __restrict__ dydx0_ = resources->dydx0();
    scalar* __restrict__ err_  = resources->err();

    scalar* __restrict__ y      = resources->y();
    scalar dt = resources->deltaTTry[system];

    // The trial state goes to yTemp_, the accepted one is copied back into y by
    // Integrator::adaptiveStep, so that a rejected step can simply be retried
    // from the untouched y
    for(label i=0; i<systemSize; ++i)
    {
        err_[INDEXVEC(i)] = dt*dydx0_[INDEXVEC(i)];
        yTemp_[INDEXVEC(i)] = y[INDEXVEC(i)] + err_[INDEXVEC(i)];
    }

    return normalizeError(y, yTemp_, err_, systemSize, absTol_, relTol_);
}
