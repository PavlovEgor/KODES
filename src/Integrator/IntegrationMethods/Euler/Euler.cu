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

    const scalar absTol_ = controls.absTol;
    const scalar relTol_ = controls.relTol;

    scalar* __restrict__ yTemp_ = resources->yTemp();
    scalar* __restrict__ dydx0_ = resources->dydx0();
    scalar* __restrict__ err_  = resources->err();

    scalar* __restrict__ y      = resources->vectors;
    scalar dt = resources->deltaTTry[INDEXVEC(0)];

    for(label i=0; i<systemSize; ++i)
    {
        err_[i] = dt*dydx0_[i];
        y[i] = yTemp_[i] + err_[i];
    }

    return normalizeError(yTemp_, y, err_, systemSize, absTol_, relTol_);
}
