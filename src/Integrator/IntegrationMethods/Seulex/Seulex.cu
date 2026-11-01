#include "Seulex.cuh"
#include "basic_linalg.cuh"
#include "thread_vector_functions.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::Seulex)

__host__ void
kodes::Seulex::allocate()
{
    IntegrationMethod::allocate();

    table_.allocate(scratchSize_);
    dfdy_.allocate(scratchSize_);
    a_.allocate(scratchSize_);

    pivotIndices_.allocate(scratchSize_);
    vv_.allocate(scratchSize_);

    // indexed by the extrapolation order, not by the component
    dtOpt_.allocate(scratchSize_);
    temp_.allocate(scratchSize_);

    y0_.allocate(scratchSize_);
    ySequence_.allocate(scratchSize_);
    scale_.allocate(scratchSize_);
    dy_.allocate(scratchSize_);
    yTemp_.allocate(scratchSize_);
    dydt_.allocate(scratchSize_);

    // one entry per thread: step control that survives a step, not a system
    kTarg_.allocate(scratchSize_);
    refreshJacobian_.allocate(scratchSize_);

    uploadSeulexConstants();
}

__host__ void
kodes::Seulex::deallocate()
{
    table_.deallocate();
    dfdy_.deallocate();
    a_.deallocate();

    pivotIndices_.deallocate();
    vv_.deallocate();

    dtOpt_.deallocate();
    temp_.deallocate();

    y0_.deallocate();
    ySequence_.deallocate();
    scale_.deallocate();
    dy_.deallocate();
    yTemp_.deallocate();
    dydt_.deallocate();

    kTarg_.deallocate();
    refreshJacobian_.deallocate();

    IntegrationMethod::deallocate();
}

__device__
kodes::Seulex::Sequence kodes::Seulex::seul
(
    kodes::ODESystem* ode,
    const scalar t0,
    const scalar dtTot,
    const label k,
    const scalar jacRedo
) const
{
    label nSteps = kSeulexStepSequence[k];
    scalar dt = dtTot/nSteps;

    {
        // Copies of the two views, as ThreadVector::update makes them: written
        // through a member of this object, a_ and dfdy_ would have their
        // pointers read back out of it after every element
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
            a(i, i) += 1/dt;
        }
    }

    LUDecompose(a_, pivotIndices_, vv_);

    scalar tNew = t0 + dt;
    ode->derivatives(tNew, currentParameter(0), y0_, dy_);

    LUBacksubstitute(a_, pivotIndices_, dy_);

    yTemp_ = y0_;

    for (label nn=1; nn<nSteps; nn++)
    {
        yTemp_ += dy_;

        tNew += dt;

        if (nn == 1 && k<=1)
        {
            const scalar dy1 = mag(dy_/scale_);

            ode->derivatives(t0 + dt, currentParameter(0), yTemp_, dydt_);
            dy_ = dydt_ - dy_/dt;

            LUBacksubstitute(a_, pivotIndices_, dy_);

            const scalar denom = min(1.0, dy1 + SMALL);
            scalar dy2 = 0;
            #pragma unroll 4
            for (label i=0; i<kSystemSize; i++)
            {
                // Test of dy_[i] to avoid overflow
                if (fabs(dy_[i]) > scale_[i]*denom)
                {
                    refreshJacobian_ = true;
                    return Sequence::Overflowed;
                }

                dy2 += sqr(dy_[i]/scale_[i]);
            }
            dy2 = sqrt(dy2);
            const scalar contraction = dy2/denom;

            refreshJacobian_ = contraction > jacRedo;

            if (contraction > 1)
            {
                return Sequence::Diverged;
            }
        }

        ode->derivatives(tNew, currentParameter(0), yTemp_, dy_);
        LUBacksubstitute(a_, pivotIndices_, dy_);
    }

    ySequence_ = yTemp_ + dy_;

    return Sequence::Converged;
}

// One accepted step: OpenFOAM's seulex::solve, cut where it changes subject.
// The arithmetic and the order of every decision are the same; what were
// breaks out of the middle of the order loop are now return values.
__device__
void kodes::Seulex::advance
(
    kodes::ODESystem* ode,
    kodes::IntegratorControls controls
) const
{
    const scalar jacRedo = min(1e-4, controls.relTol);

    startStep(controls);

    const ThreadVector<scalar> y = currentVector();
    scalar& t = currentT;

    // Read once: the sequences of order 0 and 1 below set refreshJacobian_
    // afresh for the next step
    const bool jacobianFresh = refreshJacobian_;

    if (jacobianFresh)
    {
        ode->jacobian(t, currentParameter(0), y, dfdy_);
    }

    scalar dt = deltaTTry;
    scalar dtNew = fabs(dt);
    label k;

    // Whether an attempt of this step has been rejected. Until 1.2.6 it was
    // StepState::prevReject, with reject beside it, though neither ever
    // outlived the step: reset here at the end of every one.
    bool rejected = false;

    for (;;)
    {
        dt = forward ? dtNew : -dtNew;

        if (fabs(dt) <= fabs(t) * sqr(SMALL))
        {
            printf("step size underflow : %0.16f \n", dt);
        }

        if (tryOrders(ode, controls, jacRedo, dt, rejected, k, dtNew))
        {
            break;
        }

        // The first rejection retries with a fresh Jacobian, if the step did
        // not start with one; later ones already have it. Was a jacUpdated
        // flag, which is exactly this.
        if (!jacobianFresh && !rejected)
        {
            ode->jacobian(t, currentParameter(0), y, dfdy_);
        }

        rejected = true;
    }

    deltaTDid = dt;
    t += dt;

    const scalar dtNext = chooseStepSize(k, chooseOrder(k), dt, rejected);

    deltaTTry = forward ? dtNext : -dtNext;
}

__device__
void kodes::Seulex::startStep
(
    const kodes::IntegratorControls& controls
) const
{
    const scalar absTol = controls.absTol;
    const scalar relTol = controls.relTol;

    temp_[0] = GREAT;
    y0_ = currentVector();
    dtOpt_[0] = fabs(0.1*deltaTTry);

    if (first)
    {
        refreshJacobian_ = true;

        const scalar logTol = -log10(relTol + absTol)*0.6 + 0.5;
        kTarg_ = max(1, min(kSeulexMaxOrder - 1, label(logTol)));
    }

    scale_ = absTol + relTol*cmptMag(currentVector());
}

__device__
bool kodes::Seulex::tryOrders
(
    kodes::ODESystem* ode,
    const kodes::IntegratorControls& controls,
    const scalar jacRedo,
    const scalar dt,
    const bool rejected,
    label& k,
    scalar& dtNew
) const
{
    const ThreadVector<scalar> y = currentVector();

    scalar errOld = 0;

    for (k = 0; k <= kTarg_ + 1; k++)
    {
        if (seul(ode, currentT, dt, k, jacRedo) != Sequence::Converged)
        {
            dtNew = fabs(dt)*kSeulexStepFactor5;
            return false;
        }

        if (k == 0)
        {
            y = ySequence_;
            continue;
        }

        table_.column(k-1) = ySequence_;
        table_.extrapolate(k, y);

        const scalar err = measureError(controls);

        if (err > 1/SMALL || (k > 1 && err >= errOld))
        {
            dtNew = fabs(dt)*kSeulexStepFactor5;
            return false;
        }

        errOld = min(4*err, 1.0);

        estimateOrderCost(k, err, dt);

        const Convergence convergence = checkConvergence(k, err, rejected);

        if (convergence == Convergence::Rejected)
        {
            dtNew = dtOpt_[kTarg_];
            return false;
        }

        if (convergence == Convergence::Accepted)
        {
            return true;
        }
    }

    // not reached: order kTarg_ + 1 always decides, and kTarg_ only moves on a
    // rejection, which returns
    return true;
}

__device__
kodes::scalar kodes::Seulex::measureError
(
    const kodes::IntegratorControls& controls
) const
{
    const scalar absTol = controls.absTol;
    const scalar relTol = controls.relTol;

    // one pass for both, so copies of the views, see seul()
    const ThreadVector<const scalar> y = currentVector();
    const ThreadVector<scalar> scale = scale_;
    const ThreadVector<const scalar> y0 = y0_;
    const ThreadVector<const scalar> extrapolated = table_.column(0);

    scalar err = 0;
    #pragma unroll 4
    for (label i=0; i<kSystemSize; ++i)
    {
        scale[i] = absTol + relTol*fabs(y0[i]);
        err += sqr((y[i] - extrapolated[i])/scale[i]);
    }

    return sqrt(err/kSystemSize);
}

__device__
void kodes::Seulex::estimateOrderCost
(
    const label k,
    const scalar err,
    const scalar dt
) const
{
    const scalar expo = 1.0/(k + 1);
    const scalar facmin = pow(kSeulexStepFactor3, expo);

    scalar fac;
    if (err == 0)
    {
        fac = 1/facmin;
    }
    else
    {
        fac = kSeulexStepFactor2/pow(err/kSeulexStepFactor1, expo);
        fac = max(facmin/kSeulexStepFactor4, min(1/facmin, fac));
    }

    dtOpt_[k] = fabs(dt*fac);
    temp_[k] = kSeulexWorkEstimate[k]/dtOpt_[k];
}

__device__
kodes::Seulex::Convergence kodes::Seulex::checkConvergence
(
    const label k,
    const scalar err,
    const bool rejected
) const
{
    label& kTarg = kTarg_;

    if ((first || last) && err <= 1)
    {
        return Convergence::Accepted;
    }

    // one order short of the target, on a step that follows an accepted one
    if (k == kTarg - 1 && !rejected && !first && !last)
    {
        if (err <= 1)
        {
            return Convergence::Accepted;
        }

        if (err > kSeulexStepSequence[kTarg]*kSeulexStepSequence[kTarg + 1]*4)
        {
            kTarg = lowerOrderIfCheaper(k);
            return Convergence::Rejected;
        }
    }

    if (k == kTarg)
    {
        if (err <= 1)
        {
            return Convergence::Accepted;
        }

        if (err > kSeulexStepSequence[k + 1]*2)
        {
            kTarg = lowerOrderIfCheaper(kTarg);
            return Convergence::Rejected;
        }
    }

    // one past the target: the last order this step tries
    if (k == kTarg + 1)
    {
        if (err > 1)
        {
            kTarg = lowerOrderIfCheaper(kTarg);
            return Convergence::Rejected;
        }

        return Convergence::Accepted;
    }

    return Convergence::Undecided;
}

__device__
kodes::label kodes::Seulex::chooseOrder(const label k) const
{
    label kopt;

    if (k == 1)
    {
        kopt = 2;
    }
    else if (k <= kTarg_)
    {
        kopt = k;
        if (temp_[k-1] < kSeulexKFactor1*temp_[k])
        {
            kopt = k - 1;
        }
        else if (temp_[k] < kSeulexKFactor2*temp_[k - 1])
        {
            kopt = min(k + 1, kSeulexMaxOrder - 1);
        }
    }
    else
    {
        kopt = k - 1;
        if (k > 2 && temp_[k-2] < kSeulexKFactor1*temp_[k - 1])
        {
            kopt = k - 2;
        }
        if (temp_[k] < kSeulexKFactor2*temp_[kopt])
        {
            kopt = min(k, kSeulexMaxOrder - 1);
        }
    }

    return kopt;
}

__device__
kodes::scalar kodes::Seulex::chooseStepSize
(
    const label k,
    const label kopt,
    const scalar dt,
    const bool rejected
) const
{
    label& kTarg = kTarg_;

    if (rejected)
    {
        kTarg = min(kopt, k);

        return min(fabs(dt), dtOpt_[kTarg]);
    }

    scalar dtNew;

    if (kopt <= k)
    {
        dtNew = dtOpt_[kopt];
    }
    else if (k < kTarg && temp_[k] < kSeulexKFactor2*temp_[k - 1])
    {
        dtNew = dtOpt_[k]*kSeulexWorkEstimate[kopt + 1]/kSeulexWorkEstimate[k];
    }
    else
    {
        dtNew = dtOpt_[k]*kSeulexWorkEstimate[kopt]/kSeulexWorkEstimate[k];
    }

    kTarg = kopt;

    return dtNew;
}
