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
    theta_.allocate(scratchSize_);

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
    theta_.deallocate();

    IntegrationMethod::deallocate();
}

__device__
bool kodes::Seulex::seul
(
    kodes::ODESystem* ode,
    const scalar t0,
    const scalar dtTot,
    const label k
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
                    theta_ = 1;
                    return false;
                }

                dy2 += sqr(dy_[i]/scale_[i]);
            }
            dy2 = sqrt(dy2);
            theta_ = dy2/denom;

            if (theta_ > 1)
            {
                return false;
            }
        }

        ode->derivatives(tNew, currentParameter(0), yTemp_, dy_);
        LUBacksubstitute(a_, pivotIndices_, dy_);
    }

    ySequence_ = yTemp_ + dy_;

    return true;
}

__device__
void kodes::Seulex::advance
(
    kodes::ODESystem* ode,
    kodes::IntegratorControls controls
) const
{
    const scalar absTol = controls.absTol;
    const scalar relTol = controls.relTol;

    const scalar jacRedo = min(1e-4, relTol);

    // Both outlive this call: kTarg is the order the previous step of this
    // system settled on, theta the contraction rate it measured. They live in
    // the thread's scratch slot, since a local would be indeterminate on every
    // step after the first - which is the one the first_ branch below writes.
    label&  kTarg = kTarg_;
    scalar& theta = theta_;

    const ThreadVector<scalar> y = currentVector();
    scalar& t = currentT;

    temp_[0] = GREAT;
    scalar dt = deltaTTry;
    y0_ = y;
    dtOpt_[0] = fabs(0.1*dt);

    if (first || prevReject)
    {
        theta = 2*jacRedo;
    }

    if (first)
    {
        const scalar logTol = -log10(relTol + absTol)*0.6 + 0.5;
        kTarg = max(1, min(kSeulexMaxOrder - 1, label(logTol)));
    }

    scale_ = absTol + relTol*cmptMag(y);

    bool jacUpdated = false;

    if (theta > jacRedo)
    {
        ode->jacobian(t, currentParameter(0), y, dfdy_);
        jacUpdated = true;
    }

    label k;
    scalar dtNew = fabs(dt);
    bool firstk = true;

    while (firstk || reject)
    {
        dt = forward ? dtNew : -dtNew;
        firstk = false;
        reject = false;

        if (fabs(dt) <= fabs(t) * sqr(SMALL))
        {
            printf("step size underflow : %0.16f \n", dt);
        }

        scalar errOld = 0;

        for (k=0; k<=kTarg+1; k++)
        {
            bool success = seul(ode, t, dt, k);

            if (!success)
            {
                reject = true;
                dtNew = fabs(dt)*kSeulexStepFactor5;
                break;
            }

            if (k == 0)
            {
                y = ySequence_;
            }
            else
            {
                table_.column(k-1) = ySequence_;
            }

            if (k != 0)
            {
                table_.extrapolate(k, y);
                // one pass for both, so copies of the views, see seul()
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
                err = sqrt(err/kSystemSize);
                if (err > 1/SMALL || (k > 1 && err >= errOld))
                {
                    reject = true;
                    dtNew = fabs(dt)*kSeulexStepFactor5;
                    break;
                }
                errOld = min(4*err, 1.0);
                scalar expo = 1.0/(k + 1);
                scalar facmin = pow(kSeulexStepFactor3, expo);
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

                if ((first || last) && err <= 1)
                {
                    break;
                }

                if
                (
                    k == kTarg - 1
                && !prevReject
                && !first && !last
                )
                {
                    if (err <= 1)
                    {
                        break;
                    }
                    else if (err > kSeulexStepSequence[kTarg]*kSeulexStepSequence[kTarg + 1]*4)
                    {
                        reject = true;
                        kTarg = k;
                        if (kTarg>1 && temp_[k-1] < kSeulexKFactor1*temp_[k])
                        {
                            kTarg--;
                        }
                        dtNew = dtOpt_[kTarg];
                        break;
                    }
                }

                if (k == kTarg)
                {
                    if (err <= 1)
                    {
                        break;
                    }
                    else if (err > kSeulexStepSequence[k + 1]*2)
                    {
                        reject = true;
                        if (kTarg>1 && temp_[k-1] < kSeulexKFactor1*temp_[k])
                        {
                            kTarg--;
                        }
                        dtNew = dtOpt_[kTarg];
                        break;
                    }
                }

                if (k == kTarg+1)
                {
                    if (err > 1)
                    {
                        reject = true;
                        if
                        (
                            kTarg > 1
                        && temp_[kTarg-1] < kSeulexKFactor1*temp_[kTarg]
                        )
                        {
                            kTarg--;
                        }
                        dtNew = dtOpt_[kTarg];
                    }
                    break;
                }
            }
        }
        if (reject)
        {
            prevReject = true;
            if (!jacUpdated)
            {
                theta = 2*jacRedo;

                if (theta > jacRedo && !jacUpdated)
                {
                    ode->jacobian(t, currentParameter(0), y, dfdy_);
                    jacUpdated = true;
                }
            }
        }

    }
    jacUpdated = false;

    deltaTDid = dt;
    t += dt;

    label kopt;
    if (k == 1)
    {
        kopt = 2;
    }
    else if (k <= kTarg)
    {
        kopt=k;
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

    if (prevReject)
    {
        kTarg = min(kopt, k);
        dtNew = min(fabs(dt), dtOpt_[kTarg]);
        prevReject = false;
    }
    else
    {
        if (kopt <= k)
        {
            dtNew = dtOpt_[kopt];
        }
        else
        {
            if (k < kTarg && temp_[k] < kSeulexKFactor2*temp_[k - 1])
            {
                dtNew = dtOpt_[k]*kSeulexWorkEstimate[kopt + 1]/kSeulexWorkEstimate[k];
            }
            else
            {
                dtNew = dtOpt_[k]*kSeulexWorkEstimate[kopt]/kSeulexWorkEstimate[k];
            }
        }
        kTarg = kopt;
    }

    deltaTTry = forward ? dtNew : -dtNew;
}
