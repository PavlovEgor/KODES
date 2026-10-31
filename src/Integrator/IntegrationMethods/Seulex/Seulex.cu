#include "Seulex.cuh"
#include "basic_linalg.cuh"

KODES_DEFINE_DEVICE_OBJECT(kodes::Seulex)

__device__
bool kodes::Seulex::seul (
    kodes::SeulexDeviceResources* resources,
    kodes::ODESystem* ode,
    const scalar t0,
    const scalar dtTot,
    const label k,
    scalar& theta
)
{
    const ThreadMatrix<scalar> dfdy = resources->dfdy();
    const ThreadMatrix<scalar> a = resources->a();
    const ThreadVector<label> pivotIndices = resources->pivotIndices();
    const ThreadVector<scalar> vv = resources->vv();

    const ThreadVector<scalar> y0 = resources->y0();
    const ThreadVector<scalar> scale = resources->scale();

    const ThreadVector<scalar> dy = resources->dy();
    const ThreadVector<scalar> yTemp = resources->yTemp();
    const ThreadVector<scalar> dydt = resources->dydt();
    const ThreadVector<scalar> y = resources->ySequence();

    label nSteps = kSeulexStepSequence[k];
    scalar dt = dtTot/nSteps;

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

    LUDecompose(a, pivotIndices, vv);

    scalar tNew = t0 + dt;
    ode->derivatives(tNew, resources->currentParameter(0), y0, dy);

    LUBacksubstitute(a, pivotIndices, dy);

    yTemp = y0;

    for (label nn=1; nn<nSteps; nn++)
    {
        yTemp += dy;

        tNew += dt;

        if (nn == 1 && k<=1)
        {
            scalar dy1 = 0;
            #pragma unroll 4
            for (label i=0; i<kSystemSize; i++)
            {
                dy1 += sqr(dy[i]/scale[i]);
            }
            dy1 = sqrt(dy1);

            ode->derivatives(t0 + dt, resources->currentParameter(0), yTemp, dydt);
            #pragma unroll 4
            for (label i=0; i<kSystemSize; i++)
            {
                dy[i] = dydt[i] - dy[i]/dt;
            }

            LUBacksubstitute(a, pivotIndices, dy);

            const scalar denom = min(1.0, dy1 + SMALL);
            scalar dy2 = 0;
            #pragma unroll 4
            for (label i=0; i<kSystemSize; i++)
            {
                // Test of dy[i] to avoid overflow
                if (fabs(dy[i]) > scale[i]*denom)
                {
                    theta = 1;
                    return false;
                }

                dy2 += sqr(dy[i]/scale[i]);
            }
            dy2 = sqrt(dy2);
            theta = dy2/denom;

            if (theta > 1)
            {
                return false;
            }
        }

        ode->derivatives(tNew, resources->currentParameter(0), yTemp, dy);
        LUBacksubstitute(a, pivotIndices, dy);
    }

    y = yTemp + dy;

    return true;
}

__device__
kodes::scalar kodes::Seulex::step
(
    kodes::ODESystem* ode,
    kodes::DeviceResources* deviceResources,
    kodes::IntegratorControls controls
) const
{
    // safe: the table entry that made this method made these resources
    SeulexDeviceResources* resources =
        static_cast<SeulexDeviceResources*>(deviceResources);

    const scalar absTol = controls.absTol;
    const scalar relTol = controls.relTol;

    const scalar jacRedo = min(1e-4, relTol);

    // Both outlive this call: kTarg is the order the previous step of this
    // system settled on, theta the contraction rate it measured. They live in
    // the thread's scratch slot, since a local would be indeterminate on every
    // step after the first - which is the one the first_ branch below writes.
    label&  kTarg = resources->kTarg();
    scalar& theta = resources->theta();

    const ExtrapolationTable table = resources->table();
    const ThreadMatrix<scalar> dfdy = resources->dfdy();

    const ThreadVector<scalar, kSeulexTableSize> dtOpt = resources->dtOpt();
    const ThreadVector<scalar, kSeulexTableSize> temp = resources->temp();
    const ThreadVector<scalar> y0 = resources->y0();
    const ThreadVector<scalar> ySequence = resources->ySequence();
    const ThreadVector<scalar> scale = resources->scale();

    const ThreadVector<scalar> y = resources->currentVector();
    scalar& t = resources->currentT;

    temp[0] = GREAT;
    scalar dt = resources->deltaTTry;
    y0 = y;
    dtOpt[0] = fabs(0.1*dt);

    if (resources->first || resources->prevReject)
    {
        theta = 2*jacRedo;
    }

    if (resources->first)
    {
        const scalar logTol = -log10(relTol + absTol)*0.6 + 0.5;
        kTarg = max(1, min(kSeulexMaxOrder - 1, label(logTol)));
    }

    #pragma unroll 4
    for (label i=0; i < kSystemSize; ++i)
    {
        scale[i] = absTol + relTol*fabs(y[i]);
    }

    bool jacUpdated = false;

    if (theta > jacRedo)
    {
        ode->jacobian(t, resources->currentParameter(0), y, dfdy);
        jacUpdated = true;
    }

    label k;
    scalar dtNew = fabs(dt);
    bool firstk = true;

    while (firstk || resources->reject)
    {
        dt = resources->forward ? dtNew : -dtNew;
        firstk = false;
        resources->reject = false;

        if (fabs(dt) <= fabs(t) * sqr(SMALL))
        {
            printf("step size underflow : %0.16f \n", dt);
        }

        scalar errOld = 0;

        for (k=0; k<=kTarg+1; k++)
        {
            bool success = seul(resources, ode, t, dt, k, theta);

            if (!success)
            {
                resources->reject = true;
                dtNew = fabs(dt)*kSeulexStepFactor5;
                break;
            }

            if (k == 0)
            {
                y = ySequence;
            }
            else
            {
                table.column(k-1) = ySequence;
            }

            if (k != 0)
            {
                table.extrapolate(k, y);
                scalar err = 0;
                #pragma unroll 4
                for (label i=0; i<kSystemSize; ++i)
                {
                    scale[i] = absTol + relTol*fabs(y0[i]);
                    err += sqr((y[i] - table(i, 0))/scale[i]);
                }
                err = sqrt(err/kSystemSize);
                if (err > 1/SMALL || (k > 1 && err >= errOld))
                {
                    resources->reject = true;
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
                dtOpt[k] = fabs(dt*fac);
                temp[k] = kSeulexWorkEstimate[k]/dtOpt[k];

                if ((resources->first || resources->last) && err <= 1)
                {
                    break;
                }

                if
                (
                    k == kTarg - 1
                && !resources->prevReject
                && !resources->first && !resources->last
                )
                {
                    if (err <= 1)
                    {
                        break;
                    }
                    else if (err > kSeulexStepSequence[kTarg]*kSeulexStepSequence[kTarg + 1]*4)
                    {
                        resources->reject = true;
                        kTarg = k;
                        if (kTarg>1 && temp[k-1] < kSeulexKFactor1*temp[k])
                        {
                            kTarg--;
                        }
                        dtNew = dtOpt[kTarg];
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
                        resources->reject = true;
                        if (kTarg>1 && temp[k-1] < kSeulexKFactor1*temp[k])
                        {
                            kTarg--;
                        }
                        dtNew = dtOpt[kTarg];
                        break;
                    }
                }

                if (k == kTarg+1)
                {
                    if (err > 1)
                    {
                        resources->reject = true;
                        if
                        (
                            kTarg > 1
                        && temp[kTarg-1] < kSeulexKFactor1*temp[kTarg]
                        )
                        {
                            kTarg--;
                        }
                        dtNew = dtOpt[kTarg];
                    }
                    break;
                }
            }
        }
        if (resources->reject)
        {
            resources->prevReject = true;
            if (!jacUpdated)
            {
                theta = 2*jacRedo;

                if (theta > jacRedo && !jacUpdated)
                {
                    ode->jacobian(t, resources->currentParameter(0), y, dfdy);
                    jacUpdated = true;
                }
            }
        }

    }
    jacUpdated = false;

    resources->deltaTDid = dt;
    t += dt;

    label kopt;
    if (k == 1)
    {
        kopt = 2;
    }
    else if (k <= kTarg)
    {
        kopt=k;
        if (temp[k-1] < kSeulexKFactor1*temp[k])
        {
            kopt = k - 1;
        }
        else if (temp[k] < kSeulexKFactor2*temp[k - 1])
        {
            kopt = min(k + 1, kSeulexMaxOrder - 1);
        }
    }
    else
    {
        kopt = k - 1;
        if (k > 2 && temp[k-2] < kSeulexKFactor1*temp[k - 1])
        {
            kopt = k - 2;
        }
        if (temp[k] < kSeulexKFactor2*temp[kopt])
        {
            kopt = min(k, kSeulexMaxOrder - 1);
        }
    }

    if (resources->prevReject)
    {
        kTarg = min(kopt, k);
        dtNew = min(fabs(dt), dtOpt[kTarg]);
        resources->prevReject = false;
    }
    else
    {
        if (kopt <= k)
        {
            dtNew = dtOpt[kopt];
        }
        else
        {
            if (k < kTarg && temp[k] < kSeulexKFactor2*temp[k - 1])
            {
                dtNew = dtOpt[k]*kSeulexWorkEstimate[kopt + 1]/kSeulexWorkEstimate[k];
            }
            else
            {
                dtNew = dtOpt[k]*kSeulexWorkEstimate[kopt]/kSeulexWorkEstimate[k];
            }
        }
        kTarg = kopt;
    }

    resources->deltaTTry = resources->forward ? dtNew : -dtNew;

    // the step is already accepted and sized; nothing for adaptiveStep to judge
    return 0.0;
}
