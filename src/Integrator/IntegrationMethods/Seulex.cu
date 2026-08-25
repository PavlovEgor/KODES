#include "Seulex.cuh"
#include "basicLinalg.cuh"

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
    scalar* __restrict__ dfdy  = resources->dfdy();
    scalar* __restrict__ a     = resources->a();
    label* __restrict__ pivotIndices = resources->pivotIndices();

    scalar* __restrict__ y0    = resources->y0();
    scalar* __restrict__ scale = resources->scale();

    scalar* __restrict__ dy    = resources->dy();
    scalar* __restrict__ yTemp = resources->yTemp();
    scalar* __restrict__ dydt  = resources->dydt();
    scalar* __restrict__ y      = resources->ySequence();

    const label systemSize = resources->systemSize();

    label nSteps = seulexStepSequence[k];
    scalar dt = dtTot/nSteps;

    for (label i=0; i<systemSize; i++)
    {
        for (label j=0; j<systemSize; j++)
        {
            a[INDEXMAT(i, j, systemSize)] = -dfdy[INDEXMAT(i, j, systemSize)];
        }
        a[INDEXMAT(i, i, systemSize)] += 1/dt;
    }

    LUDecompose(a, pivotIndices, systemSize);

    scalar tNew = t0 + dt;
    ode->derivatives(tNew, resources->currentParameter(0), y0, dy);

    LUBacksubstitute(a, pivotIndices, dy, systemSize);

    copyVec(yTemp, y0, systemSize);

    for (label nn=1; nn<nSteps; nn++)
    {
        sumVec(yTemp, yTemp, dy, systemSize);

        tNew += dt;

        if (nn == 1 && k<=1)
        {
            scalar dy1 = 0;
            for (label i=0; i<systemSize; i++)
            {
                dy1 += sqr(dy[INDEXVEC(i)]/scale[INDEXVEC(i)]);
            }
            dy1 = sqrt(dy1);

            ode->derivatives(t0 + dt, resources->currentParameter(0), yTemp, dydt);
            for (label i=0; i<systemSize; i++)
            {
                dy[INDEXVEC(i)] = dydt[INDEXVEC(i)] - dy[INDEXVEC(i)]/dt;
            }

            LUBacksubstitute(a, pivotIndices, dy, systemSize);

            const scalar denom = min(1.0, dy1 + SMALL);
            scalar dy2 = 0;
            for (label i=0; i<systemSize; i++)
            {
                // Test of dy[i] to avoid overflow
                if (fabs(dy[INDEXVEC(i)]) > scale[INDEXVEC(i)]*denom)
                {
                    theta = 1;
                    return false;
                }

                dy2 += sqr(dy[INDEXVEC(i)]/scale[INDEXVEC(i)]);
            }
            dy2 = sqrt(dy2);
            theta = dy2/denom;

            if (theta > 1)
            {
                return false;
            }
        }

        ode->derivatives(tNew, resources->currentParameter(0), yTemp, dy);
        LUBacksubstitute(a, pivotIndices, dy, systemSize);
    }

    sumVec(y, yTemp, dy, systemSize);

    return true;
}

__device__
void kodes::Seulex::extrapolate (const label k, const label systemSize, scalar* __restrict__ table, scalar* __restrict__ y)
{
    for (label j=k-1; j>0; j--)
    {
        for (label i=0; i<systemSize; i++)
        {
            table[INDEXMAT(i, j-1, systemSize)] =
                table[INDEXMAT(i, j, systemSize)] + seulexExtrapolationCoeff[k + j*KODES_SEULEX_TABLE_SIZE]*(table[INDEXMAT(i, j, systemSize)] - table[INDEXMAT(i, j-1, systemSize)]);
        }
    }

    for (label i=0; i<systemSize; i++)
    {
        y[INDEXVEC(i)] = table[INDEXMAT(i, 0, systemSize)] + seulexExtrapolationCoeff[k]*(table[INDEXMAT(i, 0, systemSize)] - y[INDEXVEC(i)]);
    }
}

__device__
scalar kodes::Seulex::step
(
    kodes::ODESystem* ode,
    kodes::DeviceResources* deviceResources,
    kodes::IntegratorControls controls
) const
{
    // safe: the table entry that made this method made these resources
    SeulexDeviceResources* resources =
        static_cast<SeulexDeviceResources*>(deviceResources);

    const label systemSize = resources->systemSize();

    const scalar absTol = controls.absTol;
    const scalar relTol = controls.relTol;

    const scalar jacRedo = min(1e-4, relTol);

    scalar theta, logTol;
    label kTarg;

    scalar* __restrict__ table = resources->table();
    scalar* __restrict__ dfdt  = resources->dfdt();
    scalar* __restrict__ dfdy  = resources->dfdy();

    scalar* __restrict__ dtOpt = resources->dtOpt();
    scalar* __restrict__ temp  = resources->temp();
    scalar* __restrict__ y0    = resources->y0();
    scalar* __restrict__ ySequence = resources->ySequence();
    scalar* __restrict__ scale = resources->scale();

    scalar* __restrict__ y      = resources->currentVector();
    scalar& t      = resources->currentT[INDEXVEC(0)];

    temp[INDEXVEC(0)] = GREAT;
    scalar dt = resources->deltaTTry[INDEXVEC(0)];
    copyVec(y0, y, systemSize);
    dtOpt[INDEXVEC(0)] = fabs(0.1*dt);

    if (resources->first[INDEXVEC(0)] || resources->prevReject[INDEXVEC(0)])
    {
        theta = 2*jacRedo;
    }

    if (resources->first[INDEXVEC(0)] )
    {
        logTol = -log10(relTol + absTol)*0.6 + 0.5;
        kTarg = max(1, min(KODES_SEULEX_MAX_ORDER - 1, label(logTol)));
    }

    for (label i=0; i < systemSize; ++i)
    {
        scale[INDEXVEC(i)] = absTol + relTol*fabs(y[INDEXVEC(i)]);
    }

    bool jacUpdated = false;

    if (theta > jacRedo)
    {
        ode->jacobian(t, resources->currentParameter(0), y, dfdt, dfdy);
        jacUpdated = true;
    }

    label k;
    scalar dtNew = fabs(dt);
    bool firstk = true;

    while (firstk || resources->reject[INDEXVEC(0)])
    {
        dt = resources->forward[INDEXVEC(0)] ? dtNew : -dtNew;
        firstk = false;
        resources->reject[INDEXVEC(0)] = false;

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
                resources->reject[INDEXVEC(0)] = true;
                dtNew = fabs(dt)*seulexStepFactor5;
                break;
            }

            if (k == 0)
            {
                copyVec(y, ySequence, systemSize);
            }
            else
            {
                for (label i=0; i<systemSize; ++i)
                {
                    table[INDEXMAT(i, k-1, systemSize)] = ySequence[INDEXVEC(i)];
                }
            }

            if (k != 0)
            {
                extrapolate(k, systemSize, table, y);
                scalar err = 0;
                for (label i=0; i<systemSize; ++i)
                {
                    scale[INDEXVEC(i)] = absTol + relTol*fabs(y0[INDEXVEC(i)]);
                    err += sqr((y[INDEXVEC(i)] - table[INDEXMAT(i, 0, systemSize)])/scale[INDEXVEC(i)]);
                }
                err = sqrt(err/systemSize);
                if (err > 1/SMALL || (k > 1 && err >= errOld))
                {
                    resources->reject[INDEXVEC(0)] = true;
                    dtNew = fabs(dt)*seulexStepFactor5;
                    break;
                }
                errOld = min(4*err, 1.0);
                scalar expo = 1.0/(k + 1);
                scalar facmin = pow(seulexStepFactor3, expo);
                scalar fac;
                if (err == 0)
                {
                    fac = 1/facmin;
                }
                else
                {
                    fac = seulexStepFactor2/pow(err/seulexStepFactor1, expo);
                    fac = max(facmin/seulexStepFactor4, min(1/facmin, fac));
                }
                dtOpt[INDEXVEC(k)] = fabs(dt*fac);
                temp[INDEXVEC(k)] = seulexWorkEstimate[k]/dtOpt[INDEXVEC(k)];

                if ((resources->first[INDEXVEC(0)] || resources->last[INDEXVEC(0)]) && err <= 1)
                {
                    break;
                }

                if
                (
                    k == kTarg - 1
                && !resources->prevReject[INDEXVEC(0)]
                && !resources->first[INDEXVEC(0)] && !resources->last[INDEXVEC(0)]
                )
                {
                    if (err <= 1)
                    {
                        break;
                    }
                    else if (err > seulexStepSequence[kTarg]*seulexStepSequence[kTarg + 1]*4)
                    {
                        resources->reject[INDEXVEC(0)] = true;
                        kTarg = k;
                        if (kTarg>1 && temp[INDEXVEC(k-1)] < seulexKFactor1*temp[INDEXVEC(k)])
                        {
                            kTarg--;
                        }
                        dtNew = dtOpt[INDEXVEC(kTarg)];
                        break;
                    }
                }

                if (k == kTarg)
                {
                    if (err <= 1)
                    {
                        break;
                    }
                    else if (err > seulexStepSequence[k + 1]*2)
                    {
                        resources->reject[INDEXVEC(0)] = true;
                        if (kTarg>1 && temp[INDEXVEC(k-1)] < seulexKFactor1*temp[INDEXVEC(k)])
                        {
                            kTarg--;
                        }
                        dtNew = dtOpt[INDEXVEC(kTarg)];
                        break;
                    }
                }

                if (k == kTarg+1)
                {
                    if (err > 1)
                    {
                        resources->reject[INDEXVEC(0)] = true;
                        if
                        (
                            kTarg > 1
                        && temp[INDEXVEC(kTarg-1)] < seulexKFactor1*temp[INDEXVEC(kTarg)]
                        )
                        {
                            kTarg--;
                        }
                        dtNew = dtOpt[INDEXVEC(kTarg)];
                    }
                    break;
                }
            }
        }
        if (resources->reject[INDEXVEC(0)])
        {
            resources->prevReject[INDEXVEC(0)] = true;
            if (!jacUpdated)
            {
                theta = 2*jacRedo;

                if (theta > jacRedo && !jacUpdated)
                {
                    ode->jacobian(t, resources->currentParameter(0), y, dfdt, dfdy);
                    jacUpdated = true;
                }
            }
        }

    }
    jacUpdated = false;

    resources->deltaTDid[INDEXVEC(0)] = dt;
    t += dt;

    label kopt;
    if (k == 1)
    {
        kopt = 2;
    }
    else if (k <= kTarg)
    {
        kopt=k;
        if (temp[INDEXVEC(k-1)] < seulexKFactor1*temp[INDEXVEC(k)])
        {
            kopt = k - 1;
        }
        else if (temp[INDEXVEC(k)] < seulexKFactor2*temp[INDEXVEC(k - 1)])
        {
            kopt = min(k + 1, KODES_SEULEX_MAX_ORDER - 1);
        }
    }
    else
    {
        kopt = k - 1;
        if (k > 2 && temp[INDEXVEC(k-2)] < seulexKFactor1*temp[INDEXVEC(k - 1)])
        {
            kopt = k - 2;
        }
        if (temp[INDEXVEC(k)] < seulexKFactor2*temp[INDEXVEC(kopt)])
        {
            kopt = min(k, KODES_SEULEX_MAX_ORDER - 1);
        }
    }

    if (resources->prevReject[INDEXVEC(0)])
    {
        kTarg = min(kopt, k);
        dtNew = min(fabs(dt), dtOpt[INDEXVEC(kTarg)]);
        resources->prevReject[INDEXVEC(0)] = false;
    }
    else
    {
        if (kopt <= k)
        {
            dtNew = dtOpt[INDEXVEC(kopt)];
        }
        else
        {
            if (k < kTarg && temp[INDEXVEC(k)] < seulexKFactor2*temp[INDEXVEC(k - 1)])
            {
                dtNew = dtOpt[INDEXVEC(k)]*seulexWorkEstimate[kopt + 1]/seulexWorkEstimate[k];
            }
            else
            {
                dtNew = dtOpt[INDEXVEC(k)]*seulexWorkEstimate[kopt]/seulexWorkEstimate[k];
            }
        }
        kTarg = kopt;
    }

    resources->deltaTTry[INDEXVEC(0)] = resources->forward[INDEXVEC(0)] ? dtNew : -dtNew;

    // the step is already accepted and sized; nothing for adaptiveStep to judge
    return 0.0;
}
