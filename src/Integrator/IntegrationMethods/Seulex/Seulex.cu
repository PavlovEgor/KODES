
template<class ODESystem>
__device__
bool kodes::Seulex<ODESystem>::seul (
    kodes::SeulexDeviceResources* resources,
    ODESystem* ode,
    const scalar t0,
    const scalar dtTot,
    const label k,
    scalar& theta
)
{
    scalar* __restrict__ dfdy_  = resources->dfdy();
    scalar* __restrict__ a_     = resources->a();
    label* __restrict__ pivotIndices_ = resources->pivotIndices();
    
    scalar* __restrict__ y0_    = resources->y0();
    scalar* __restrict__ scale = resources->scale();
    
    scalar* __restrict__ dy_    = resources->dy();
    scalar* __restrict__ yTemp_ = resources->yTemp();
    scalar* __restrict__ dydt_  = resources->dydt();
    scalar* __restrict__ y      = resources->ySequence();

    const label systemSize = resources->systemSize();

    label nSteps = nSeq_[k];
    scalar dt = dtTot/nSteps;
    
    for (label i=0; i<systemSize; i++)
    { 
        for (label j=0; j<systemSize; j++)
        {
            a_[INDEXMAT(i, j, systemSize)] = -dfdy_[INDEXMAT(i, j, systemSize)];
        }
        a_[INDEXMAT(i, i, systemSize)] += 1/dt;
    }
    
    LUDecompose(a_, pivotIndices_, systemSize);

    scalar tNew = t0 + dt;
    ode->derivatives(tNew, resources->currentParameter(0), y0_, dy_);

    LUBacksubstitute(a_, pivotIndices_, dy_, systemSize);

    copyVec(yTemp_, y0_, systemSize);

    for (label nn=1; nn<nSteps; nn++)
    {
        sumVec(yTemp_, yTemp_, dy_, systemSize);

        tNew += dt;

        if (nn == 1 && k<=1)
        {
            scalar dy1 = 0;
            for (label i=0; i<systemSize; i++)
            {
                dy1 += sqr(dy_[INDEXVEC(i)]/scale[INDEXVEC(i)]);
            }
            dy1 = sqrt(dy1);

            ode->derivatives(t0 + dt, resources->currentParameter(0), yTemp_, dydt_);
            for (label i=0; i<systemSize; i++)
            {
                dy_[INDEXVEC(i)] = dydt_[INDEXVEC(i)] - dy_[INDEXVEC(i)]/dt;
            }

            LUBacksubstitute(a_, pivotIndices_, dy_, systemSize);

            const scalar denom = min(1.0, dy1 + SMALL);
            scalar dy2 = 0;
            for (label i=0; i<systemSize; i++)
            {
                // Test of dy_[i] to avoid overflow
                if (fabs(dy_[INDEXVEC(i)]) > scale[INDEXVEC(i)]*denom)
                {
                    theta = 1;
                    return false;
                }

                dy2 += sqr(dy_[INDEXVEC(i)]/scale[INDEXVEC(i)]);
            }
            dy2 = sqrt(dy2);
            theta = dy2/denom;

            if (theta > 1)
            {
                return false;
            }
        }

        ode->derivatives(tNew, resources->currentParameter(0), yTemp_, dy_);
        LUBacksubstitute(a_, pivotIndices_, dy_, systemSize);
    }

    sumVec(y, yTemp_, dy_, systemSize);

    return true;
}

template<class ODESystem>
__device__
void kodes::Seulex<ODESystem>::extrapolate (const label k, const label systemSize, scalar* __restrict__ table, scalar* __restrict__ y)
{
    for (label j=k-1; j>0; j--)
    {
        for (label i=0; i<systemSize; i++)
        {
            table[INDEXMAT(i, j-1, systemSize)] =
                table[INDEXMAT(i, j, systemSize)] + coeff_[k + j*iMaxx_]*(table[INDEXMAT(i, j, systemSize)] - table[INDEXMAT(i, j-1, systemSize)]);
        }
    }

    for (label i=0; i<systemSize; i++)
    {
        y[INDEXVEC(i)] = table[INDEXMAT(i, 0, systemSize)] + coeff_[k]*(table[INDEXMAT(i, 0, systemSize)] - y[INDEXVEC(i)]);
    }
}

template<class ODESystem>
__device__
void kodes::Seulex<ODESystem>::step
(
    ODESystem* ode,
    kodes::SeulexDeviceResources* resources,
    kodes::IntegratorControls controls
)
{
    const label systemSize = resources->systemSize();

    const scalar absTol_ = controls.absTol;
    const scalar relTol_ = controls.relTol;

    const scalar jacRedo_ = min(1e-4, relTol_);

    scalar theta_, logTol;
    label kTarg_;

    scalar* __restrict__ table_ = resources->table();
    scalar* __restrict__ dfdt_  = resources->dfdt();
    scalar* __restrict__ dfdy_  = resources->dfdy();
    
    scalar* __restrict__ dtOpt_ = resources->dtOpt();
    scalar* __restrict__ temp_  = resources->temp();
    scalar* __restrict__ y0_    = resources->y0();
    scalar* __restrict__ ySequence_ = resources->ySequence();
    scalar* __restrict__ scale_ = resources->scale();
    
    scalar* __restrict__ y      = resources->currentVector();
    scalar& t      = resources->currentT[INDEXVEC(0)];

    temp_[INDEXVEC(0)] = GREAT;
    scalar dt = resources->deltaTTry[INDEXVEC(0)];
    copyVec(y0_, y, systemSize);
    dtOpt_[INDEXVEC(0)] = fabs(0.1*dt);

    if (resources->first[INDEXVEC(0)] || resources->prevReject[INDEXVEC(0)])
    {
        theta_ = 2*jacRedo_;
    }

    if (resources->first[INDEXVEC(0)] )
    {
        logTol = -log10(relTol_ + absTol_)*0.6 + 0.5;
        kTarg_ = max(1, min(kMaxx_ - 1, label(logTol)));
    }

    for (label i=0; i < systemSize; ++i)
    {
        scale_[INDEXVEC(i)] = absTol_ + relTol_*fabs(y[INDEXVEC(i)]);
    }

    bool jacUpdated = false;

    if (theta_ > jacRedo_)
    {
        ode->jacobian(t, resources->currentParameter(0), y, dfdt_, dfdy_);
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

        for (k=0; k<=kTarg_+1; k++)
        {
            bool success = seul(resources, ode, t, dt, k, theta_);

            if (!success)
            {
                resources->reject[INDEXVEC(0)] = true;
                dtNew = fabs(dt)*stepFactor5_;
                break;
            }

            if (k == 0)
            {
                copyVec(y, ySequence_, systemSize);
            }
            else
            {
                for (label i=0; i<systemSize; ++i)
                {
                    table_[INDEXMAT(i, k-1, systemSize)] = ySequence_[INDEXVEC(i)];
                }
            }

            if (k != 0)
            {
                extrapolate(k, systemSize, table_, y);
                scalar err = 0;
                for (label i=0; i<systemSize; ++i)
                {
                    scale_[INDEXVEC(i)] = absTol_ + relTol_*fabs(y0_[INDEXVEC(i)]);
                    err += sqr((y[INDEXVEC(i)] - table_[INDEXMAT(i, 0, systemSize)])/scale_[INDEXVEC(i)]);
                }
                err = sqrt(err/systemSize);
                if (err > 1/SMALL || (k > 1 && err >= errOld))
                {
                    resources->reject[INDEXVEC(0)] = true;
                    dtNew = fabs(dt)*stepFactor5_;
                    break;
                }
                errOld = min(4*err, 1.0);
                scalar expo = 1.0/(k + 1);
                scalar facmin = pow(stepFactor3_, expo);
                scalar fac;
                if (err == 0)
                {
                    fac = 1/facmin;
                }
                else
                {
                    fac = stepFactor2_/pow(err/stepFactor1_, expo);
                    fac = max(facmin/stepFactor4_, min(1/facmin, fac));
                }
                dtOpt_[INDEXVEC(k)] = fabs(dt*fac);
                temp_[INDEXVEC(k)] = gpu_[k]/dtOpt_[INDEXVEC(k)];

                if ((resources->first[INDEXVEC(0)] || resources->last[INDEXVEC(0)]) && err <= 1)
                {
                    break;
                }

                if
                (
                    k == kTarg_ - 1
                && !resources->prevReject[INDEXVEC(0)]
                && !resources->first[INDEXVEC(0)] && !resources->last[INDEXVEC(0)]
                )
                {
                    if (err <= 1)
                    {
                        break;
                    }
                    else if (err > nSeq_[kTarg_]*nSeq_[kTarg_ + 1]*4)
                    {
                        resources->reject[INDEXVEC(0)] = true;
                        kTarg_ = k;
                        if (kTarg_>1 && temp_[INDEXVEC(k-1)] < kFactor1_*temp_[INDEXVEC(k)])
                        {
                            kTarg_--;
                        }
                        dtNew = dtOpt_[INDEXVEC(kTarg_)];
                        break;
                    }
                }

                if (k == kTarg_)
                {
                    if (err <= 1)
                    {
                        break;
                    }
                    else if (err > nSeq_[k + 1]*2)
                    {
                        resources->reject[INDEXVEC(0)] = true;
                        if (kTarg_>1 && temp_[INDEXVEC(k-1)] < kFactor1_*temp_[INDEXVEC(k)])
                        {
                            kTarg_--;
                        }
                        dtNew = dtOpt_[INDEXVEC(kTarg_)];
                        break;
                    }
                }

                if (k == kTarg_+1)
                {
                    if (err > 1)
                    {
                        resources->reject[INDEXVEC(0)] = true;
                        if
                        (
                            kTarg_ > 1
                        && temp_[INDEXVEC(kTarg_-1)] < kFactor1_*temp_[INDEXVEC(kTarg_)]
                        )
                        {
                            kTarg_--;
                        }
                        dtNew = dtOpt_[INDEXVEC(kTarg_)];
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
                theta_ = 2*jacRedo_;

                if (theta_ > jacRedo_ && !jacUpdated)
                {
                    ode->jacobian(t, resources->currentParameter(0), y, dfdt_, dfdy_);
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
    else if (k <= kTarg_)
    {
        kopt=k;
        if (temp_[INDEXVEC(k-1)] < kFactor1_*temp_[INDEXVEC(k)])
        {
            kopt = k - 1;
        }
        else if (temp_[INDEXVEC(k)] < kFactor2_*temp_[INDEXVEC(k - 1)])
        {
            kopt = min(k + 1, kMaxx_ - 1);
        }
    }
    else
    {
        kopt = k - 1;
        if (k > 2 && temp_[INDEXVEC(k-2)] < kFactor1_*temp_[INDEXVEC(k - 1)])
        {
            kopt = k - 2;
        }
        if (temp_[INDEXVEC(k)] < kFactor2_*temp_[INDEXVEC(kopt)])
        {
            kopt = min(k, kMaxx_ - 1);
        }
    }
    
    if (resources->prevReject[INDEXVEC(0)])
    {
        kTarg_ = min(kopt, k);
        dtNew = min(fabs(dt), dtOpt_[INDEXVEC(kTarg_)]);
        resources->prevReject[INDEXVEC(0)] = false;
    }
    else
    {
        if (kopt <= k)
        {
            dtNew = dtOpt_[INDEXVEC(kopt)];
        }
        else
        {
            if (k < kTarg_ && temp_[INDEXVEC(k)] < kFactor2_*temp_[INDEXVEC(k - 1)])
            {
                dtNew = dtOpt_[INDEXVEC(k)]*gpu_[kopt + 1]/gpu_[k];
            }
            else
            {
                dtNew = dtOpt_[INDEXVEC(k)]*gpu_[kopt]/gpu_[k];
            }
        }
        kTarg_ = kopt;
    }
    
    resources->deltaTTry[INDEXVEC(0)] = resources->forward[INDEXVEC(0)] ? dtNew : -dtNew;
}

