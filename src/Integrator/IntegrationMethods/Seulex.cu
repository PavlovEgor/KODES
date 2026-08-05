
template<class ODESystem>
__device__
bool kodes::Seulex<ODESystem>::seul (
    kodes::SeulexDeviceResources* resources,
    ODESystem* ode,
    const scalar x0,
    const scalar dtTot,
    const label k,
    scalar& theta
)
{
    scalar* dfdy_  = resources->dfdy();
    scalar* a_     = resources->a();
    label* pivotIndices_ = resources->pivotIndices();
    
    scalar* y0_    = resources->y0();
    scalar* scale = resources->scale();
    
    scalar* dy_    = resources->dy();
    scalar* yTemp_ = resources->yTemp();
    scalar* dydt_  = resources->dydt();
    scalar* y      = resources->ySequence();

    label nSteps = nSeq_[k];
    scalar dt = dtTot/nSteps;
    
    for (label i=0; i<resources->systemSize(); i++)
    { 
        for (label j=0; j<resources->systemSize(); j++)
        {
            a_[INDEXMAT(i, j, resources->systemSize())] = -dfdy_[INDEXMAT(i, j, resources->systemSize())];
        }
        a_[INDEXMAT(i, i, resources->systemSize())] += 1/dt;
    }
    
    LUDecompose(a_, pivotIndices_, resources->systemSize());

    scalar xnew = x0 + dt;
    ode->derivatives(xnew, resources->parameters[INDEXVEC(0)], y0_, dy_);

    LUBacksubstitute(a_, pivotIndices_, dy_, resources->systemSize());

    copyVec(yTemp_, y0_, resources->systemSize());

    for (label nn=1; nn<nSteps; nn++)
    {
        sumVec(yTemp_, yTemp_, dy_, resources->systemSize());

        xnew += dt;

        if (nn == 1 && k<=1)
        {
            scalar dy1 = 0;
            for (label i=0; i<resources->systemSize(); i++)
            {
                dy1 += sqr(dy_[INDEXVEC(i)]/scale[INDEXVEC(i)]);
            }
            dy1 = sqrt(dy1);

            ode->derivatives(x0 + dt, resources->parameters[INDEXVEC(0)], yTemp_, dydt_);
            for (label i=0; i<resources->systemSize(); i++)
            {
                dy_[INDEXVEC(i)] = dydt_[INDEXVEC(i)] - dy_[INDEXVEC(i)]/dt;
            }

            LUBacksubstitute(a_, pivotIndices_, dy_, resources->systemSize());

            const scalar denom = min(1.0, dy1 + SMALL);
            scalar dy2 = 0;
            for (label i=0; i<resources->systemSize(); i++)
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

        ode->derivatives(xnew, resources->parameters[INDEXVEC(0)], yTemp_, dy_);
        LUBacksubstitute(a_, pivotIndices_, dy_, resources->systemSize());
    }

    sumVec(y, yTemp_, dy_, resources->systemSize());

    return true;
}

template<class ODESystem>
__device__
void kodes::Seulex<ODESystem>::extrapolate (const label k,const label sizeOfSystem, scalar* table, scalar* y)
{
    for (label j=k-1; j>0; j--)
    {
        for (label i=0; i<sizeOfSystem; i++)
        {
            table[INDEXMAT(i, j-1, sizeOfSystem)] =
                table[INDEXMAT(i, j, sizeOfSystem)] + coeff_[k][j]*(table[INDEXMAT(i, j, sizeOfSystem)] - table[INDEXMAT(i, j-1, sizeOfSystem)]);
        }
    }

    for (label i=0; i<sizeOfSystem; i++)
    {
        y[INDEXVEC(i)] = table[INDEXMAT(i, 0, sizeOfSystem)] + coeff_[k][0]*(table[INDEXMAT(i, 0, sizeOfSystem)] - y[INDEXVEC(i)]);
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
    const scalar absTol_ = controls.absTol;
    const scalar relTol_ = controls.relTol;
    

    const scalar jacRedo_ = min(1e-4, relTol_);

    scalar theta_, logTol;
    label kTarg_;

    scalar* table_ = resources->table();
    scalar* dfdt_  = resources->dfdt();
    scalar* dfdy_  = resources->dfdy();
    
    
    scalar* dtOpt_ = resources->dtOpt();
    scalar* temp_  = resources->temp();
    scalar* y0_    = resources->y0();
    scalar* ySequence_ = resources->ySequence();
    scalar* scale_ = resources->scale();
    
    scalar* y      = resources->vectors;

    scalar& t = resources->currentT[INDEXVEC(0)];

    temp_[INDEXVEC(0)] = GREAT;
    scalar dt = resources->deltaTTry[INDEXVEC(0)];
    copyVec(y0_, y, resources->systemSize());
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

    for (label i=0; i < resources->systemSize(); ++i)
    {
        scale_[INDEXVEC(i)] = absTol_ + relTol_*fabs(y[INDEXVEC(i)]);
    }

    bool jacUpdated = false;

    if (theta_ > jacRedo_)
    {
        ode->jacobian(t, resources->parameters[INDEXVEC(0)], y, dfdt_, dfdy_);
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
                copyVec(y, ySequence_, resources->systemSize());
            }
            else
            {
                for (label i=0; i<resources->systemSize(); ++i)
                {
                    table_[INDEXMAT(i, k-1, resources->systemSize())] = ySequence_[INDEXVEC(i)];
                }
            }

            if (k != 0)
            {
                extrapolate(k, resources->systemSize(), table_, y);
                scalar err = 0;
                for (label i=0; i<resources->systemSize(); ++i)
                {
                    scale_[INDEXVEC(i)] = absTol_ + relTol_*fabs(y0_[INDEXVEC(i)]);
                    err += sqr((y[INDEXVEC(i)] - table_[INDEXMAT(i, 0, resources->systemSize())])/scale_[INDEXVEC(i)]);
                }
                err = sqrt(err/resources->systemSize());
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
                temp_[INDEXVEC(k)] = cpu_[k]/dtOpt_[INDEXVEC(k)];

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
                    ode->jacobian(t, resources->parameters[INDEXVEC(0)], y, dfdt_, dfdy_);
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
                dtNew = dtOpt_[INDEXVEC(k)]*cpu_[kopt + 1]/cpu_[k];
            }
            else
            {
                dtNew = dtOpt_[INDEXVEC(k)]*cpu_[kopt]/cpu_[k];
            }
        }
        kTarg_ = kopt;
    }
    
    resources->deltaTTry[INDEXVEC(0)] = resources->forward[INDEXVEC(0)] ? dtNew : -dtNew;
}



