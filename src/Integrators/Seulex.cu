
template<class ODESystem>
__device__
bool seul (
    kodes::SeulexDeviceResources* res,
    ODESystem* ode,
    const scalar x0,
    const scalar dxTot,
    const label k,
    scalar theta
)
{
    scalar* dfdy_  = res->dfdy();
    scalar* a_     = res->a();
    label* pivotIndices_ = res->pivotIndices();
    
    scalar* y0_    = res->y0();
    scalar* scale = res->scale();
    
    scalar* dy_    = res->dy();
    scalar* yTemp_ = res->yTemp();
    scalar* dydx_  = res->dydx();
    scalar* y      = res->ySequence();

    label nSteps = nSeq_[k];
    scalar dx = dxTot/nSteps;
    
    for (label i=0; i<res->sizeOfSystem(); i++)
    { 
        for (label j=0; j<res->sizeOfSystem(); j++)
        {
            a_[INDEXMAT(i, j, res->sizeOfSystem())] = -dfdy_[INDEXMAT(i, j, res->sizeOfSystem())];
        }
        a_[INDEXMAT(i, i, res->sizeOfSystem())] += 1/dx;
    }
    
    LUDecompose(a_, pivotIndices_, res->sizeOfSystem());

    scalar xnew = x0 + dx;
    ode->derivatives(xnew, res->parameters[INDEXVEC(0)], y0_, dy_);

    LUBacksubstitute(a_, pivotIndices_, dy_, res->sizeOfSystem());

    copyVec(yTemp_, y0_, res->sizeOfSystem());

    for (label nn=1; nn<nSteps; nn++)
    {
        sumVec(yTemp_, yTemp_, dy_, res->sizeOfSystem());

        xnew += dx;

        if (nn == 1 && k<=1)
        {
            scalar dy1 = 0;
            for (label i=0; i<res->sizeOfSystem(); i++)
            {
                dy1 += sqr(dy_[INDEXVEC(i)]/scale[INDEXVEC(i)]);
            }
            dy1 = sqrt(dy1);

            ode->derivatives(x0 + dx, res->parameters[INDEXVEC(0)], yTemp_, dydx_);
            for (label i=0; i<res->sizeOfSystem(); i++)
            {
                dy_[INDEXVEC(i)] = dydx_[INDEXVEC(i)] - dy_[INDEXVEC(i)]/dx;
            }

            LUBacksubstitute(a_, pivotIndices_, dy_, res->sizeOfSystem());

            const scalar denom = min(1.0, dy1 + SMALL);
            scalar dy2 = 0;
            for (label i=0; i<res->sizeOfSystem(); i++)
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

        ode->derivatives(xnew, res->parameters[INDEXVEC(0)], yTemp_, dy_);
        LUBacksubstitute(a_, pivotIndices_, dy_, res->sizeOfSystem());
    }

    sumVec(y, yTemp_, dy_, res->sizeOfSystem());

    return true;
}

template<class ODESystem>
__global__
void seulex_solve(ODESystem* ode, kodes::SeulexDeviceResources* res, stepState step)
{
    if ((INDEXVEC(0) < res->numOfSystems()) && (res->vectors[INDEXVEC(0)] > 0))
    {
        scalar theta_, logTol;
        label kTarg_;


        scalar* table_ = res->table();
        scalar* dfdx_  = res->dfdx();
        scalar* dfdy_  = res->dfdy();
        
        
        scalar* dxOpt_ = res->dxOpt();
        scalar* temp_  = res->temp();
        scalar* y0_    = res->y0();
        scalar* ySequence_ = res->ySequence();
        scalar* scale_ = res->scale();
        
        scalar* y      = res->vectors;

        scalar x = 0;
        scalar xEnd = step.dxTry;
        scalar dx = step.dxTry;

        do
        {
            temp_[INDEXVEC(0)] = GREAT;
            dx = step.dxTry;
            copyVec(y0_, y, res->sizeOfSystem());
            dxOpt_[INDEXVEC(0)] = fabs(0.1*dx);

            if (step.first || step.prevReject)
            {
                theta_ = 2*jacRedo_;
            }

            if (step.first)
            {
                logTol = -log10(relTol_ + absTol_)*0.6 + 0.5;
                kTarg_ = max(1, min(kMaxx_ - 1, label(logTol)));
            }

            for (label i=0; i < res->sizeOfSystem(); ++i)
            {
                scale_[INDEXVEC(i)] = absTol_ + relTol_*fabs(y[INDEXVEC(i)]);
            }

            bool jacUpdated = false;

            if (theta_ > jacRedo_)
            {
                ode->jacobian(x, res->parameters[INDEXVEC(0)], y, dfdx_, dfdy_);
                jacUpdated = true;
            }

            label k;
            scalar dxNew = fabs(dx);
            bool firstk = true;

            while (firstk || step.reject)
            {
                dx = step.forward ? dxNew : -dxNew;
                firstk = false;
                step.reject = false;

                if (fabs(dx) <= fabs(x) * sqr(SMALL))
                {
                    printf("step size underflow : %0.16f \n", dx);
                }

                scalar errOld = 0;

                for (k=0; k<=kTarg_+1; k++)
                {
                    bool success = seul(res, ode, x, dx, k, theta_);

                    if (!success)
                    {
                        step.reject = true;
                        dxNew = fabs(dx)*stepFactor5_;
                        break;
                    }

                    if (k == 0)
                    {
                        copyVec(y, ySequence_, res->sizeOfSystem());
                    }
                    else
                    {
                        for (label i=0; i<res->sizeOfSystem(); ++i)
                        {
                            table_[INDEXMAT(i, k-1, res->sizeOfSystem())] = ySequence_[INDEXVEC(i)];
                        }
                    }

                    if (k != 0)
                    {
                        extrapolate(k, res->sizeOfSystem(), table_, y);
                        scalar err = 0;
                        for (label i=0; i<res->sizeOfSystem(); ++i)
                        {
                            scale_[INDEXVEC(i)] = absTol_ + relTol_*fabs(y0_[INDEXVEC(i)]);
                            err += sqr((y[INDEXVEC(i)] - table_[INDEXMAT(i, 0, res->sizeOfSystem())])/scale_[INDEXVEC(i)]);
                        }
                        err = sqrt(err/res->sizeOfSystem());
                        if (err > 1/SMALL || (k > 1 && err >= errOld))
                        {
                            step.reject = true;
                            dxNew = fabs(dx)*stepFactor5_;
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
                        dxOpt_[INDEXVEC(k)] = fabs(dx*fac);
                        temp_[INDEXVEC(k)] = cpu_[k]/dxOpt_[INDEXVEC(k)];

                        if ((step.first || step.last) && err <= 1)
                        {
                            break;
                        }

                        if
                        (
                            k == kTarg_ - 1
                        && !step.prevReject
                        && !step.first && !step.last
                        )
                        {
                            if (err <= 1)
                            {
                                break;
                            }
                            else if (err > nSeq_[kTarg_]*nSeq_[kTarg_ + 1]*4)
                            {
                                step.reject = true;
                                kTarg_ = k;
                                if (kTarg_>1 && temp_[INDEXVEC(k-1)] < kFactor1_*temp_[INDEXVEC(k)])
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[INDEXVEC(kTarg_)];
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
                                step.reject = true;
                                if (kTarg_>1 && temp_[INDEXVEC(k-1)] < kFactor1_*temp_[INDEXVEC(k)])
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[INDEXVEC(kTarg_)];
                                break;
                            }
                        }

                        if (k == kTarg_+1)
                        {
                            if (err > 1)
                            {
                                step.reject = true;
                                if
                                (
                                    kTarg_ > 1
                                && temp_[INDEXVEC(kTarg_-1)] < kFactor1_*temp_[INDEXVEC(kTarg_)]
                                )
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[INDEXVEC(kTarg_)];
                            }
                            break;
                        }
                    }
                } 
                if (step.reject)
                {
                    step.prevReject = true;
                    if (!jacUpdated)
                    {
                        theta_ = 2*jacRedo_;

                        if (theta_ > jacRedo_ && !jacUpdated)
                        {
                            ode->jacobian(x, res->parameters[INDEXVEC(0)], y, dfdx_, dfdy_);
                            jacUpdated = true;
                        }
                    }
                }

            }
            jacUpdated = false;
            
            step.dxDid = dx;
            x += dx;

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
            
            if (step.prevReject)
            {
                kTarg_ = min(kopt, k);
                dxNew = min(fabs(dx), dxOpt_[INDEXVEC(kTarg_)]);
                step.prevReject = false;
            }
            else
            {
                if (kopt <= k)
                {
                    dxNew = dxOpt_[INDEXVEC(kopt)];
                }
                else
                {
                    if (k < kTarg_ && temp_[INDEXVEC(k)] < kFactor2_*temp_[INDEXVEC(k - 1)])
                    {
                        dxNew = dxOpt_[INDEXVEC(k)]*cpu_[kopt + 1]/cpu_[k];
                    }
                    else
                    {
                        dxNew = dxOpt_[INDEXVEC(k)]*cpu_[kopt]/cpu_[k];
                    }
                }
                kTarg_ = kopt;
            }
            
            step.dxTry = step.forward ? dxNew : -dxNew;

            for (label i=0; i < res->sizeOfSystem(); ++i)
            {
                y[INDEXVEC(i)] = max(0.0, y[INDEXVEC(i)]);
            }
        } 
        while (x < xEnd);
    }
}

template<class ODESystem>
kodes::Seulex<ODESystem>::Seulex(ODESystem* ode, SeulexDeviceResources* res, label numOfSystems)
: Integrator<ODESystem, SeulexDeviceResources>(ode, res, numOfSystems) {}

template<class ODESystem>
void kodes::Seulex<ODESystem>::solve(stepState step)
{
    seulex_solve<ODESystem><<<this->blocks, this->threads, this->sharedMemSize>>>(this->ode_, this->res_, step);
}

