
template<class ODESystem>
__device__
bool seul (
    kodes::SeulexDeviceResources* resources,
    ODESystem* ode,
    const scalar x0,
    const scalar dtTot,
    const label k,
    scalar& theta,
    SeulexProfile& profile
)
{
    ++profile.nSeul;

    scalar* hess_  = resources->dfdy();
    scalar* tau_   = resources->hessTau();
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

    // The Jacobian was reduced to Hessenberg form when it was evaluated, so
    // the stage matrix 1/dt I - J is factorised in O(n^2) and no O(n^3) work
    // is repeated per stage
    long long tProfile = clock64();
    hessenbergShiftedFactorise(hess_, 1/dt, a_, pivotIndices_, resources->systemSize());
    profile.stageFactorise += clock64() - tProfile;
    ++profile.nStageFactorise;

    scalar xnew = x0 + dt;

    tProfile = clock64();
    ode->derivatives(xnew, resources->parameters[INDEXVEC(0)], y0_, dy_);
    profile.derivatives += clock64() - tProfile;
    ++profile.nDerivatives;

    tProfile = clock64();
    hessenbergSolve(hess_, tau_, a_, pivotIndices_, dy_, resources->systemSize());
    profile.stageSolve += clock64() - tProfile;
    ++profile.nStageSolve;

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

            tProfile = clock64();
            ode->derivatives(x0 + dt, resources->parameters[INDEXVEC(0)], yTemp_, dydt_);
            profile.derivatives += clock64() - tProfile;
            ++profile.nDerivatives;

            for (label i=0; i<resources->systemSize(); i++)
            {
                dy_[INDEXVEC(i)] = dydt_[INDEXVEC(i)] - dy_[INDEXVEC(i)]/dt;
            }

            tProfile = clock64();
            hessenbergSolve(hess_, tau_, a_, pivotIndices_, dy_, resources->systemSize());
            profile.stageSolve += clock64() - tProfile;
            ++profile.nStageSolve;

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

        tProfile = clock64();
        ode->derivatives(xnew, resources->parameters[INDEXVEC(0)], yTemp_, dy_);
        profile.derivatives += clock64() - tProfile;
        ++profile.nDerivatives;

        tProfile = clock64();
        hessenbergSolve(hess_, tau_, a_, pivotIndices_, dy_, resources->systemSize());
        profile.stageSolve += clock64() - tProfile;
        ++profile.nStageSolve;
    }

    sumVec(y, yTemp_, dy_, resources->systemSize());

    return true;
}

template<class ODESystem>
__global__
void seulex_solve
(
    ODESystem* ode,
    kodes::SeulexDeviceResources* resources,
    scalar deltaT,
    label realBatchSize,
    kodes::IntegratorControls controls,
    label profileSystem,
    bool    firstBatch
)
{
    if ((INDEXVEC(0) < realBatchSize) && (resources->vectors[INDEXVEC(0)] > 0))
    {
        if (firstBatch)
        {
            resources->setDeltaT(deltaT);
        } else
        {
            scalar tmp = resources->deltaTTry[INDEXVEC(0)];
            resources->setDeltaT(deltaT);
            resources->deltaTTry[INDEXVEC(0)] = tmp;
        }

        SeulexProfile profile;
        const long long tKernel = clock64();

        const scalar absTol_ = controls.absTol;
        const scalar relTol_ = controls.relTol;
        const label  maxSteps_ = controls.maxSteps;

        const scalar jacRedo_ = min(1e-4, relTol_);

        scalar theta_, logTol;
        label kTarg_;

        scalar* table_ = resources->table();
        scalar* dfdt_  = resources->dfdt();
        scalar* dfdy_  = resources->dfdy();
        scalar* hessTau_ = resources->hessTau();
        
        
        scalar* dtOpt_ = resources->dtOpt();
        scalar* temp_  = resources->temp();
        scalar* y0_    = resources->y0();
        scalar* ySequence_ = resources->ySequence();
        scalar* scale_ = resources->scale();
        
        scalar* y      = resources->vectors;

        scalar tStart   = 0;
        scalar tEnd     = deltaT;
        scalar t = tStart;

        scalar dt = deltaT;

        bool reachedEnd = false;

        for (label nStep=0; nStep<maxSteps_; ++nStep)
        {
            ++profile.nStep;

            // Store previous iteration dtTry
            scalar dtTry0 = resources->deltaTTry[INDEXVEC(0)];

            resources->reject[INDEXVEC(0)] = false;

            // Check if this is a truncated step and set dtTry to integrate to tEnd
            if ((t + resources->deltaTTry[INDEXVEC(0)] - tEnd)*(t + resources->deltaTTry[INDEXVEC(0)] - tStart) > 0)
            {
                resources->last[INDEXVEC(0)] = true;
                resources->deltaTTry[INDEXVEC(0)] = tEnd - t;
            }

            // Integrate as far as possible up to resources->deltaTTry[INDEXVEC(0)]
            {
                temp_[INDEXVEC(0)] = GREAT;
                dt = resources->deltaTTry[INDEXVEC(0)];
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
                    long long tProfile = clock64();
                    ode->jacobian(t, resources->parameters[INDEXVEC(0)], y, dfdt_, dfdy_);
                    profile.jacobian += clock64() - tProfile;
                    ++profile.nJacobian;

                    // The only O(n^3) step left. Every stage matrix built from
                    // this Jacobian, over this step and over as many following
                    // ones as it survives, is factorised from the result
                    tProfile = clock64();
                    hessenbergReduce(dfdy_, hessTau_, resources->systemSize());
                    profile.hessReduce += clock64() - tProfile;
                    ++profile.nHessReduce;

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
                        bool success = seul(resources, ode, t, dt, k, theta_, profile);

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
                        ++profile.nReject;

                        resources->prevReject[INDEXVEC(0)] = true;
                        if (!jacUpdated)
                        {
                            theta_ = 2*jacRedo_;

                            if (theta_ > jacRedo_ && !jacUpdated)
                            {
                                long long tProfile = clock64();
                                ode->jacobian(t, resources->parameters[INDEXVEC(0)], y, dfdt_, dfdy_);
                                profile.jacobian += clock64() - tProfile;
                                ++profile.nJacobian;

                                tProfile = clock64();
                                hessenbergReduce(dfdy_, hessTau_, resources->systemSize());
                                profile.hessReduce += clock64() - tProfile;
                                ++profile.nHessReduce;

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

            if ((t - tEnd)*(tEnd - tStart) >= 0)
            {
                if (nStep > 0 && resources->last[INDEXVEC(0)])
                {
                    resources->deltaTTry[INDEXVEC(0)] = dtTry0;
                }

                reachedEnd = true;

                break;
            }

            resources->first[INDEXVEC(0)] = false;

            if (resources->reject[INDEXVEC(0)])
            {
                resources->prevReject[INDEXVEC(0)] = true;
            }
        }

        if (!reachedEnd)
        {
            printf
            (
                "Integration steps greater than maximum %d : system %d, "
                "t = %0.16e, tEnd = %0.16e, deltaTDid = %0.16e \n",
                maxSteps_, INDEXVEC(0), t, tEnd, resources->deltaTDid[INDEXVEC(0)]
            );
        }

        for (label i=0; i < resources->systemSize(); ++i)
        {
            y[INDEXVEC(i)] = max(0.0, y[INDEXVEC(i)]);
        }

        resources->findMinDeltaT();

        profile.total = clock64() - tKernel;

        if (INDEXVEC(0) == profileSystem)
        {
            profile.print(INDEXVEC(0));
        }
    }
}

template<class ODESystem>
kodes::Seulex<ODESystem>::Seulex
(
    ODESystem* ode,
    SeulexDeviceResources* resources,
    label ensembleSize,
    const IntegratorControls& controls
)
:
    Integrator<ODESystem, SeulexDeviceResources>(ode, resources, ensembleSize, controls),
    profileSystem_(-1)
{}

template<class ODESystem>
void kodes::Seulex<ODESystem>::solve(scalar deltaT, label realBatchSize, bool firstBatch)
{
    seulex_solve<ODESystem><<<this->blocks, this->threads, this->sharedMemSize>>>
    (
        this->ode_, this->resources_, deltaT, realBatchSize, this->controls_, profileSystem_, firstBatch
    );
}

