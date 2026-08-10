
template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
__global__
void kodes::adaptive_solve
(
    ODESystem* ode,
    IntegratorDeviceResources* resources,
    kodes::IntegratorControls controls
)
{
    if ((INDEXVEC(0) < controls.realBatchSize) && (resources->vectors[INDEXVEC(0)] > controls.Treact))
    {
        resources->resetStep();

        const label maxSteps_ = controls.maxSteps;

        scalar tStart = 0;
        scalar tEnd   = controls.deltaT;
        resources->currentT[INDEXVEC(0)] = tStart;
        scalar& t = resources->currentT[INDEXVEC(0)];

        bool reachedEnd = false;

        for (label nStep = 0; nStep < maxSteps_; ++nStep)
        {
            scalar dtTry0 = resources->deltaTTry[INDEXVEC(0)];
            resources->reject[INDEXVEC(0)] = false;

            if ((t + resources->deltaTTry[INDEXVEC(0)] - tEnd)*(t + resources->deltaTTry[INDEXVEC(0)] - tStart) > 0)
            {
                resources->last[INDEXVEC(0)] = true;
                resources->deltaTTry[INDEXVEC(0)] = tEnd - t;
            }

            if constexpr (IntegrationMethod::useAdaptiveStep)
            {
                Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::adaptiveStep(ode, resources, controls);
            }
            else
            {
                IntegrationMethod::step(ode, resources, controls);
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

        scalar* y = resources->vectors;
        for (label i = 0; i < resources->systemSize(); ++i)
        {
            y[INDEXVEC(i)] = max(0.0, y[INDEXVEC(i)]);
        }
    }
}

template<class IntegratorDeviceResources>
__global__
void kodes::setDeltaT(const scalar deltaT, IntegratorDeviceResources* resources)
{
    resources->setDeltaT(deltaT);
}

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
__device__
void kodes::Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::adaptiveStep
(
    ODESystem* ode,
    IntegratorDeviceResources* resources,
    kodes::IntegratorControls controls
)
{
    const label systemSize = resources->systemSize();

    const scalar safeScale_ = controls.safeScale;
    const scalar alphaInc_ = controls.alphaIncrease;
    const scalar alphaDec_ = controls.alphaDecrease;
    const scalar minScale_ = controls.minScale;
    const scalar maxScale_ = controls.maxScale;

    scalar* __restrict__ yTemp_ = resources->yTemp();
    scalar* __restrict__ dydx0_ = resources->dydx0();

    scalar* __restrict__ y      = resources->vectors;
    scalar& t      = resources->currentT[INDEXVEC(0)];

    scalar dt = resources->deltaTTry[INDEXVEC(0)];
    scalar err = 0.0;

    ode->derivatives(t, resources->parameters[INDEXVEC(0)], y, dydx0_);

    // Loop over solver and adjust step-size as necessary
    // to achieve desired error
    do
    {
        // Solve step and provide error estimate
        err = IntegrationMethod::step(ode, resources, controls);

        // If error is large reduce dx
        if (err > 1)
        {
            scalar scale = max(safeScale_*pow(err, -alphaDec_), minScale_);
            dt *= scale;

            if (dt < SMALL)
            {
                printf
                (
                    "thread: %d stepsize underflow \n",
                    INDEXVEC(0)
                );
            }
        }
    } while (err > 1);

    // Update the state
    t += dt;
    copyVec(y, yTemp_, systemSize);

    // If the error is small increase the step-size
    if (err > pow(maxScale_/safeScale_, -1.0/alphaInc_))
    {
        scalar scale = safeScale_*pow(err, -alphaInc_);
        resources->deltaTTry[INDEXVEC(0)] = clamp(scale, minScale_, maxScale_)*dt;
    }
    else
    {
        resources->deltaTTry[INDEXVEC(0)] = safeScale_*maxScale_*dt;
    }
}

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
kodes::Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::Integrator
(
    ODESystem* ode,
    IntegratorDeviceResources* resources,
    label batchSize,
    const IntegratorControls& controls
)
: ode_(ode), resources_(resources), controls_(controls)
{
    if (!ode_ || !resources_)
    {
        fprintf(stderr, "Integrator ctor error at %s:%d: null ode/resources pointer\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    if (batchSize <= 0)
    {
        fprintf(stderr, "Integrator ctor error at %s:%d: batchSize <= 0\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    threads = kodes::blockSize(batchSize);
    blocks = kodes::numOfBlocks(batchSize);
    sharedMemSize = kodes::sharedMemorySize(batchSize);

    if (batchSize != threads * blocks)
    {
        fprintf(stderr, "Integrator ctor error at %s:%d: batchSize != threads * blocks\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }
}


template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
kodes::Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::~Integrator()
{
}

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
void kodes::Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::setDeltaT(const scalar deltaT)
{
    kodes::setDeltaT<IntegratorDeviceResources><<<blocks, threads, sharedMemSize>>>(deltaT, resources_);
    CUDA_CHECK_LAST();
}

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
void kodes::Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::solve(scalar deltaT, label realBatchSize)
{
    controls_.realBatchSize = realBatchSize;
    controls_.deltaT = deltaT;

    kodes::adaptive_solve<ODESystem, IntegrationMethod, IntegratorDeviceResources>
        <<<blocks, threads, sharedMemSize>>>
        (ode_, resources_, controls_);
    CUDA_CHECK_LAST();
}
