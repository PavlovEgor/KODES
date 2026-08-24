
// Each thread owns one scratch slot and walks its share of the batch in a
// grid-stride loop, pulling one system into the slot, integrating it there and
// writing it back.
template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
__global__
void kodes::adaptive_solve
(
    ODESystem* ode,
    IntegratorDeviceResources* resources,
    kodes::IntegratorControls controls
)
{
    kodes::IntegratorControls ctrl = controls;

    const label systemSize = resources->systemSize();
    const label maxSteps_ = ctrl.maxSteps;

    scalar* __restrict__ y = resources->currentVector();

    for (label system = T_ID; system < controls.realBatchSize; system += GRID_DIM)
    {
        if (resources->vectorComponent(system, 0) <= ctrl.Treact)
        {
            continue;
        }

        ctrl.system = system;

        resources->loadSystem(system);
        resources->resetStep(system);

        scalar tStart = 0;
        scalar tEnd   = ctrl.deltaT;
        resources->currentT[system] = tStart;
        scalar& t = resources->currentT[system];

        bool reachedEnd = false;

        for (label nStep = 0; nStep < maxSteps_; ++nStep)
        {
            scalar dtTry0 = resources->deltaTTry[system];
            resources->reject[system] = false;

            if ((t + resources->deltaTTry[system] - tEnd)*(t + resources->deltaTTry[system] - tStart) > 0)
            {
                resources->last[system] = true;
                resources->deltaTTry[system] = tEnd - t;
            }

            if constexpr (IntegrationMethod::useAdaptiveStep)
            {
                Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::adaptiveStep(ode, resources, ctrl);
            }
            else
            {
                IntegrationMethod::step(ode, resources, ctrl);
            }

            if ((t - tEnd)*(tEnd - tStart) >= 0)
            {
                if (nStep > 0 && resources->last[system])
                {
                    resources->deltaTTry[system] = dtTry0;
                }
                reachedEnd = true;
                break;
            }

            resources->first[system] = false;

            if (resources->reject[system])
            {
                resources->prevReject[system] = true;
            }
        }

        if (!reachedEnd)
        {
            printf
            (
                "Integration steps greater than maximum %d : system %d, "
                "t = %0.16e, tEnd = %0.16e, deltaTDid = %0.16e \n",
                maxSteps_, system, t, tEnd, resources->deltaTDid[system]
            );
        }

        for (label i = 0; i < systemSize; ++i)
        {
            y[INDEXVEC(i)] = max(0.0, y[INDEXVEC(i)]);
        }

        resources->storeSystem(system);
    }
}

template<class IntegratorDeviceResources>
__global__
void kodes::setDeltaT(const scalar deltaT, IntegratorDeviceResources* resources)
{
    const label batchSize = resources->batchSize();

    for (label system = T_ID; system < batchSize; system += GRID_DIM)
    {
        resources->setDeltaT(deltaT, system);
    }
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
    const label system = controls.system;

    const scalar safeScale_ = controls.safeScale;
    const scalar alphaInc_ = controls.alphaIncrease;
    const scalar alphaDec_ = controls.alphaDecrease;
    const scalar minScale_ = controls.minScale;
    const scalar maxScale_ = controls.maxScale;

    scalar* __restrict__ yTemp_ = resources->yTemp();
    scalar* __restrict__ dydt0_ = resources->dydt0();

    scalar* __restrict__ y      = resources->currentVector();
    scalar& t      = resources->currentT[system];

    scalar dt = resources->deltaTTry[system];
    scalar err = 0.0;

    ode->derivatives(t, resources->currentParameter(0), y, dydt0_);

    // Loop over solver and adjust step-size as necessary
    // to achieve desired error
    do
    {
        // Solve step and provide error estimate
        err = IntegrationMethod::step(ode, resources, controls);

        // If error is large reduce dt and retry the step
        if (err > 1)
        {
            scalar scale = max(safeScale_*pow(err, -alphaDec_), minScale_);
            dt *= scale;
            resources->deltaTTry[system] = dt;

            if (dt < SMALL)
            {
                printf
                (
                    "system: %d stepsize underflow \n",
                    system
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
        resources->deltaTTry[system] = clamp(scale, minScale_, maxScale_)*dt;
    }
    else
    {
        resources->deltaTTry[system] = safeScale_*maxScale_*dt;
    }
}

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
__host__ label kodes::maxConcurrentSystems(const label threads)
{
    return maxConcurrentThreads
    (
        (const void*)adaptive_solve<ODESystem, IntegrationMethod, IntegratorDeviceResources>,
        threads,
        sharedMemorySize(threads)
    );
}

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
__host__ kodes::LaunchConfig kodes::planLaunch
(
    const label ensembleSize,
    const label systemSize,
    const label parameterSize,
    const size_t extraScratchBytesPerThread,
    const LaunchConfig& request
)
{
    if (ensembleSize <= 0 || systemSize <= 0 || request.threads <= 0)
    {
        fprintf(stderr, "kodes::planLaunch error at %s:%d: non-positive ensembleSize/systemSize/threads\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    return makePlan
    (
        request,
        ensembleSize,
        maxConcurrentSystems<ODESystem, IntegrationMethod, IntegratorDeviceResources>(request.threads),
        IntegratorDeviceResources::scratchBytesPerThread(systemSize, parameterSize)
      + extraScratchBytesPerThread,
        IntegratorDeviceResources::stateBytesPerSystem(systemSize, parameterSize),
        freeDeviceMemory()
    );
}

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
kodes::Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::Integrator
(
    ODESystem* ode,
    IntegratorDeviceResources* resources,
    const LaunchConfig& config,
    const IntegratorControls& controls
)
: config_(config), ode_(ode), resources_(resources), controls_(controls)
{
    if (!ode_ || !resources_)
    {
        fprintf(stderr, "Integrator ctor error at %s:%d: null ode/resources pointer\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    if (config_.threads <= 0 || config_.blocks <= 0)
    {
        fprintf(stderr, "Integrator ctor error at %s:%d: empty launch configuration\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    if (config_.scratchSize != config_.threads * config_.blocks)
    {
        fprintf(stderr, "Integrator ctor error at %s:%d: scratchSize != threads * blocks\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    if (config_.batchSize <= 0)
    {
        fprintf(stderr, "Integrator ctor error at %s:%d: batchSize <= 0\n", __FILE__, __LINE__);
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
    kodes::setDeltaT<IntegratorDeviceResources>
        <<<config_.blocks, config_.threads, config_.sharedMemSize>>>
        (deltaT, resources_);
    CUDA_CHECK_LAST();
}

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
void kodes::Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::solve(scalar deltaT, label realBatchSize)
{
    if (realBatchSize > config_.batchSize)
    {
        fprintf(stderr, "Integrator::solve error at %s:%d: realBatchSize > batchSize\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    controls_.realBatchSize = realBatchSize;
    controls_.deltaT = deltaT;

    kodes::adaptive_solve<ODESystem, IntegrationMethod, IntegratorDeviceResources>
        <<<config_.blocks, config_.threads, config_.sharedMemSize>>>
        (ode_, resources_, controls_);
    CUDA_CHECK_LAST();
}
