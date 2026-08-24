
// Each thread owns one scratch slot and walks its share of the batch in a
// grid-stride loop, pulling one system into the slot, integrating it there and
// writing it back. It walks the balanced order, so the systems a warp picks up
// are 32 neighbours of that order.
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

    for (label i = T_ID; i < controls.realBatchSize; i += GRID_DIM)
    {
        const label system = resources->systemAt(i);

        if (resources->vectorComponent(system, 0) <= ctrl.Treact)
        {
            continue;
        }

        ctrl.system = system;

        resources->loadSystem(system);
        resources->resetStep();

        scalar tStart = 0;
        scalar tEnd   = ctrl.deltaT;
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
                Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::adaptiveStep(ode, resources, ctrl);
            }
            else
            {
                IntegrationMethod::step(ode, resources, ctrl);
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
                maxSteps_, system, t, tEnd, resources->deltaTDid[INDEXVEC(0)]
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
    resources->setDeltaT(deltaT);
}

template<class IntegratorDeviceResources>
__global__
void kodes::useOrder(IntegratorDeviceResources* resources, const label* order)
{
    resources->useOrder(order);
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
    scalar* __restrict__ dydt0_ = resources->dydt0();

    scalar* __restrict__ y      = resources->currentVector();
    scalar& t      = resources->currentT[INDEXVEC(0)];

    scalar dt = resources->deltaTTry[INDEXVEC(0)];
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
            resources->deltaTTry[INDEXVEC(0)] = dt;

            if (dt < SMALL)
            {
                printf
                (
                    "system: %d stepsize underflow \n",
                    controls.system
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
    const size_t extraStateBytesPerSystem,
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
        IntegratorDeviceResources::stateBytesPerSystem(systemSize, parameterSize)
      + extraStateBytesPerSystem,
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
: config_(config), ode_(ode), resources_(resources),
  balancer_(nullptr), balancerStub_(nullptr), controls_(controls)
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
void kodes::Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::setBalancer
(
    Balancer* balancer,
    Balancer* hostStub
)
{
    if (bool(balancer) != bool(hostStub))
    {
        fprintf(stderr, "Integrator::setBalancer error at %s:%d: need both the device balancer and its host stub\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    balancer_ = balancer;
    balancerStub_ = hostStub;

    kodes::useOrder<IntegratorDeviceResources><<<1, 1>>>
    (
        resources_, hostStub ? hostStub->order() : nullptr
    );
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
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

    if (balancerStub_)
    {
        balancerStub_->balance(balancer_, resources_, realBatchSize, config_);
    }

    controls_.realBatchSize = realBatchSize;
    controls_.deltaT = deltaT;

    kodes::adaptive_solve<ODESystem, IntegrationMethod, IntegratorDeviceResources>
        <<<config_.blocks, config_.threads, config_.sharedMemSize>>>
        (ode_, resources_, controls_);
    CUDA_CHECK_LAST();
}
