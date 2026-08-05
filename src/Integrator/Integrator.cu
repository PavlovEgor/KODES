

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
__global__
void kodes::adaptive_solve
(
    ODESystem* ode,
    IntegratorDeviceResources* resources,
    scalar deltaT,
    label realBatchSize,
    kodes::IntegratorControls controls
)
{
    if ((INDEXVEC(0) < realBatchSize) && (resources->vectors[INDEXVEC(0)] > 0))
    {
        resources->setDeltaT(deltaT);

        const label maxSteps_ = controls.maxSteps;

        scalar tStart = 0;
        scalar tEnd   = deltaT;
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

            IntegrationMethod::step(ode, resources, controls);

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

    CUDA_CHECK(cudaMalloc(&deltaTMinDevice_, sizeof(scalar)));
}


template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
kodes::Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::~Integrator()
{
    CUDA_CHECK(cudaFree(deltaTMinDevice_));
}

template<class ODESystem, class IntegrationMethod, class IntegratorDeviceResources>
void kodes::Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>::solve(scalar deltaT, label realBatchSize)
{
    adaptive_solve<ODESystem, IntegrationMethod, IntegratorDeviceResources>
        <<<blocks, threads, sharedMemSize>>>
        (ode_, resources_, deltaT, realBatchSize, controls_);
    CUDA_CHECK_LAST();
}