#include "Integrator.cuh"
#include "method_table.cuh"
#include "system_table.cuh"

// Each thread owns one scratch slot and walks its share of the batch in a
// grid-stride loop, pulling one system into the slot, integrating it there on
// its own and writing it back.
__global__
void kodes::solveKernel
(
    kodes::ODESystem* ode,
    kodes::DeviceResources* resources,
    const kodes::IntegrationMethod* method,
    kodes::IntegratorControls controls
)
{
    kodes::IntegratorControls ctrl = controls;

    const label systemSize = resources->systemSize();
    const label maxSteps = ctrl.maxSteps;

    scalar* __restrict__ y = resources->currentVector();

    for (label system = T_ID; system < controls.realBatchSize; system += GRID_DIM)
    {
        if (resources->vectorComponent(system, 0) <= ctrl.Treact)
        {
            continue;
        }

        ctrl.system = system;

        resources->loadSystem(system);
        resources->resetStep();

        const scalar tStart = 0;
        const scalar tEnd = ctrl.deltaT;
        resources->currentT[INDEXVEC(0)] = tStart;
        scalar& t = resources->currentT[INDEXVEC(0)];

        bool reachedEnd = false;

        for (label nStep = 0; nStep < maxSteps; ++nStep)
        {
            const scalar dtTry0 = resources->deltaTTry[INDEXVEC(0)];
            resources->reject[INDEXVEC(0)] = false;

            if ((t + resources->deltaTTry[INDEXVEC(0)] - tEnd)*(t + resources->deltaTTry[INDEXVEC(0)] - tStart) > 0)
            {
                resources->last[INDEXVEC(0)] = true;
                resources->deltaTTry[INDEXVEC(0)] = tEnd - t;
            }

            method->advance(ode, resources, ctrl);

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
                maxSteps, system, t, tEnd, resources->deltaTDid[INDEXVEC(0)]
            );
        }

        for (label i = 0; i < systemSize; ++i)
        {
            y[INDEXVEC(i)] = max(0.0, y[INDEXVEC(i)]);
        }

        resources->storeSystem(system);
    }
}

__global__
void kodes::setDeltaTKernel(const scalar deltaT, kodes::DeviceResources* resources)
{
    resources->setDeltaT(deltaT);
}

__host__ label kodes::maxConcurrentSystems(const label threads, const size_t sharedMemSize)
{
    return maxConcurrentThreads((const void*)solveKernel, threads, sharedMemSize);
}

__host__ kodes::LaunchConfig kodes::planLaunch
(
    const label ensembleSize,
    const label systemSize,
    const label parameterSize,
    const char* methodName,
    const char* systemName,
    const LaunchConfig& request,
    const size_t extraScratchBytesPerThread,
    const size_t deviceStackBytes
)
{
    if (ensembleSize <= 0 || systemSize <= 0 || request.threads <= 0)
    {
        fprintf(stderr, "kodes::planLaunch error at %s:%d: non-positive ensembleSize/systemSize/threads\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    // Zero leaves the driver's own value alone, which is what it should be
    // doing: see KODES_DEVICE_STACK_BYTES. When a caller does override it, this
    // has to happen before freeDeviceMemory() below - the driver reserves the
    // stack of every resident thread out of the same VRAM the plan is about to
    // divide up, so sizing it after the plan would promise that memory twice.
    if (deviceStackBytes > 0)
    {
        setDeviceStack(deviceStackBytes);
    }

    const size_t scratchPerThread =
        methodScratchBytesPerThread(methodName, systemSize, parameterSize)
      + systemScratchBytesPerThread(systemName, systemSize, parameterSize)
      + extraScratchBytesPerThread;

    const size_t scratchPerBlock =
        methodScratchBytesPerBlock(methodName, systemSize, parameterSize)
      + systemScratchBytesPerBlock(systemName, systemSize, parameterSize);

    const size_t maxSharedPerBlock = maxSharedMemoryPerBlock();

    // The occupancy of the solve kernel depends on how much shared memory it
    // is launched with, and that depends on whether the block scratch is going
    // there - so the decision is taken here, before the question is asked, and
    // makePlan is handed the same inputs and reaches the same answer.
    const size_t sharedMemSize = sharedMemorySize
    (
        request.threads,
        blockScratchFitsShared(request.threads, scratchPerBlock, maxSharedPerBlock)
      ? scratchPerBlock
      : 0
    );

    return makePlan
    (
        request,
        ensembleSize,
        maxConcurrentSystems(request.threads, sharedMemSize),
        scratchPerThread,
        scratchPerBlock,
        methodStateBytesPerSystem(methodName, systemSize, parameterSize)
      + systemStateBytesPerSystem(systemName, systemSize, parameterSize),
        freeDeviceMemory(),
        maxSharedPerBlock
    );
}

kodes::Integrator::Integrator
(
    kodes::ODESystem* ode,
    kodes::DeviceResources* resources,
    const kodes::IntegrationMethod* method,
    const kodes::LaunchConfig& config,
    const kodes::IntegratorControls& controls
)
: config_(config), ode_(ode), resources_(resources), method_(method),
  controls_(controls)
{
    if (!ode_ || !resources_ || !method_)
    {
        fprintf(stderr, "Integrator ctor error at %s:%d: null ode/resources/method pointer\n", __FILE__, __LINE__);
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


kodes::Integrator::~Integrator()
{
}

void kodes::Integrator::setDeltaT(const scalar deltaT)
{
    // One trial step per scratch slot, seeded by the thread that owns it
    kodes::setDeltaTKernel
        <<<config_.blocks, config_.threads, config_.sharedMemSize>>>
        (deltaT, resources_);
    CUDA_CHECK_LAST();
}

void kodes::Integrator::solve(scalar deltaT, label realBatchSize)
{
    if (realBatchSize > config_.batchSize)
    {
        fprintf(stderr, "Integrator::solve error at %s:%d: realBatchSize > batchSize\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    controls_.realBatchSize = realBatchSize;
    controls_.deltaT = deltaT;

    kodes::solveKernel
        <<<config_.blocks, config_.threads, config_.sharedMemSize>>>
        (ode_, resources_, method_, controls_);
    CUDA_CHECK_LAST();
}
