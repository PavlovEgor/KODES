#include "Integrator.cuh"
#include "methodTable.cuh"
#include "balancerTable.cuh"

// Each thread owns one scratch slot and walks its share of the batch in a
// grid-stride loop, pulling one system into the slot, integrating it there and
// writing it back. It walks the balanced order, so the systems a warp picks up
// are 32 neighbours of that order.
__global__
void kodes::adaptive_solve
(
    kodes::ODESystem* ode,
    kodes::DeviceResources* resources,
    const kodes::IntegrationMethod* method,
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

__global__
void kodes::setDeltaTKernel(const scalar deltaT, kodes::DeviceResources* resources)
{
    resources->setDeltaT(deltaT);
}

__global__
void kodes::useOrderKernel(kodes::DeviceResources* resources, const label* order)
{
    resources->useOrder(order);
}

__host__ label kodes::maxConcurrentSystems(const label threads)
{
    return maxConcurrentThreads
    (
        (const void*)adaptive_solve,
        threads,
        sharedMemorySize(threads)
    );
}

__host__ kodes::LaunchConfig kodes::planLaunch
(
    const label ensembleSize,
    const label systemSize,
    const label parameterSize,
    const char* methodName,
    const char* balancerName,
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
        maxConcurrentSystems(request.threads),
        methodScratchBytesPerThread(methodName, systemSize, parameterSize)
      + balancerScratchBytesPerThread(balancerName, systemSize, parameterSize)
      + extraScratchBytesPerThread,
        methodStateBytesPerSystem(methodName, systemSize, parameterSize)
      + balancerStateBytesPerSystem(balancerName, systemSize, parameterSize),
        freeDeviceMemory()
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
  balancer_(nullptr), balancerStub_(nullptr), controls_(controls)
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

void kodes::Integrator::setBalancer(kodes::Balancer* balancer, kodes::Balancer* hostStub)
{
    if (bool(balancer) != bool(hostStub))
    {
        fprintf(stderr, "Integrator::setBalancer error at %s:%d: need both the device balancer and its host stub\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    balancer_ = balancer;
    balancerStub_ = hostStub;

    kodes::useOrderKernel<<<1, 1>>>
    (
        resources_, hostStub ? hostStub->order() : nullptr
    );
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void kodes::Integrator::setDeltaT(const scalar deltaT)
{
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

    if (balancerStub_)
    {
        balancerStub_->balance(balancer_, resources_, ode_, realBatchSize, config_);
    }

    controls_.realBatchSize = realBatchSize;
    controls_.deltaT = deltaT;

    kodes::adaptive_solve
        <<<config_.blocks, config_.threads, config_.sharedMemSize>>>
        (ode_, resources_, method_, controls_);
    CUDA_CHECK_LAST();
}
