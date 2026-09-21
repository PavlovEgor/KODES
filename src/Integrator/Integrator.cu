#include "Integrator.cuh"
#include "method_table.cuh"
#include "balancer_table.cuh"
#include "block_reduce.cuh"

// A block owns blockDim.x scratch slots and walks its share of the batch a
// chunk at a time, pulling blockDim.x systems into those slots, integrating
// them together and writing them back. It walks the balanced order, so a chunk
// is blockDim.x *neighbours* of that order - which is what makes it reasonable
// for them to share one Jacobian and one step size.
//
// The loop below therefore has one rule behind every line of it: from the top
// of a chunk to the bottom, every branch is taken by the whole block. A
// __syncthreads() that some threads reach and others do not is undefined
// behaviour, not a slow path, and the method this calls into is made almost
// entirely of them. Hence `chunks` computed the same in every thread rather
// than a condition on the loop variable, and hence `active` as a mask rather
// than the two `continue`s that used to stand here.
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

    // Carried from the previous launch: the step this block ended the last
    // batch with is the one it starts this one from.
    kodes::BlockStep blockStep;
    scalar deltaTTry = resources->deltaTTry[B_ID];

    const label chunks = (controls.realBatchSize + GRID_DIM - 1) / GRID_DIM;

    for (label c = 0; c < chunks; ++c)
    {
        const label position = T_ID + c * GRID_DIM;
        const bool inBatch = position < controls.realBatchSize;

        const label system = inBatch ? resources->systemAt(position) : 0;

        // A cold cell is not integrated, but it does not leave the block
        // either: it goes on reaching every barrier its neighbours do, and
        // contributes nothing but a losing value to their reductions.
        const bool active =
            inBatch && resources->vectorComponent(system, 0) > ctrl.Treact;

        if (!blockAny(active))
        {
            continue;
        }

        ctrl.position = c * GRID_DIM + blockIdx.x * blockDim.x;

        if (active)
        {
            resources->loadSystem(system);
        }

        blockStep.reset(deltaTTry);

        const scalar tStart = 0;
        const scalar tEnd = ctrl.deltaT;

        bool reachedEnd = false;

        for (label nStep = 0; nStep < maxSteps; ++nStep)
        {
            const scalar dtTry0 = blockStep.deltaTTry;
            blockStep.reject = false;

            if ((blockStep.t + blockStep.deltaTTry - tEnd)*(blockStep.t + blockStep.deltaTTry - tStart) > 0)
            {
                blockStep.last = true;
                blockStep.deltaTTry = tEnd - blockStep.t;
            }

            method->advance(ode, resources, blockStep, active, ctrl);

            if ((blockStep.t - tEnd)*(tEnd - tStart) >= 0)
            {
                if (nStep > 0 && blockStep.last)
                {
                    blockStep.deltaTTry = dtTry0;
                }
                reachedEnd = true;
                break;
            }

            blockStep.first = false;

            if (blockStep.reject)
            {
                blockStep.prevReject = true;
            }
        }

        if (!reachedEnd && threadIdx.x == 0)
        {
            printf
            (
                "Integration steps greater than maximum %d : positions %d..%d, "
                "t = %0.16e, tEnd = %0.16e, deltaTDid = %0.16e \n",
                maxSteps, ctrl.position, ctrl.position + blockDim.x - 1,
                blockStep.t, tEnd, blockStep.deltaTDid
            );
        }

        if (active)
        {
            for (label i = 0; i < systemSize; ++i)
            {
                y[INDEXVEC(i)] = max(0.0, y[INDEXVEC(i)]);
            }

            resources->storeSystem(system);
        }

        deltaTTry = blockStep.deltaTTry;
    }

    // Every thread read this slot on the way in, and a batch of nothing but
    // cold cells would put no barrier between that read and the write below.
    __syncthreads();

    if (threadIdx.x == 0)
    {
        resources->deltaTTry[B_ID] = deltaTTry;
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
    const char* balancerName,
    const size_t extraScratchBytesPerThread,
    const LaunchConfig& request,
    const size_t deviceStackBytes
)
{
    if (ensembleSize <= 0 || systemSize <= 0 || request.threads <= 0)
    {
        fprintf(stderr, "kodes::planLaunch error at %s:%d: non-positive ensembleSize/systemSize/threads\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    if (deviceStackBytes == 0)
    {
        fprintf(stderr, "kodes::planLaunch error at %s:%d: deviceStackBytes is zero\n", __FILE__, __LINE__);
        std::exit(EXIT_FAILURE);
    }

    // First, and before freeDeviceMemory() below: the driver reserves the stack
    // of every resident thread out of the same VRAM the plan is about to divide
    // up. Sizing the stack after the plan would mean promising memory twice.
    setDeviceStack(deviceStackBytes);

    const size_t scratchPerThread =
        methodScratchBytesPerThread(methodName, systemSize, parameterSize)
      + balancerScratchBytesPerThread(balancerName, systemSize, parameterSize)
      + extraScratchBytesPerThread;

    const size_t scratchPerBlock =
        methodScratchBytesPerBlock(methodName, systemSize, parameterSize)
      + balancerScratchBytesPerBlock(balancerName, systemSize, parameterSize);

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
      + balancerStateBytesPerSystem(balancerName, systemSize, parameterSize),
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

    // Where the block scratch lives is a decision of the plan, and the kernel
    // is the only place it is acted on, so it travels with the rest of what
    // the kernel is told.
    controls_.blockScratchInShared = config_.blockScratchInShared;
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
    // One trial step per block, so one thread per block to seed it
    kodes::setDeltaTKernel<<<config_.blocks, 1>>>(deltaT, resources_);
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

    kodes::solveKernel
        <<<config_.blocks, config_.threads, config_.sharedMemSize>>>
        (ode_, resources_, method_, controls_);
    CUDA_CHECK_LAST();
}
