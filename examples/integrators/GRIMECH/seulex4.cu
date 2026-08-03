#include "seulex3.cuh"


int main(){

    const auto tic = []{ return std::chrono::steady_clock::now(); };

    const auto toc = [](const std::chrono::steady_clock::time_point& start)
    {
        return std::chrono::duration<double, std::milli>
        (
            std::chrono::steady_clock::now() - start
        ).count();
    };

    // The first CUDA call creates the context, time it on its own so that it is
    // not charged to the first allocation
    auto tStart = tic();
    cudaFree(0);
    const double msContext = toc(tStart);

    label ensembleSize = 2050;

    label batchSize = 1024;
    label numOfBatches = (ensembleSize + batchSize - 1) / batchSize;

    tStart = tic();
    kodes::HostResources            host_res(ensembleSize, NSP, 1);
    set_same_initial_conditions(host_res.ensembleSize(), host_res.vectors, host_res.parameters);
    const double msHostResources = toc(tStart);

    host_res.printVectori(0);

    mechanism_memory *h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    mechanism_memory *d_mem = nullptr;

    tStart = tic();
    initialize_gpu_memory(batchSize, &h_mem, &d_mem);
    const double msMechanismMemory = toc(tStart);

    tStart = tic();
    kodes::SeulexDeviceResources   host_res_dev(batchSize, host_res.systemSize(), host_res.parameterSize());

    kodes::SeulexDeviceResources*   res_prt = kodes::SeulexDeviceResources::create(batchSize, host_res.systemSize(), host_res.parameterSize(), &host_res_dev);
    const double msDeviceResources = toc(tStart);

    tStart = tic();
    kodes::pyJacSystem* ode_prt = kodes::pyJacSystem::createGPU(d_mem);
    const double msOdeSystem = toc(tStart);

    tStart = tic();
    kodes::Operator<kodes::HostResources, kodes::SeulexDeviceResources> op(&host_res, &host_res_dev);

    kodes::IntegratorControls controls(1e-12, 1e-4, 10000);

    kodes::Seulex<kodes::pyJacSystem> solver(ode_prt, res_prt, batchSize, controls);
    const double msSolver = toc(tStart);

    // Print the cycle breakdown of one system, negative to keep the kernel quiet
    solver.setProfileSystem(0);

    const double msSetup =
        msContext + msHostResources + msMechanismMemory
      + msDeviceResources + msOdeSystem + msSolver;

    scalar xEnd = 10.0;

    solver.resetDeltaTMin();

    // solve() only launches a kernel, so it is timed with cuda events on the
    // stream rather than on the host, otherwise the copy that follows would be
    // charged with the integration time
    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);

    float ms = 0;
    float msHostToDevice = 0, msSolve = 0, msDeviceToHost = 0;

    const auto wallStart = tic();

    for (label i=0; i < numOfBatches; i++)
    {
        cudaEventRecord(evStart);
        op.cpyHostToDevice(i);
        cudaEventRecord(evStop);
        cudaEventSynchronize(evStop);
        cudaEventElapsedTime(&ms, evStart, evStop);
        msHostToDevice += ms;

        cudaEventRecord(evStart);
        solver.solve(xEnd, op.getRealBatchSize(i));
        cudaEventRecord(evStop);
        cudaEventSynchronize(evStop);
        cudaEventElapsedTime(&ms, evStart, evStop);
        msSolve += ms;

        printf("batch %2d : %5d systems, solve %12.3f ms \n", i, op.getRealBatchSize(i), ms);

        cudaEventRecord(evStart);
        op.cpyDeviceToHost(i);
        cudaEventRecord(evStop);
        cudaEventSynchronize(evStop);
        cudaEventElapsedTime(&ms, evStart, evStop);
        msDeviceToHost += ms;
    }

    const double msWall = toc(wallStart);

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);

    scalar deltaTMin = solver.deltaTMin();

    host_res.printVectori(0);

    tStart = tic();
    kodes::pyJacSystem::destroyGPU(ode_prt);
    kodes::SeulexDeviceResources::destroy(res_prt, &host_res_dev);
    const double msTeardown = toc(tStart);

    printf("\n");
    printf("systems           : %12d \n", ensembleSize);
    printf("batches           : %12d of %d \n", numOfBatches, batchSize);
    printf("integrated to     : %12.6f s \n", xEnd);

    printf("\n--- setup ---------------------------- \n");
    printf("cuda context      : %12.3f ms \n", msContext);
    printf("host resources    : %12.3f ms \n", msHostResources);
    printf("mechanism memory  : %12.3f ms \n", msMechanismMemory);
    printf("device resources  : %12.3f ms \n", msDeviceResources);
    printf("ode system        : %12.3f ms \n", msOdeSystem);
    printf("operator, solver  : %12.3f ms \n", msSolver);
    printf("setup total       : %12.3f ms \n", msSetup);

    printf("\n--- integration ---------------------- \n");
    printf("host to device    : %12.3f ms \n", msHostToDevice);
    printf("solve             : %12.3f ms \n", msSolve);
    printf("device to host    : %12.3f ms \n", msDeviceToHost);
    printf("total wall clock  : %12.3f ms \n", msWall);
    printf("solve per system  : %12.3f us \n", 1e3*msSolve/ensembleSize);
    printf("throughput        : %12.1f systems/s \n", 1e3*ensembleSize/msSolve);

    printf("\n--- teardown ------------------------- \n");
    printf("destroy           : %12.3f ms \n", msTeardown);

    printf("\n");
    printf("min deltaTTry over %d systems : %0.16e \n", ensembleSize, deltaTMin);

    return 0;
}
