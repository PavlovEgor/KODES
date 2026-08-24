#include "seulex3.cuh"


int main(){

    label ensembleSize = 3 * 8192;

    kodes::HostResources            host_res(ensembleSize, NSP, 1);

    set_same_initial_conditions(host_res.ensembleSize(), host_res.vectors, host_res.parameters);

    host_res.printVectori(0);

    // kodes::LaunchConfig("best")  - take the whole device
    // kodes::LaunchConfig("half")  - take one half of it
    // kodes::LaunchConfig(8192, 1000000) - concurrent systems and batch by hand
    kodes::LaunchConfig config = kodes::planLaunch
    <
        kodes::pyJacSystem,
        kodes::Seulex<kodes::pyJacSystem>,
        kodes::SeulexDeviceResources
    >
    (
        ensembleSize,
        host_res.systemSize(),
        host_res.parameterSize(),
        required_mechanism_size() + kodes::StiffnessBalancer::scratchBytesPerThread(NSP),
        kodes::StiffnessBalancer::bytesPerSystem(),
        kodes::LaunchConfig("best")
    );

    config.print("seulex5");

    label batchSize = config.batchSize;
    label numOfBatches = config.numOfBatches(ensembleSize);

    mechanism_memory *h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    mechanism_memory *d_mem = nullptr;

    // pyJac's scratch is per thread, so it is padded to the resident threads
    initialize_gpu_memory(config.scratchSize, &h_mem, &d_mem);

    kodes::SeulexDeviceResources   host_res_dev(batchSize, config.scratchSize, host_res.systemSize(), host_res.parameterSize());

    kodes::SeulexDeviceResources*   res_prt = kodes::SeulexDeviceResources::create(batchSize, config.scratchSize, host_res.systemSize(), host_res.parameterSize(), &host_res_dev);

    kodes::pyJacSystem* ode_prt = kodes::pyJacSystem::createGPU(d_mem);

    // temperature first, then the norm of the right hand side inside each band
    // of it - kodes::TemperatureBalancer is the cheaper single key version
    kodes::StiffnessBalancer   balancer_dev(batchSize, config.scratchSize, host_res.systemSize());

    kodes::StiffnessBalancer*  balancer_prt = kodes::StiffnessBalancer::create(batchSize, config.scratchSize, host_res.systemSize(), &balancer_dev);

    kodes::Operator<kodes::HostResources, kodes::SeulexDeviceResources> op(&host_res, &host_res_dev);

    kodes::IntegratorControls controls(1e-10, 1e-1, 10000);

    kodes::Integrator<kodes::pyJacSystem, kodes::Seulex<kodes::pyJacSystem>, kodes::SeulexDeviceResources> solver(ode_prt, res_prt, config, controls);

    solver.setBalancer(balancer_prt, &balancer_dev);

    scalar tEnd = 10.0;
    solver.setDeltaT(tEnd);

    for (label i=0; i < numOfBatches; i++)
    {
        op.cpyHostToDevice(i);
        solver.solve(tEnd, op.getRealBatchSize(i));
        op.cpyDeviceToHost(i);
    }

    host_res.printVectori(0);

    kodes::StiffnessBalancer::destroy(balancer_prt, &balancer_dev);
    kodes::pyJacSystem::destroyGPU(ode_prt);
    kodes::SeulexDeviceResources::destroy(res_prt, &host_res_dev);

    free_gpu_memory(&h_mem, &d_mem);
    free(h_mem);

    return 0;
}
