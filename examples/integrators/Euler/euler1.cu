#include "euler1.cuh"


int main(){

    label ensembleSize = 3 * 8192;

    kodes::HostResources            host_res(ensembleSize, NSP, 1);

    set_same_initial_conditions(host_res.ensembleSize(), host_res.vectors, host_res.parameters);

    host_res.printVectori(0);

    kodes::LaunchConfig config = kodes::planLaunch
    <
        kodes::pyJacSystem,
        kodes::Euler<kodes::pyJacSystem>,
        kodes::EulerDeviceResources
    >
    (
        ensembleSize,
        host_res.systemSize(),
        host_res.parameterSize(),
        required_mechanism_size(),
        kodes::LaunchConfig("best")
    );

    config.print("euler1");

    label batchSize = config.batchSize;
    label numOfBatches = config.numOfBatches(ensembleSize);

    mechanism_memory *h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    mechanism_memory *d_mem = nullptr;

    // pyJac's scratch is per thread, so it is padded to the resident threads
    initialize_gpu_memory(config.scratchSize, &h_mem, &d_mem);

    kodes::EulerDeviceResources   host_res_dev(batchSize, config.scratchSize, host_res.systemSize(), host_res.parameterSize());

    kodes::EulerDeviceResources*   res_prt = kodes::EulerDeviceResources::create(batchSize, config.scratchSize, host_res.systemSize(), host_res.parameterSize(), &host_res_dev);

    kodes::pyJacSystem* ode_prt = kodes::pyJacSystem::createGPU(d_mem);

    kodes::Operator<kodes::HostResources, kodes::EulerDeviceResources> op(&host_res, &host_res_dev);

    kodes::IntegratorControls controls(1e-10, 1e-1, 10000);

    kodes::Integrator<kodes::pyJacSystem, kodes::Euler<kodes::pyJacSystem>, kodes::EulerDeviceResources> solver(ode_prt, res_prt, config, controls);

    scalar tEnd = 10.0;
    solver.setDeltaT(1e-10);

    for (label i=0; i < numOfBatches; i++)
    {
        op.cpyHostToDevice(i);
        solver.solve(tEnd, op.getRealBatchSize(i));
        op.cpyDeviceToHost(i);
    }

    host_res.printVectori(0);

    kodes::pyJacSystem::destroyGPU(ode_prt);
    kodes::EulerDeviceResources::destroy(res_prt, &host_res_dev);

    free_gpu_memory(&h_mem, &d_mem);
    free(h_mem);

    return 0;
}
