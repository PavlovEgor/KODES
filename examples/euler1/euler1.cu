#include "euler1.cuh"


int main(){

    label ensembleSize = 3 * 8192;

    // the same program as GRIMECH/seulex5, with one name changed
    const char* method = "euler";
    const char* balancer = "temperature";

    kodes::HostResources            host_res(ensembleSize, NSP, 1);

    set_same_initial_conditions(host_res.ensembleSize(), host_res.vectors, host_res.parameters);

    host_res.printVectori(0);

    kodes::LaunchConfig config = kodes::planLaunch
    (
        ensembleSize,
        host_res.systemSize(),
        host_res.parameterSize(),
        method,
        balancer,
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

    kodes::Handle<kodes::DeviceResources> resources = kodes::newResources
    (
        method, batchSize, config.scratchSize,
        host_res.systemSize(), host_res.parameterSize()
    );

    kodes::Handle<kodes::IntegrationMethod> integrationMethod = kodes::newMethod
    (
        method, batchSize, config.scratchSize,
        host_res.systemSize(), host_res.parameterSize()
    );

    kodes::Handle<kodes::Balancer> balancing = kodes::newBalancer
    (
        balancer, batchSize, config.scratchSize,
        host_res.systemSize(), host_res.parameterSize()
    );

    kodes::PyJacSystem* ode_prt = kodes::PyJacSystem::create(d_mem);

    kodes::Operator op(&host_res, resources.host());

    kodes::IntegratorControls controls(1e-10, 1e-1, 10000);

    kodes::Integrator solver
    (
        ode_prt, resources.device(), integrationMethod.device(), config, controls
    );

    solver.setBalancer(balancing.device(), balancing.host());

    scalar tEnd = 10.0;
    solver.setDeltaT(1e-10);

    for (label i=0; i < numOfBatches; i++)
    {
        op.cpyHostToDevice(i);
        solver.solve(tEnd, op.getRealBatchSize(i));
        op.cpyDeviceToHost(i);
    }

    host_res.printVectori(0);

    kodes::PyJacSystem::destroy(ode_prt);

    free_gpu_memory(&h_mem, &d_mem);
    free(h_mem);

    return 0;
}
