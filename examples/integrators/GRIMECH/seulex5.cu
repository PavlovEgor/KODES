#include "seulex3.cuh"


int main(){

    label ensembleSize = 3 * 8192;

    // Both are names looked up in a table when the program runs, not types
    // baked in when it was compiled:
    //   method   - "seulex" or "euler",           see methodTable.cu
    //   balancer - "temperature", "rhsNorm",
    //              "stiffness" or "none",         see balancerTable.cu
    const char* method = "seulex";
    const char* balancer = "stiffness";

    kodes::HostResources            host_res(ensembleSize, NSP, 1);

    set_same_initial_conditions(host_res.ensembleSize(), host_res.vectors, host_res.parameters);

    host_res.printVectori(0);

    // The two names are what tell the plan how much device memory the run will
    // need: the method's entry knows what its resources cost, the balancer's
    // knows what the keys and the order cost. Everything owned outside both -
    // here pyJac's own per thread scratch - goes in the extra.
    //
    // kodes::LaunchConfig("best")  - take the whole device
    // kodes::LaunchConfig("half")  - take one half of it
    // kodes::LaunchConfig(8192, 1000000) - concurrent systems and batch by hand
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

    config.print("seulex5");

    label batchSize = config.batchSize;
    label numOfBatches = config.numOfBatches(ensembleSize);

    mechanism_memory *h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    mechanism_memory *d_mem = nullptr;

    // pyJac's scratch is per thread, so it is padded to the resident threads
    initialize_gpu_memory(config.scratchSize, &h_mem, &d_mem);

    // Each handle owns a device object and the host stub holding its buffers,
    // and hands both back when it goes out of scope.
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

    // temperature first, then the norm of the right hand side inside each band
    // of it - "temperature" is the cheaper single key version
    kodes::Handle<kodes::Balancer> balancing = kodes::newBalancer
    (
        balancer, batchSize, config.scratchSize,
        host_res.systemSize(), host_res.parameterSize()
    );

    kodes::pyJacSystem* ode_prt = kodes::pyJacSystem::createGPU(d_mem);

    kodes::Operator op(&host_res, resources.host());

    kodes::IntegratorControls controls(1e-10, 1e-1, 10000);

    kodes::Integrator solver
    (
        ode_prt, resources.device(), integrationMethod.device(), config, controls
    );

    solver.setBalancer(balancing.device(), balancing.host());

    scalar tEnd = 10.0;
    solver.setDeltaT(tEnd);

    for (label i=0; i < numOfBatches; i++)
    {
        op.cpyHostToDevice(i);
        solver.solve(tEnd, op.getRealBatchSize(i));
        op.cpyDeviceToHost(i);
    }

    host_res.printVectori(0);

    kodes::pyJacSystem::destroyGPU(ode_prt);

    free_gpu_memory(&h_mem, &d_mem);
    free(h_mem);

    return 0;
}
