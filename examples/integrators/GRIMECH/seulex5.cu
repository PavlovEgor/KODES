#include "seulex3.cuh"


// GRI-Mech 3.0, an ensemble of identical reactors, integrated on the GPU.
//
// Nothing about the run is compiled in: the method, the balancer, the
// tolerances and the sizes all come out of a JSON file, which is looked up in
// the tables when the program starts.
//
//     ./seulex5 [settings.json]      default: seulex5.json
int main(int argc, char** argv)
{
    kodes::Settings settings(argc > 1 ? argv[1] : "seulex5.json");

    settings.print();

    const char* method = settings.method().c_str();
    const char* balancer = settings.balancer().c_str();

    label ensembleSize = settings.ensembleSize();

    kodes::HostResources            host_res(ensembleSize, NSP, 1);

    set_same_initial_conditions(host_res.ensembleSize(), host_res.vectors, host_res.parameters);

    host_res.printVectori(0);

    // The two names are what tell the plan how much device memory the run will
    // need: the method's entry knows what its resources cost, the balancer's
    // knows what the keys and the order cost. Everything owned outside both -
    // here pyJac's own per thread scratch - goes in the extra.
    kodes::LaunchConfig config = kodes::planLaunch
    (
        ensembleSize,
        host_res.systemSize(),
        host_res.parameterSize(),
        method,
        balancer,
        required_mechanism_size(),
        settings.launchRequest()
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

    // an empty handle when the settings say "none", which leaves the batch in
    // the order it was copied in
    kodes::Handle<kodes::Balancer> balancing = kodes::newBalancer
    (
        balancer, batchSize, config.scratchSize,
        host_res.systemSize(), host_res.parameterSize()
    );

    kodes::pyJacSystem* ode_prt = kodes::pyJacSystem::createGPU(d_mem);

    kodes::Operator op(&host_res, resources.host());

    kodes::Integrator solver
    (
        ode_prt, resources.device(), integrationMethod.device(),
        config, settings.controls()
    );

    solver.setBalancer(balancing.device(), balancing.host());

    scalar tEnd = settings.endTime();
    solver.setDeltaT(settings.initialTimeStep());

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
