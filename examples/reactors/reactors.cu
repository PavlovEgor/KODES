#include "reactors.cuh"

#include <chrono>

// An ensemble of identical constant-pressure reactors, integrated on the GPU.
//
// One program, two cases. Which mechanism it holds is fixed when it is built,
// because pyJac generates C for one mechanism and NSP is a macro of its
// mechanism.cuh; everything else - the integration method, the balancer, the
// tolerances, how many reactors and how far to integrate them - is a name or a
// number in the settings file, read when the program starts.
//
//     ./reactors_grimech [settings.json]     default: grimech.json
//     ./reactors_h2o2    [settings.json]     default: h2o2.json
//
// So switching from seulex to euler, or from grouping the batch by temperature
// to grouping it by stiffness, is an edit to the .json - not a rebuild.
int main(int argc, char** argv)
{
    kodes::Settings settings(argc > 1 ? argv[1] : KODES_DEFAULT_SETTINGS);

    printf
    (
        "\n%s: %d species, %d reactions\n",
        KODES_MECHANISM_NAME, label(NSP), label(FWD_RATES)
    );

    settings.print();

    const char* method = settings.method().c_str();
    const char* balancer = settings.balancer().c_str();

    label ensembleSize = settings.ensembleSize();

    kodes::HostResources hostRes(ensembleSize, NSP, 1);

    set_same_initial_conditions(hostRes.ensembleSize(), hostRes.vectors, hostRes.parameters);

    printf("\ninitial state of reactor 0:\n  ");
    hostRes.printVectori(0);

    // The two names carry what the classes they select will cost: the method's
    // entry knows what its resources need, the balancer's knows what the keys
    // and the order need. pyJac's own per-thread scratch is owned by neither,
    // so it is the one thing left to declare.
    kodes::LaunchConfig config = kodes::planLaunch
    (
        ensembleSize,
        hostRes.systemSize(),
        hostRes.parameterSize(),
        method,
        balancer,
        required_mechanism_size(),
        settings.launchRequest()
    );

    config.print(KODES_MECHANISM_NAME);

    label batchSize = config.batchSize;
    label numOfBatches = config.numOfBatches(ensembleSize);

    mechanism_memory* hostMechanismMemory = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    mechanism_memory* deviceMechanismMemory = nullptr;

    // pyJac's scratch is per thread, so it is padded to the resident threads
    initialize_gpu_memory(config.scratchSize, &hostMechanismMemory, &deviceMechanismMemory);

    // Each handle owns a device object and the host stub holding its buffers,
    // and hands both back when it goes out of scope.
    kodes::Handle<kodes::DeviceResources> resources = kodes::newResources
    (
        method, batchSize, config.scratchSize,
        hostRes.systemSize(), hostRes.parameterSize()
    );

    kodes::Handle<kodes::IntegrationMethod> integrationMethod = kodes::newMethod
    (
        method, batchSize, config.scratchSize,
        hostRes.systemSize(), hostRes.parameterSize()
    );

    // an empty handle when the settings say "none", which leaves the batch in
    // the order it was copied in
    kodes::Handle<kodes::Balancer> balancing = kodes::newBalancer
    (
        balancer, batchSize, config.scratchSize,
        hostRes.systemSize(), hostRes.parameterSize()
    );

    kodes::PyJacSystem* ode = kodes::PyJacSystem::create(deviceMechanismMemory);

    kodes::Operator op(&hostRes, resources.host());

    kodes::Integrator solver
    (
        ode, resources.device(), integrationMethod.device(),
        config, settings.controls()
    );

    solver.setBalancer(balancing.device(), balancing.host());

    scalar endTime = settings.endTime();
    solver.setDeltaT(settings.initialTimeStep());

    const auto wallStart = std::chrono::steady_clock::now();

    for (label i = 0; i < numOfBatches; i++)
    {
        op.cpyHostToDevice(i);
        solver.solve(endTime, op.getRealBatchSize(i));
        op.cpyDeviceToHost(i);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    const auto wallEnd = std::chrono::steady_clock::now();

    printf("\nfinal state of reactor 0 at t = %g s:\n  ", endTime);
    hostRes.printVectori(0);

    const scalar wallSeconds =
        std::chrono::duration<scalar>(wallEnd - wallStart).count();

    printf
    (
        "\n%d reactors in %d batch(es): %.3f s wall, %.1f reactors/s\n\n",
        ensembleSize, numOfBatches, wallSeconds, ensembleSize/wallSeconds
    );

    kodes::PyJacSystem::destroy(ode);

    free_gpu_memory(&hostMechanismMemory, &deviceMechanismMemory);
    free(hostMechanismMemory);

    return 0;
}
