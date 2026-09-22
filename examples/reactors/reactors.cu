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
    const char* odeSystem = settings.system().c_str();

    label ensembleSize = settings.ensembleSize();

    kodes::HostResources hostRes(ensembleSize, NSP, 1);

    set_same_initial_conditions(hostRes.ensembleSize(), hostRes.vectors, hostRes.parameters);

    printf("\ninitial state of reactor 0:\n  ");
    hostRes.printVectori(0);

    // The three names carry what the classes they select will cost: the
    // method's entry knows what its resources need, the balancer's knows what
    // the keys and the order need, and the system's knows what pyJac's
    // per-thread workspace needs. Nothing is left for the caller to declare.
    kodes::LaunchConfig config = kodes::planLaunch
    (
        ensembleSize,
        hostRes.systemSize(),
        hostRes.parameterSize(),
        method,
        balancer,
        odeSystem,
        settings.launchRequest()
    );

    config.print(KODES_MECHANISM_NAME);

    label batchSize = config.batchSize;
    label numOfBatches = config.numOfBatches(ensembleSize);

    // Each handle owns a device object and the host stub holding its buffers,
    // and hands both back when it goes out of scope. pyJac's own per-thread
    // scratch is one of those buffers - the system's allocate() pads it to
    // config.scratchSize - so there is nothing to initialize or free here.
    kodes::Handle<kodes::DeviceResources> resources = kodes::newResources
    (
        method, batchSize, config.scratchSize, config.blocks,
        hostRes.systemSize(), hostRes.parameterSize()
    );

    kodes::Handle<kodes::IntegrationMethod> integrationMethod = kodes::newMethod
    (
        method, batchSize, config.scratchSize, config.blocks,
        hostRes.systemSize(), hostRes.parameterSize()
    );

    // an empty handle when the settings say "none", which leaves the batch in
    // the order it was copied in
    kodes::Handle<kodes::Balancer> balancing = kodes::newBalancer
    (
        balancer, batchSize, config.scratchSize, config.blocks,
        hostRes.systemSize(), hostRes.parameterSize()
    );

    kodes::Handle<kodes::ODESystem> ode = kodes::newSystem
    (
        odeSystem, batchSize, config.scratchSize, config.blocks,
        hostRes.systemSize(), hostRes.parameterSize()
    );

    kodes::Operator op(&hostRes, resources.host());

    kodes::Integrator solver
    (
        ode.device(), resources.device(), integrationMethod.device(),
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

    return 0;
}
