#include "Settings.cuh"
#include "methodTable.cuh"

#include <stdexcept>

// A missing or malformed file is a mistake in the case, not something to throw
// at a caller that has no handler; reported the way the rest of the library
// reports one.
kodes::Config kodes::Settings::openOrExit(const std::string& path)
{
    try
    {
        return Config(path);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "kodes::Settings error: %s\n", e.what());
        std::exit(EXIT_FAILURE);
    }
}

kodes::Settings::Settings(const std::string& path)
:
config_(openOrExit(path)),
path_(path),
method_(config_.getString("method", "seulex")),
balancer_(config_.getString("balancer", KODES_NO_BALANCER))
{
    // Both look their name up in the table straight away rather than when the
    // object is built, so a typo fails here - before the plan, before pyJac's
    // scratch, before anything has been allocated on the device.
    methodType(method_.c_str());
    balancerType(balancer_.c_str());
}

kodes::LaunchConfig kodes::Settings::launchRequest() const
{
    const label threads =
        config_.getInt("device.threadsPerBlock", KODES_BLOCK_SIZE);

    // Sizing the run by hand takes both numbers: how many systems run at the
    // same time, and how many of them travel per transfer
    const bool hasConcurrent = config_.hasKey("device.concurrentSystems");
    const bool hasBatch = config_.hasKey("device.batchSize");

    if (hasConcurrent != hasBatch)
    {
        fprintf
        (
            stderr,
            "kodes::Settings warning: %s gives only one of "
            "device.concurrentSystems and device.batchSize. Both are needed to "
            "size the run by hand, so the plan is left to device.share and the "
            "entry is ignored.\n",
            path_.c_str()
        );
    }

    if (hasConcurrent && hasBatch)
    {
        return LaunchConfig
        (
            config_.getInt("device.concurrentSystems", 0),
            config_.getInt("device.batchSize", 0),
            threads
        );
    }

    return LaunchConfig(config_.getString("device.share", "best").c_str(), threads);
}

kodes::IntegratorControls kodes::Settings::controls() const
{
    IntegratorControls controls
    (
        config_.getDouble("controls.absTol", 1e-10),
        config_.getDouble("controls.relTol", 1e-1),
        config_.getInt("controls.maxSteps", 10000),
        config_.getDouble("controls.Treact", 0.0)
    );

    controls.safeScale = config_.getDouble("controls.safeScale", controls.safeScale);
    controls.alphaIncrease = config_.getDouble("controls.alphaIncrease", controls.alphaIncrease);
    controls.alphaDecrease = config_.getDouble("controls.alphaDecrease", controls.alphaDecrease);
    controls.minScale = config_.getDouble("controls.minScale", controls.minScale);
    controls.maxScale = config_.getDouble("controls.maxScale", controls.maxScale);

    return controls;
}

label kodes::Settings::ensembleSize() const
{
    return config_.getInt("run.ensembleSize", 3 * 8192);
}

scalar kodes::Settings::endTime() const
{
    return config_.getDouble("run.endTime", 10.0);
}

scalar kodes::Settings::initialTimeStep() const
{
    // The first trial step of every thread slot. Seulex is happy to be handed
    // the whole interval and cut it down itself; a method taking trial steps
    // wants something it can actually take.
    return config_.getDouble("run.initialTimeStep", endTime());
}

void kodes::Settings::print() const
{
    const IntegratorControls c = controls();

    printf("kodes settings from %s:\n", path_.c_str());
    printf("  method   = %s\n", method_.c_str());
    printf("  balancer = %s\n", balancer_.c_str());
    printf
    (
        "  controls = absTol %g, relTol %g, maxSteps %d, Treact %g\n",
        c.absTol, c.relTol, c.maxSteps, c.Treact
    );
    printf
    (
        "  run      = %d systems to t = %g, first step %g\n",
        ensembleSize(), endTime(), initialTimeStep()
    );
}
