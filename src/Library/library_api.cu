#include "library_api.h"

#include <type_traits>

#include "basic_types.cuh"
#include "HostResources.cuh"
#include "DeviceResources.cuh"
#include "Operator.cuh"

#include "Integrator.cuh"
#include "IntegrationMethod.cuh"
#include "method_table.cuh"
#include "system_table.cuh"

// The mechanism this library is built for. Only NSP and FWD_RATES come from
// here: the system itself is built through system_table.cuh, like everywhere
// else.
#include "mechanism.cuh"

// Its specie names, as KODES_MECHANISM_SPECIES. pyJac writes them only into a
// comment of mechanism.cuh, so cmake/kodes.cmake pulls them out of it into this
// generated header when the library is configured.
#include "mechanism_species.h"

// What dlsym() can see. The library is compiled -fvisibility=hidden and linked
// with --exclude-libs,ALL, so these two are the only symbols it exports.
#define KODES_LIBRARY_EXPORT extern "C" __attribute__((visibility("default")))

// The header promises the caller double and int without including anything of
// KODES; this is where that promise is checked
static_assert(std::is_same<scalar, double>::value, "library_api.h says double");
static_assert(std::is_same<label, int>::value, "library_api.h says int");

static_assert
(
    kodes::SolverSettings().threadsPerBlock == KODES_BLOCK_SIZE,
    "SolverSettings must default to the block size the library does"
);

static_assert
(
    kodes::SolverSettings().deviceStackBytes == KODES_DEVICE_STACK_BYTES,
    "SolverSettings must default to the device stack the library does"
);

namespace kodes
{

static const char* const kMechanismSpecies[] = { KODES_MECHANISM_SPECIES };

static_assert
(
    sizeof(kMechanismSpecies)/sizeof(kMechanismSpecies[0]) == NSP,
    "the Species Indexes block of mechanism.cuh does not list NSP species"
);

// A pyJac system carries one parameter, the pressure
constexpr label kPyJacParameterSize = 1;

// The three runtime-selected objects, the host buffers the Operator copies
// through and the Integrator driving them, in the order they have to be built
// in: the plan first, since every one of the rest is sized by it.
//
// There is no balancer. Every thread integrates its own system on its own, so
// the order the batch is taken in would change nothing but which thread gets
// which system.
class LibrarySolver
:
    public Solver
{
    const label ensembleSize_;

    HostResources host_;

    LaunchConfig config_;

    Handle<DeviceResources> resources_;
    Handle<IntegrationMethod> method_;
    Handle<ODESystem> system_;

    Operator operator_;

    Integrator integrator_;

    static LaunchConfig request(const SolverSettings& settings)
    {
        const bool hasConcurrent = settings.concurrentSystems > 0;
        const bool hasBatch = settings.batchSize > 0;

        if (hasConcurrent != hasBatch)
        {
            fprintf
            (
                stderr,
                "kodes::Library error at %s:%d: concurrentSystems %d and batchSize %d"
                " size a run by hand only together, give both or neither\n",
                __FILE__, __LINE__, settings.concurrentSystems, settings.batchSize
            );
            std::exit(EXIT_FAILURE);
        }

        if (hasConcurrent)
        {
            return LaunchConfig
            (
                settings.concurrentSystems,
                settings.batchSize,
                settings.threadsPerBlock
            );
        }

        return LaunchConfig(settings.deviceShare, settings.threadsPerBlock);
    }

public:

    LibrarySolver(const label ensembleSize, const SolverSettings& settings)
    :
        ensembleSize_(ensembleSize),
        host_(ensembleSize, NSP, kPyJacParameterSize),
        config_
        (
            planLaunch
            (
                ensembleSize,
                NSP,
                kPyJacParameterSize,
                settings.method,
                kDefaultSystem,
                request(settings),
                0,
                settings.deviceStackBytes
            )
        ),
        resources_
        (
            newResources
            (
                settings.method, config_.batchSize, config_.scratchSize,
                config_.blocks, NSP, kPyJacParameterSize
            )
        ),
        method_
        (
            newMethod
            (
                settings.method, config_.batchSize, config_.scratchSize,
                config_.blocks, NSP, kPyJacParameterSize
            )
        ),
        system_
        (
            newSystem
            (
                kDefaultSystem, config_.batchSize, config_.scratchSize,
                config_.blocks, NSP, kPyJacParameterSize
            )
        ),
        // the Operator reads the device addresses off the host stub
        operator_(&host_, resources_.host()),
        integrator_
        (
            system_.device(),
            resources_.device(),
            method_.device(),
            config_,
            IntegratorControls
            (
                settings.absTol,
                settings.relTol,
                settings.maxSteps,
                settings.Treact
            )
        )
    {}

    int threads() const override { return config_.threads; }

    int blocks() const override { return config_.blocks; }

    int concurrentSystems() const override { return config_.scratchSize; }

    int batchSize() const override { return config_.batchSize; }

    int numOfBatches() const override
    {
        return config_.numOfBatches(ensembleSize_);
    }

    bool blockScratchInShared() const override
    {
        return config_.blockScratchInShared;
    }

    size_t deviceStack() const override { return kodes::deviceStack(); }

    const char* method() const override { return method_.name(); }

    void setDeltaT(const double deltaT) override
    {
        integrator_.setDeltaT(deltaT);
    }

    void solve
    (
        const double deltaT,
        double* const* state,
        double* const* parameters
    ) override
    {
        // The Operator copies through whatever host_ points at when it is
        // called, so pointing it at the caller's arrays is the whole hand-over
        for (label i = 0; i < host_.systemSize(); ++i)
        {
            host_.setVector(i, state[i]);
        }

        for (label i = 0; i < host_.parameterSize(); ++i)
        {
            host_.setParameter(i, parameters[i]);
        }

        const label numOfBatches = config_.numOfBatches(ensembleSize_);

        for (label batchIndex = 0; batchIndex < numOfBatches; ++batchIndex)
        {
            operator_.cpyHostToDevice(batchIndex);
            integrator_.solve(deltaT, operator_.getRealBatchSize(batchIndex));
            operator_.cpyDeviceToHost(batchIndex);
        }
    }
};

class LibraryMechanism
:
    public Library
{
public:

    const char* mechanism() const override { return KODES_MECHANISM_NAME; }

    const char* version() const override { return KODES_VERSION; }

    int nSpecie() const override { return NSP; }

    int nReaction() const override { return FWD_RATES; }

    const char* specie(const int i) const override
    {
        if (i < 0 || i >= NSP)
        {
            fprintf
            (
                stderr,
                "kodes::Library error at %s:%d: specie %d of a mechanism of %d\n",
                __FILE__, __LINE__, i, label(NSP)
            );
            std::exit(EXIT_FAILURE);
        }

        return kMechanismSpecies[i];
    }

    // temperature and NSP-1 mass fractions
    int systemSize() const override { return NSP; }

    int parameterSize() const override { return kPyJacParameterSize; }

    Solver* newSolver
    (
        const int ensembleSize,
        const SolverSettings& settings
    ) const override
    {
        return new LibrarySolver(ensembleSize, settings);
    }
};

}

// The names have to be the ones library_api.h gives as kLibraryApiVersionSymbol
// and kLibrarySymbol
KODES_LIBRARY_EXPORT int kodesLibraryApiVersion()
{
    return kodes::kLibraryApiVersion;
}

KODES_LIBRARY_EXPORT const kodes::Library* kodesLibrary()
{
    static const kodes::LibraryMechanism library;

    return &library;
}
