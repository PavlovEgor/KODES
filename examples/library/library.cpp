#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "library_api.h"

// An ensemble of identical constant-pressure reactors, integrated by a
// prebuilt KODES - the reactors example again, from a program that has no
// CUDA in it. Everything it knows about the mechanism it asks the library.
//
//     ./library h2o2                   opens libkodes_h2o2.so from the
//                                      library path
//     ./library path/to/libkodes_h2o2.so [reactors [method]]
//
// The mixture is stoichiometric hydrogen and oxygen, with the rest of the mass
// in whatever specie the mechanism leaves out of the state - the bath gas of
// every mechanism this has been tried with - at 1500 K and 1 atm.

static void fail(const char* message, const char* detail)
{
    fprintf(stderr, "library: %s%s\n", message, detail ? detail : "");
    exit(EXIT_FAILURE);
}

static void* findSymbol(void* handle, const char* name)
{
    void* symbol = dlsym(handle, name);

    if (!symbol)
    {
        fail("not a KODES library, no symbol ", name);
    }

    return symbol;
}

// The state component holding a specie's mass fraction: one past its index,
// since component 0 is the temperature
static int component(const kodes::Library& library, const char* name)
{
    for (int i = 0; i < library.nSpecie() - 1; ++i)
    {
        if (strcmp(library.specie(i), name) == 0)
        {
            return i + 1;
        }
    }

    fail("the mechanism has no specie in its state named ", name);

    return -1;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        fail("usage: library <mechanism | path to libkodes_<mechanism>.so>", nullptr);
    }

    std::string path = argv[1];

    if (path.find(".so") == std::string::npos)
    {
        path = "libkodes_" + path + ".so";
    }

    const int ensembleSize = argc > 2 ? atoi(argv[2]) : 65536;

    kodes::SolverSettings settings;

    settings.method = argc > 3 ? argv[3] : "seulex";
    settings.relTol = 1e-3;

    // RTLD_LOCAL: nothing of one KODES library is visible to the next
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);

    if (!handle)
    {
        fail("", dlerror());
    }

    const int libraryApiVersion = reinterpret_cast<kodes::LibraryApiVersionFunction>
    (
        findSymbol(handle, kodes::kLibraryApiVersionSymbol)
    )();

    if (libraryApiVersion != kodes::kLibraryApiVersion)
    {
        fprintf
        (
            stderr,
            "library: %s has interface version %d, this program was built for %d\n",
            path.c_str(), libraryApiVersion, kodes::kLibraryApiVersion
        );
        return EXIT_FAILURE;
    }

    const kodes::Library& library = *reinterpret_cast<kodes::LibraryFunction>
    (
        findSymbol(handle, kodes::kLibrarySymbol)
    )();

    printf
    (
        "\n%s: KODES %s, %s: %d species, %d reactions, %s left out of the state\n",
        path.c_str(), library.version(), library.mechanism(),
        library.nSpecie(), library.nReaction(),
        library.specie(library.nSpecie() - 1)
    );

    // Component-major, one array per component, the way a CFD code holds its
    // fields
    std::vector<std::vector<double>> state
    (
        library.systemSize(), std::vector<double>(ensembleSize, 0)
    );

    std::vector<std::vector<double>> parameters
    (
        library.parameterSize(), std::vector<double>(ensembleSize, 101325)
    );

    for (int system = 0; system < ensembleSize; ++system)
    {
        state[0][system] = 1500;
        state[component(library, "H2")][system] = 0.0285;
        state[component(library, "O2")][system] = 0.2264;
    }

    std::vector<double*> statePointers;
    std::vector<double*> parameterPointers;

    for (std::vector<double>& c : state)
    {
        statePointers.push_back(c.data());
    }

    for (std::vector<double>& c : parameters)
    {
        parameterPointers.push_back(c.data());
    }

    kodes::Solver* solver = library.newSolver(ensembleSize, settings);

    printf
    (
        "%s: %d blocks x %d threads, %d reactors per batch, %d batch(es)\n\n",
        solver->method(), solver->blocks(), solver->threads(),
        solver->batchSize(), solver->numOfBatches()
    );

    solver->setDeltaT(1e-7);

    const double deltaT = 2e-5;

    for (int step = 1; step <= 10; ++step)
    {
        solver->solve(deltaT, statePointers.data(), parameterPointers.data());

        printf
        (
            "t = %.1e s: T = %.2f K in reactor 0, %.2f K in reactor %d\n",
            step*deltaT, state[0][0], state[0][ensembleSize - 1],
            ensembleSize - 1
        );
    }

    delete solver;

    // The library stays open until the program ends: unloading a CUDA module
    // while the runtime is still up gains nothing and can only go wrong.
    return EXIT_SUCCESS;
}
