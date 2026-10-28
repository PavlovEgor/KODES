#ifndef KODES_LIBRARY_API
#define KODES_LIBRARY_API

#pragma once

#include <stddef.h>

// The interface of a prebuilt KODES: libkodes_<mechanism>.so, one shared
// library per pyJac mechanism, opened with dlopen() by a program that has no
// CUDA compiler and includes nothing of KODES but this header.
//
// Why one library per mechanism, and why only this much crosses into it. The
// device code is compiled -rdc=true and the solve kernel reaches the method,
// the balancer and the mechanism through device-only virtuals, which are
// resolved at *device* link time. CUDA does not link device code across shared
// objects, so everything a kernel can call - the mechanism included - has to go
// through one device link, and that link is done when the .so is built. What is
// left for the caller is host code only: the classes below, plain C++ with no
// CUDA type in them, so this header compiles with any C++ compiler.
//
// The types are spelled `double` and `int`, not `scalar` and `label`. Those are
// KODES's names for exactly these two, and a caller with a scalar and a label
// of its own - OpenFOAM built with 64-bit labels, say - converts at the call
// rather than being handed a typedef that need not match its own.
//
// Two symbols are exported, both extern "C" so that dlsym() finds them by these
// names; everything else in the library is hidden. The version is a plain
// function of its own because it is the one call that has to keep working when
// the layout of the classes below does not.

namespace kodes
{

// Raised whenever anything below changes shape: a virtual added, removed or
// reordered, a member of SolverSettings added or retyped. A caller compares it
// with the library's before it touches anything else.
//
// 2 is this line, kodes-v2; 1 is kodes-v1, whose SolverSettings has no
// balancer and whose Solver no balancer(). Both lines started out as 1.
constexpr int kLibraryApiVersion = 2;

constexpr const char* kLibraryApiVersionSymbol = "kodesLibraryApiVersion";
constexpr const char* kLibrarySymbol = "kodesLibrary";

class Library;

typedef int (*LibraryApiVersionFunction)();
typedef const Library* (*LibraryFunction)();

// Everything a run is chosen and sized by: the names and numbers of a KODES
// settings file (see Settings/Settings.cuh), less the mechanism, which is the
// library itself. The defaults are the library's own.
struct SolverSettings
{
    // seulex | euler
    const char* method = "seulex";

    // temperature | rhsNorm | stiffness | none. Null or empty is none.
    const char* balancer = "temperature";

    // How much of the device one run may take: best | half
    const char* deviceShare = "best";

    // Size the run by hand instead: systems integrated at the same time, and
    // systems per host->device transfer. Both or neither, and 0 is neither;
    // when both are given deviceShare is not read.
    int concurrentSystems = 0;
    int batchSize = 0;

    int threadsPerBlock = 256;

    // Bytes of call stack per device thread, or 0 to leave the driver's own
    // value alone. See KODES_DEVICE_STACK_BYTES before raising it.
    size_t deviceStackBytes = 0;

    double absTol = 1e-12;
    double relTol = 1e-1;
    int maxSteps = 10000;

    // A system whose first component - the temperature - is at or below this
    // is left as it is
    double Treact = 0;
};

// One run: a plan resolved against the device, and the method, the balancer,
// the mechanism and their buffers built against that plan. Made by
// Library::newSolver(), and given back with delete.
class Solver
{
public:

    virtual ~Solver() {}

    // The plan, for the caller to report
    virtual int threads() const = 0;

    virtual int blocks() const = 0;

    // Systems integrated at the same time, threads()*blocks()
    virtual int concurrentSystems() const = 0;

    // Systems per host->device transfer
    virtual int batchSize() const = 0;

    // Transfers per solve(), the last one usually short
    virtual int numOfBatches() const = 0;

    virtual bool blockScratchInShared() const = 0;

    // The call stack of a device thread as the driver has it now
    virtual size_t deviceStack() const = 0;

    // What the settings selected, "none" for no balancer
    virtual const char* method() const = 0;

    virtual const char* balancer() const = 0;

    // The trial step every block starts its first solve() from. Afterwards each
    // block carries over the step it ended the previous batch with.
    virtual void setDeltaT(const double deltaT) = 0;

    // Integrate every system of the ensemble over deltaT, in place.
    //
    // Component-major, like KODES's HostResources: state[i][system] for
    // i < Library::systemSize() and parameters[i][system] for
    // i < Library::parameterSize(), each array one entry per system. For a pyJac
    // mechanism component 0 is the temperature, component i the mass fraction of
    // specie i-1 in the mechanism's order - the last specie is not in the state
    // and follows from the others - and the one parameter is the pressure.
    //
    // The pointers are read anew on every call. The parameters are copied back
    // along with the state, unchanged, so they too have to be writable.
    virtual void solve
    (
        const double deltaT,
        double* const* state,
        double* const* parameters
    ) = 0;
};

// The mechanism the library was built for, and the factory of solvers for it.
// There is one, owned by the library: kodesLibrary() hands out its address.
class Library
{
public:

    // The name the library was built under, the X of libkodes_X.so
    virtual const char* mechanism() const = 0;

    // The KODES version it was built from
    virtual const char* version() const = 0;

    virtual int nSpecie() const = 0;

    virtual int nReaction() const = 0;

    // Specie i in the mechanism's own order, for i < nSpecie(). The last one is
    // the specie the state leaves out.
    virtual const char* specie(const int i) const = 0;

    // Components of one system's state, and its parameters
    virtual int systemSize() const = 0;

    virtual int parameterSize() const = 0;

    // Plan a run of ensembleSize systems against the free memory of the current
    // device and build it. An unknown name stops the program with the list of
    // the known ones, like everything else in KODES that is chosen by name.
    virtual Solver* newSolver
    (
        const int ensembleSize,
        const SolverSettings& settings
    ) const = 0;

protected:

    // Not deleted through this: the library owns the only one
    ~Library() {}
};

}

#endif
