#ifndef KODES_SYSTEM_TABLE
#define KODES_SYSTEM_TABLE

#pragma once

#include "ODESystem.cuh"
#include "type_table.cuh"

// The ODE systems a name may select, and how to build the one it selects. The
// balancer's table again, for the third abstract base of the library.
//
// This header is safe for any caller to include - it declares, it does not
// define - and that is most of the point of it here. PyJacSystem.cuh pulls in
// pyJac's own generated headers (dydt.cuh, jacob.cuh, mechanism.cuh); a caller
// that goes through this table includes none of them, holds nothing but a
// Handle<ODESystem>, and needs no mechanism on its include path.
//
// The table has one row per build, and will until there is a second kind of
// system: pyJac generates C for one mechanism and NSP is a macro of its
// mechanism.cuh, so "which mechanism" is the library's one compile-time
// choice, made in cmake/kodes.cmake and nowhere in src/. The row is still
// worth having - it is what lets planLaunch ask what the system costs before
// there is one, and Handle own it the way it owns the other three.

namespace kodes
{

// The name of the pyJac-generated system, whichever mechanism this build put
// on the include path. Also the default when a settings file names nothing.
constexpr const char* kDefaultSystem = "pyJac";

// Every ODE system this build knows, in the order they are listed to the user
__host__ const TypeEntry<ODESystem>* systemTable();

__host__ label systemTableSize();

// The entry `name` selects. Fails with the list of known names if there is no
// such system.
__host__ const TypeEntry<ODESystem>* systemType(const char* name);

// What that system would cost, before there is one - planLaunch adds these to
// the method's and the balancer's to size the run all three are then built
// against.
__host__ size_t systemScratchBytesPerThread
(
    const char* name,
    const label systemSize,
    const label parameterSize
);

__host__ size_t systemScratchBytesPerBlock
(
    const char* name,
    const label systemSize,
    const label parameterSize
);

__host__ size_t systemStateBytesPerSystem
(
    const char* name,
    const label systemSize,
    const label parameterSize
);

// Build it. Unlike the balancer there is no "none": a run with no equations to
// integrate is not a run, so an unknown or empty name stops the program.
__host__ Handle<ODESystem> newSystem
(
    const char* name,
    const label batchSize,
    const label scratchSize,
    const label blocks,
    const label systemSize,
    const label parameterSize
);

}

#endif
