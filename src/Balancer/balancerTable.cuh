#ifndef KODES_BALANCER_TABLE
#define KODES_BALANCER_TABLE

#pragma once

#include "Balancer.cuh"
#include "typeTable.cuh"

// The balancers a name may select, and how to build the one it selects.
//
// This header is safe for any caller to include - it declares, it does not
// define, so nothing here makes a caller emit a construction kernel or a host
// vtable. The table itself is in balancerTable.cu.

namespace kodes
{

// The name that switches the balancing off. Also what an empty entry in a
// settings file means.
#define KODES_NO_BALANCER "none"

// Every balancer this build knows, in the order they are listed to the user
__host__ const TypeEntry<Balancer>* balancerTable();

__host__ label balancerTableSize();

// The entry `name` selects, or null for KODES_NO_BALANCER. Fails with the list
// of known names if it is neither.
__host__ const TypeEntry<Balancer>* balancerType(const char* name);

// What that balancer would cost, before there is one - planLaunch needs both
// numbers to size the run that the balancer is then built against.
__host__ size_t balancerStateBytesPerSystem
(
    const char* name,
    const label systemSize,
    const label parameterSize
);

__host__ size_t balancerScratchBytesPerThread
(
    const char* name,
    const label systemSize,
    const label parameterSize
);

// Build it. An empty handle for KODES_NO_BALANCER, which Integrator takes to
// mean the batch is integrated in the order it was copied in.
__host__ Handle<Balancer> newBalancer
(
    const char* name,
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    const label parameterSize
);

}

#endif
