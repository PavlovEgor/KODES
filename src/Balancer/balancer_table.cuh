#ifndef KODES_BALANCER_TABLE
#define KODES_BALANCER_TABLE

#pragma once

#include "Balancer.cuh"
#include "type_table.cuh"

// The balancers a name may select, and how to build the one it selects. The
// method's table again, with one difference: a run may name none.
//
// Safe for any caller to include - it declares, it does not define.

namespace kodes
{

// The name that switches the balancing off. Also what an empty name means.
constexpr const char* kNoBalancer = "none";

__host__ const TypeEntry<Balancer>* balancerTable();

__host__ label balancerTableSize();

// The entry `name` selects, or null for kNoBalancer. Fails with the list of
// known names if it is neither.
__host__ const TypeEntry<Balancer>* balancerType(const char* name);

// What that balancer would cost, before there is one - planLaunch adds these to
// the method's and the system's to size the run all three are then built
// against. Zero for kNoBalancer.
__host__ size_t balancerScratchBytesPerThread
(
    const char* name,
    const label systemSize,
    const label parameterSize
);

__host__ size_t balancerScratchBytesPerBlock
(
    const char* name,
    const label systemSize,
    const label parameterSize
);

__host__ size_t balancerStateBytesPerSystem
(
    const char* name,
    const label systemSize,
    const label parameterSize
);

// Build it. An empty handle for kNoBalancer, which Integrator takes to mean
// the batch is integrated in the order it was copied in.
__host__ Handle<Balancer> newBalancer
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
