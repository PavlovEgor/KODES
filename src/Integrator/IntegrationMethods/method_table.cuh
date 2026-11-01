#ifndef KODES_METHOD_TABLE
#define KODES_METHOD_TABLE

#pragma once

#include "IntegrationMethod.cuh"
#include "type_table.cuh"

// The integration methods a name may select, and how to build the one it
// selects. Until 1.2 an entry named two classes, a method and the resources
// holding its scratch; now a method is its own resources, and the table is
// the ODE system's again.
//
// Safe for any caller to include - it declares, it does not define.

namespace kodes
{

__host__ const TypeEntry<IntegrationMethod>* methodTable();

__host__ label methodTableSize();

// The entry `name` selects. Fails with the list of known names if there is no
// such method.
__host__ const TypeEntry<IntegrationMethod>* methodType(const char* name);

// What the method costs, before there is one - planLaunch needs it to size the
// run it is then built against.
__host__ size_t methodScratchBytesPerThread
(
    const char* name,
    const label systemSize,
    const label parameterSize
);

__host__ size_t methodScratchBytesPerBlock
(
    const char* name,
    const label systemSize,
    const label parameterSize
);

__host__ size_t methodStateBytesPerSystem
(
    const char* name,
    const label systemSize,
    const label parameterSize
);

__host__ Handle<IntegrationMethod> newMethod
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
