#ifndef KODES_METHOD_TABLE
#define KODES_METHOD_TABLE

#pragma once

#include "IntegrationMethod.cuh"
#include "DeviceResources.cuh"
#include "typeTable.cuh"

// The integration methods a name may select, and how to build the one it
// selects. The balancer's table with one addition: a method and the resources
// holding its scratch are chosen together, so one entry names both.
//
// Safe for any caller to include - it declares, it does not define.

namespace kodes
{

struct MethodType
{
    const char* name;

    TypeEntry<IntegrationMethod> method;

    // The DeviceResources subclass carrying the scratch this method's step()
    // casts down to reach. Nothing else may be handed to it.
    TypeEntry<DeviceResources> resources;
};

__host__ const MethodType* methodTable();

__host__ label methodTableSize();

// The entry `name` selects. Fails with the list of known names if there is no
// such method.
__host__ const MethodType* methodType(const char* name);

// What the method and its resources cost together, before there are any -
// planLaunch needs both numbers to size the run they are then built against.
__host__ size_t methodScratchBytesPerThread
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
    const label systemSize,
    const label parameterSize
);

// The resources that go with that method. Built separately from the method
// because the Operator needs the host stub of these and nothing else does.
__host__ Handle<DeviceResources> newResources
(
    const char* name,
    const label batchSize,
    const label scratchSize,
    const label systemSize,
    const label parameterSize
);

}

#endif
