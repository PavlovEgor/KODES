#ifndef KODES_INTEGRATION_METHOD
#define KODES_INTEGRATION_METHOD

#pragma once

#include "basic_types.cuh"
#include "device_object.cuh"
#include "DeviceResources.cuh"
#include "ODESystem.cuh"
#include "IntegratorControls.cuh"

namespace kodes
{

// One numerical method for advancing one system by one step, together with
// the scratch it does that in.
//
// A device object in the sense of Factory/device_object.cuh, exactly like an
// ODESystem: it lives in device memory, the solve kernel dispatches on it
// through its vtable, and which subclass it is is decided at run time by a name
// out of methodTable.
//
// A method *is* its resources, as OpenFOAM's seulex is its own work arrays: the
// batch and the scratch every method needs come from DeviceResources, and a
// subclass adds its own scratch as members and uses them by name. Until 1.2
// they were two objects, the method and a DeviceResources subclass chosen by
// the same row of the table, and every step began by casting the one it was
// handed down to the one it knew it was.
//
// Every entry point here is called by one thread for the system in its own
// scratch slot, and nothing in it waits for another thread: each thread keeps
// its own step size, order and accept-or-reject, in its slot of StepState.
//
// The indirect call costs nothing measurable here. It happens once per step of
// a system - against an implicit step that factorises a systemSize^2 matrix -
// and the kernel already makes the same kind of call twice per step, into
// ODESystem::derivatives and ODESystem::jacobian, so it was never going to
// inline the arithmetic anyway.
class IntegrationMethod
    :
    public DeviceResources
{
public:

    __device__ __host__
    IntegrationMethod
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : DeviceResources(batchSize, scratchSize, blocks, systemSize, parameterSize)
    {}

    __device__ __host__
    virtual ~IntegrationMethod() = default;

    // Advance the system the calling thread holds in its scratch slot by one
    // accepted step, starting with deltaTTry: either the method runs its own
    // step size control, as Seulex does, or it is an AdaptiveIntegrationMethod,
    // whose trial steps a shared controller accepts or rejects.
    __device__ virtual void
    advance
    (
        ODESystem* ode,
        IntegratorControls controls
    ) const = 0;
};

}

#endif
