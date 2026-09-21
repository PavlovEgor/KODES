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

// One numerical method for advancing a block of systems by one step.
//
// A device object in the sense of Factory/deviceObject.cuh, exactly like a
// Balancer: it lives in device memory, the solve kernel dispatches on it
// through its vtable, and which subclass it is is decided at run time by a name
// out of methodTable. Before, the method was a template argument of the whole
// Integrator and the choice was made when the program was compiled.
//
// Every entry point here is *collective*: the whole block calls it, and it
// contains barriers. A step size, an order and an accept-or-reject are one
// value for the block, held in the BlockStep every thread carries a copy of,
// so that the branches taken around those barriers are the same in all of
// them. A thread whose system is inactive - past the end of the batch, or too
// cold to react - still makes the call, with `active` false, and takes no
// part in anything but the reductions.
//
// The indirect call costs nothing measurable here. It happens once per step of
// a system - against an implicit step that factorises a systemSize^2 matrix -
// and the kernel already makes the same kind of call twice per step, into
// ODESystem::derivatives and ODESystem::jacobian, so it was never going to
// inline the arithmetic anyway.
//
// A method has no storage of its own: every temporary it needs is a scratch
// slot of the DeviceResources subclass registered alongside it, which is why
// the two are named by one entry of the table and step() may cast down to it.
class IntegrationMethod
{
protected:

    label   batchSize_;
    label   scratchSize_;
    label   blocks_;
    label   systemSize_;
    label   parameterSize_;

    // Whether step() is a *trial* step, whose error estimate adaptiveStep()
    // then accepts or rejects, or a method that runs its own step size control
    // inside step() and returns nothing to judge.
    bool    usesAdaptiveStep_;

public:

    __device__ __host__
    IntegrationMethod
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize,
        const bool usesAdaptiveStep
    )
        : batchSize_(batchSize),
          scratchSize_(scratchSize),
          blocks_(blocks),
          systemSize_(systemSize),
          parameterSize_(parameterSize),
          usesAdaptiveStep_(usesAdaptiveStep)
    {}

    __device__ __host__
    virtual ~IntegrationMethod() = default;

    __device__ static void* operator new(size_t size, void* ptr) { return ptr; }

    // Advance the systems the calling block holds in its threads' scratch
    // slots by one step of step.deltaTTry.
    //
    // `resources` is the subclass the method's own table entry created, so an
    // implementation is free to cast it down to the one holding its scratch.
    //
    // Returns the error of the trial step over the whole block, normalised so
    // that 1 is the tolerance, when usesAdaptiveStep() is set; the value is
    // ignored otherwise. Being a block figure, it is the same in every thread.
    __device__ virtual scalar
    step
    (
        ODESystem* ode,
        DeviceResources* resources,
        BlockStep& step,
        const bool active,
        IntegratorControls controls
    ) const = 0;

    __device__ __host__ bool usesAdaptiveStep() const { return usesAdaptiveStep_; }

    __device__ __host__ label systemSize() const { return systemSize_; }

    // The step size controller shared by every method that only knows how to
    // take a trial step: retry with a smaller step until the error is inside
    // the tolerance, then grow the next one. Needs the two extra scratch
    // vectors of AdaptiveDeviceResources.
    __device__ void
    adaptiveStep
    (
        ODESystem* ode,
        DeviceResources* resources,
        BlockStep& step,
        const bool active,
        IntegratorControls controls
    ) const;

    // What the solve kernel calls: one accepted step, whichever way the method
    // arrives at one.
    __device__ void
    advance
    (
        ODESystem* ode,
        DeviceResources* resources,
        BlockStep& blockStep,
        const bool active,
        IntegratorControls controls
    ) const
    {
        if (usesAdaptiveStep_)
        {
            adaptiveStep(ode, resources, blockStep, active, controls);
        }
        else
        {
            step(ode, resources, blockStep, active, controls);
        }
    }

    // A method owns no device memory - the resources registered with it own all
    // of it - but it is built by the same factory as everything else, so it
    // answers the same four questions.
    __host__ void allocate() {}

    __host__ void deallocate() {}

    __host__ static size_t
    scratchBytesPerThread(const label systemSize, const label parameterSize)
    {
        return 0;
    }

    __host__ static size_t
    scratchBytesPerBlock(const label systemSize, const label parameterSize)
    {
        return 0;
    }

    __host__ static size_t
    stateBytesPerSystem(const label systemSize, const label parameterSize)
    {
        return 0;
    }
};

}

#endif
