#ifndef KODES_ODE_SYSTEM
#define KODES_ODE_SYSTEM
#pragma once

#include "basic_types.cuh"

namespace kodes
{

// The equations a run integrates: the right hand side and its Jacobian.
//
// A device object in the sense of Factory/device_object.cuh, exactly like the
// integration method and the resources. It lives in device
// memory because derivatives() and jacobian() are called from inside the solve
// kernel, whatever buffers it owns are allocated by the host into a stub of
// the same type, and the device object is placement-newed on top of a byte
// copy of that stub.
//
// So the constructor takes the same five sizes every device object of this
// library takes, and - the rule that is easy to break - it must set value
// members only. It runs after the copy that brought the buffer addresses in,
// and a `something_(nullptr)` in its initializer list would silently overwrite
// one. PyJacSystem's mechanism scratch is allocated in allocate(), on the
// stub, for exactly that reason.
//
// Chosen by name out of systemTable(), so which system a run integrates is a
// decision made when the program starts - see ODESystem/system_table.cuh for
// why that table has one row per build.
class ODESystem
{

protected:

    label batchSize_;
    label scratchSize_;
    label blocks_;
    label systemSize_;
    label parameterSize_;

public:

    __device__ __host__
    ODESystem
    (
        const label batchSize,
        const label scratchSize,
        const label blocks,
        const label systemSize,
        const label parameterSize
    )
        : batchSize_(batchSize),
          scratchSize_(scratchSize),
          blocks_(blocks),
          systemSize_(systemSize),
          parameterSize_(parameterSize)
    {}

    __device__ __host__
    virtual ~ODESystem() = default;

    __device__ static void* operator new(size_t size, void* ptr) { return ptr; }

    __device__ virtual void
    derivatives
    (
        const scalar t, const scalar parameter, const scalar* y, scalar* dydt
    ) const = 0;

    // df/dy only: the systems this library integrates are autonomous, so the
    // df/dt this used to take alongside it was a vector every caller allocated
    // per thread, passed in, and never read. pyJac never wrote it either.
    __device__ virtual void
    jacobian
    (
        const scalar t, const scalar parameter, const scalar* y, scalar* dfdy
    ) const = 0;

    __device__ __host__ label batchSize() const { return batchSize_; }

    __device__ __host__ label scratchSize() const { return scratchSize_; }

    __device__ __host__ label blocks() const { return blocks_; }

    __device__ __host__ label systemSize() const { return systemSize_; }

    __device__ __host__ label parameterSize() const { return parameterSize_; }

    // A system with no scratch of its own - one whose derivatives() is a
    // formula - is built by the same four calls with these two inherited
    // empty.
    __host__ void allocate() {}

    __host__ void deallocate() {}

    // An ODE system owns no slot of the batch and nothing a block shares: the
    // state is the resources', and derivatives()/jacobian() are called by one
    // thread on one system. Both stated here once, so a subclass restates
    // neither. What a system does cost - pyJac's per-thread workspace - is
    // scratchBytesPerThread, and that one every subclass must give, since
    // there is no sensible default to inherit.
    __host__ static size_t
    stateBytesPerSystem(const label systemSize, const label parameterSize)
    {
        return 0;
    }

    __host__ static size_t
    scratchBytesPerBlock(const label systemSize, const label parameterSize)
    {
        return 0;
    }
};

}

#endif
